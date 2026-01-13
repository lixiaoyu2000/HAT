import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from projects.mmdet3d_plugin.core.box3d import *
from .blocks import MLN

__all__ = ["DetInstanceBank", "MapInstanceBank"]


def topk(confidence, k, *inputs):
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (
        indices + torch.arange(bs, device=indices.device)[:, None] * N
    ).reshape(-1)
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs


@PLUGIN_LAYERS.register_module()
class DetInstanceBank(nn.Module):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
        cached_feat_ego_align=False,
        cached_anchor_ego_align=False,
    ):
        super(DetInstanceBank, self).__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval

        if anchor_handler is not None:
            anchor_handler = build_from_cfg(anchor_handler, PLUGIN_LAYERS)
            assert hasattr(anchor_handler, "anchor_projection")
        self.anchor_handler = anchor_handler
        
        self.acc_range = self.anchor_handler.acc_range
        self.yaw_rate_range = self.anchor_handler.yaw_rate_range
        self.num_motion_model = self.anchor_handler.num_motion_model
        self.mutli_motion_list = self.anchor_handler.motion_models
        self.anchor_handler = anchor_handler
        self.cached_feat_ego_align = cached_feat_ego_align
        self.cached_anchor_ego_align = cached_anchor_ego_align
        if self.cached_feat_ego_align:
            self.ego_pose_memory = MLN(144)
        
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        if len(anchor.shape) == 3: # for map
            anchor = anchor.reshape(anchor.shape[0], -1)
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        self.reset()

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        self.metas = None
        self.mask = None
        self.confidence = None
        self.temp_confidence = None
        self.instance_id = None
        self.prev_id = 0
        self.T_temp2cur = None
    
    def get_mh(self, batch_size, metas=None, dn_metas=None):
        # feature and anchor at the current timestamp
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))
        self.batch_size = batch_size

        if (
            self.cached_anchor is not None
            and batch_size == self.cached_anchor.shape[0]
        ):
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            self.mask = torch.abs(time_interval) <= self.max_time_interval

            if self.anchor_handler is not None:
                T_temp2cur = self.cached_anchor.new_tensor(
                    np.stack(
                        [
                            x["T_global_inv"]
                            @ self.metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(metas["img_metas"])
                        ]
                    )
                )
                self.T_temp2cur = T_temp2cur
                
                # model acceleration
                cached_acc = self.anchor_handler.pred_acc(self.cached_anchor, self.cached_feature)
                
                # model yaw rate
                cached_yaw_rate = self.anchor_handler.pred_yaw_rate(self.cached_anchor, self.cached_feature)

                # [bs, num_q, num_model, state_dim]
                temp_cached_anchor = self.anchor_handler.multi_motion_anchor_projection(
                    self.cached_anchor,
                    [T_temp2cur],
                    acc=None if cached_acc is None else torch.clip(cached_acc, min=-self.acc_range, max=self.acc_range),
                    yaw_rate=None if cached_yaw_rate is None else torch.clip(cached_yaw_rate, min=-self.yaw_rate_range, max=self.yaw_rate_range),
                    time_intervals=[-time_interval],
                    instance_feature=self.cached_feature
                )[0]

                
                # The definition of 'instance feature' in sparse4Dv2: 
                # The instance feature is a high-order semantic feature extracted from the image,
                # mainly from the image encoder.
                temp_cached_feature = self.cached_feature[:, :, None, :].tile(1, 1, self.num_motion_model, 1)
                
                # origin sparse4D have invalid cached instance for adjacent batch sampling
                # but they don't process structured feature (anchor) of cached instance.
                # In contrast, we sample feature of these predict instance by their anchor,
                # which will cause catastrophic predictions without timestamp limitation
                temp_time_mask = self.mask
                if temp_time_mask.sum() != batch_size:
                    # if condition can be removed, it's only for speedup
                    temp_time_interval = torch.where(
                        temp_time_mask,
                        time_interval,
                        time_interval.new_tensor(self.default_time_interval),
                    )
                    T_cur2cur = anchor.new_tensor(
                            np.stack(
                                [
                                    x["T_global_inv"] 
                                    @ x["T_global"] 
                                    for x in metas["img_metas"]
                                ])
                        )
                    
                    anchor_acc = self.anchor_handler.pred_acc(anchor[:, :self.num_temp_instances], instance_feature[:, :self.num_temp_instances])
                    anchor_yaw_rate = self.anchor_handler.pred_yaw_rate(anchor[:, :self.num_temp_instances], instance_feature[:, :self.num_temp_instances])

                    temp_curr_anchor = self.anchor_handler.multi_motion_anchor_projection(
                        anchor[:, :self.num_temp_instances],
                        [T_cur2cur],
                        acc=None if anchor_acc is None else torch.clip(anchor_acc, min=-self.acc_range, max=self.acc_range),
                        yaw_rate=None if anchor_yaw_rate is None else torch.clip(anchor_yaw_rate, min=-self.yaw_rate_range, max=self.yaw_rate_range),
                        time_intervals=[-temp_time_interval],
                        instance_feature=instance_feature[:, :self.num_temp_instances]
                    )[0]

                    temp_cached_anchor = torch.where(
                        temp_time_mask[:, None, None, None],
                        temp_cached_anchor,
                        temp_curr_anchor,
                    )
                    temp_cached_feature = torch.where(
                        temp_time_mask[:, None, None, None],
                        temp_cached_feature,
                        instance_feature[:, :, None, :].tile(1, 1, self.num_motion_model, 1)[:, :self.num_temp_instances]
                    )

                    self.cached_anchor = torch.where(
                        temp_time_mask[:, None, None],
                        self.cached_anchor,
                        anchor[:, :self.num_temp_instances],
                    )

                    self.cached_feature = torch.where(
                        temp_time_mask[:, None, None],
                        self.cached_feature,
                        instance_feature[:, :self.num_temp_instances]
                    )

                    self.T_temp2cur = torch.where(
                        temp_time_mask[:, None, None],
                        self.T_temp2cur,
                        T_cur2cur
                    )
                    
                    # self.confidence[temp_time_mask] = 0
                
                if self.cached_feat_ego_align:
                    _T_temp2cur = self.T_temp2cur[:, None, ...].tile(1, self.num_temp_instances, 1, 1)
                    memory_ego_motion = torch.cat([_T_temp2cur[..., :3, :].flatten(-2)], dim=-1)
                    memory_ego_motion = self.nerf_positional_encoding(memory_ego_motion)
                    temp_cached_feature = self.ego_pose_memory(temp_cached_feature[..., 0, :], memory_ego_motion)
                    temp_cached_feature = temp_cached_feature[:, :, None, :].tile(1, 1, self.num_motion_model, 1)

            if (
                self.anchor_handler is not None
                and dn_metas is not None
                and batch_size == dn_metas["dn_anchor"].shape[0]
            ):
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2),
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]
                dn_metas["dn_anchor"] = dn_anchor.reshape(
                    batch_size, num_dn_group, num_dn, -1
                )
            time_interval = torch.where(
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                time_interval.new_tensor(self.default_time_interval),
            )
        else:
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )
            
            # the logic below is just for parallel training
            anchor_acc = self.anchor_handler.pred_acc(anchor, instance_feature)
            anchor_yaw_rate = self.anchor_handler.pred_yaw_rate(anchor, instance_feature)
                
            T_cur2cur = anchor.new_tensor(
                np.stack(
                    [
                        x["T_global_inv"] 
                        @ x["T_global"] 
                        for x in metas["img_metas"]
                    ])
                )
            self.T_temp2cur = T_cur2cur
            
            temp_cached_anchor = self.anchor_handler.multi_motion_anchor_projection(
                anchor,
                [T_cur2cur],
                acc=None if anchor_acc is None else torch.clip(anchor_acc, min=-self.acc_range, max=self.acc_range),
                yaw_rate=None if anchor_yaw_rate is None else torch.clip(anchor_yaw_rate, min=-self.yaw_rate_range, max=self.yaw_rate_range),
                time_intervals=[-time_interval],
                instance_feature=instance_feature
                )[0][:, :self.num_temp_instances]
        
            if self.cached_feat_ego_align:
                _T_temp2cur = self.T_temp2cur[:, None, ...].tile(1, self.num_temp_instances, 1, 1)
                memory_ego_motion = torch.cat([_T_temp2cur[..., :3, :].flatten(-2)], dim=-1)
                memory_ego_motion = self.nerf_positional_encoding(memory_ego_motion)
                temp_cached_feature = self.ego_pose_memory(instance_feature[:, :self.num_temp_instances], memory_ego_motion)
                temp_cached_feature = temp_cached_feature[:, :, None, :].tile(1, 1, self.num_motion_model, 1)
            else:
                temp_cached_feature = instance_feature[:, :, None, :].tile(1, 1, self.num_motion_model, 1)[:, :self.num_temp_instances]

        return (
            instance_feature,  # from self.instance_feature -> [bs, num_q(900), num_embed]
            anchor,  # from self.anchor -> [bs, num_q(900), state_dim]
            temp_cached_feature, # from last timestamp -> [bs, num_temp(600), num_model, num_embed]
            temp_cached_anchor, # from last timestamp prediction -> [bs, num_temp(600), num_model, state_dim]
            time_interval, # [bz,]
        )


    def get(self, batch_size, metas=None, dn_metas=None):
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        if (
            self.cached_anchor is not None
            and batch_size == self.cached_anchor.shape[0]
        ):
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            self.mask = torch.abs(time_interval) <= self.max_time_interval

            if self.anchor_handler is not None:
                T_temp2cur = self.cached_anchor.new_tensor(
                    np.stack(
                        [
                            x["T_global_inv"]
                            @ self.metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(metas["img_metas"])
                        ]
                    )
                )
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor,
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]

            if (
                self.anchor_handler is not None
                and dn_metas is not None
                and batch_size == dn_metas["dn_anchor"].shape[0]
            ):
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2),
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]
                dn_metas["dn_anchor"] = dn_anchor.reshape(
                    batch_size, num_dn_group, num_dn, -1
                )
            time_interval = torch.where(
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                time_interval.new_tensor(self.default_time_interval),
            )
        else:
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )

        return (
            instance_feature,
            anchor,
            self.cached_feature,
            self.cached_anchor,
            time_interval,
        )

    def update(self, instance_feature, anchor, confidence):
        if self.cached_feature is None:
            return instance_feature, anchor

        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        N = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk(
            confidence, N, instance_feature, anchor
        )
        selected_feature = torch.cat(
            [self.cached_feature, selected_feature], dim=1
        )
        selected_anchor = torch.cat(
            [self.cached_anchor, selected_anchor], dim=1
        )
        # instance_feature = torch.where(
        #     self.mask[:, None, None], selected_feature, instance_feature
        # )
        # anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        instance_feature = selected_feature
        anchor = selected_anchor
        self.confidence = torch.where(
            self.mask[:, None],
            self.confidence,
            self.confidence.new_tensor(0)
        )
        if self.instance_id is not None:
            self.instance_id = torch.where(
                self.mask[:, None],
                self.instance_id,
                self.instance_id.new_tensor(-1),
            )

        if num_dn > 0:
            instance_feature = torch.cat(
                [instance_feature, dn_instance_feature], dim=1
            )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

    def cache_temp(
        self,
        temp_instance_feature,
        temp_anchor,
    ):
        # To ensure compatibility with the original sparse4D, 
        if self.cached_feature is None:
            return
        self.cached_feature = temp_instance_feature
        self.cached_anchor = temp_anchor
        
    def get_cache_temp(self):
        return self.cached_feature, self.cached_anchor

    def get_cache_temp_feat(self):
        # encode the ego pose
        T_temp2cur = self.T_temp2cur[:, None, ...].tile(1, self.num_temp_instances, 1, 1)
        memory_ego_motion = torch.cat([T_temp2cur[..., :3, :].flatten(-2)], dim=-1)
        memory_ego_motion = self.nerf_positional_encoding(memory_ego_motion)
        
        if self.cached_feature is None:
            instance_feature = torch.tile(self.instance_feature[None], (self.batch_size, 1, 1))[:, :self.num_temp_instances, :]
            temp_cached_feature = self.ego_pose_memory(instance_feature, memory_ego_motion)
            return temp_cached_feature
        else:
            temp_cached_feature = self.ego_pose_memory(self.cached_feature, memory_ego_motion)
            return temp_cached_feature
    
    def get_cache_temp_anchor(self):
        if self.cached_anchor is None:
            return self.cached_anchor
        else:
            if 'CV' in self.mutli_motion_list:
                static_idx = self.mutli_motion_list.index('CV')
            elif 'CA' in self.mutli_motion_list:
                static_idx = self.mutli_motion_list.index('CA')
            
            time_interval = self.cached_anchor.new_tensor(
                [0.0] * self.batch_size
            )
            temp_cached_anchor = self.anchor_handler.multi_motion_anchor_projection(
                self.cached_anchor,
                [self.T_temp2cur],
                acc=torch.randn(size=self.cached_anchor[..., [X]].shape, device=self.cached_anchor.device) * 0.1, 
                yaw_rate=torch.randn(size=self.cached_anchor[..., [X]].shape, device=self.cached_anchor.device) * 0.1,
                time_intervals=[time_interval],
                )[0][..., static_idx, :]
            return temp_cached_anchor

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        self.metas = metas
        confidence = confidence.max(dim=-1).values.sigmoid()
        if self.confidence is not None:
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
        self.temp_confidence = confidence

        (
            self.confidence,
            (self.cached_feature, self.cached_anchor),
        ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)

    def get_instance_id(self, confidence, anchor=None, threshold=None):
        confidence = confidence.max(dim=-1).values.sigmoid()
        instance_id = confidence.new_full(confidence.shape, -1).long()

        if (
            self.instance_id is not None
            and self.instance_id.shape[0] == instance_id.shape[0]
        ):
            instance_id[:, : self.instance_id.shape[1]] = self.instance_id

        mask = instance_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)
        num_new_instance = mask.sum()
        new_ids = torch.arange(num_new_instance).to(instance_id) + self.prev_id
        instance_id[torch.where(mask)] = new_ids
        self.prev_id += num_new_instance
        self.update_instance_id(instance_id, confidence)
        return instance_id

    def update_instance_id(self, instance_id=None, confidence=None):
        if self.temp_confidence is None:
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence
        instance_id = topk(temp_conf, self.num_temp_instances, instance_id)[1][
            0
        ]
        instance_id = instance_id.squeeze(dim=-1)
        self.instance_id = F.pad(
            instance_id,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )
    
    @staticmethod
    def nerf_positional_encoding(tensor, num_encoding_functions=6, include_input=False, log_sampling=True):
        # TESTED
        # Trivially, the input tensor is added to the positional encoding.
        encoding = [tensor] if include_input else []
        frequency_bands = None
        if log_sampling:
            frequency_bands = 2.0 ** torch.linspace(
                0.0,
                num_encoding_functions - 1,
                num_encoding_functions,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        else:
            frequency_bands = torch.linspace(
                2.0 ** 0.0,
                2.0 ** (num_encoding_functions - 1),
                num_encoding_functions,
                dtype=tensor.dtype,
                device=tensor.device,
            )

        for freq in frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)


@PLUGIN_LAYERS.register_module()
class MapInstanceBank(nn.Module):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
    ):
        super(MapInstanceBank, self).__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval

        if anchor_handler is not None:
            anchor_handler = build_from_cfg(anchor_handler, PLUGIN_LAYERS)
            assert hasattr(anchor_handler, "anchor_projection")
        self.anchor_handler = anchor_handler
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        if len(anchor.shape) == 3: # for map
            anchor = anchor.reshape(anchor.shape[0], -1)
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        self.reset()

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        self.metas = None
        self.mask = None
        self.confidence = None
        self.temp_confidence = None
        self.instance_id = None
        self.prev_id = 0

    def get(self, batch_size, metas=None, dn_metas=None):
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        if (
            self.cached_anchor is not None
            and batch_size == self.cached_anchor.shape[0]
        ):
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            self.mask = torch.abs(time_interval) <= self.max_time_interval

            if self.anchor_handler is not None:
                T_temp2cur = self.cached_anchor.new_tensor(
                    np.stack(
                        [
                            x["T_global_inv"]
                            @ self.metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(metas["img_metas"])
                        ]
                    )
                )
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor,
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]

            if (
                self.anchor_handler is not None
                and dn_metas is not None
                and batch_size == dn_metas["dn_anchor"].shape[0]
            ):
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2),
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]
                dn_metas["dn_anchor"] = dn_anchor.reshape(
                    batch_size, num_dn_group, num_dn, -1
                )
            time_interval = torch.where(
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                time_interval.new_tensor(self.default_time_interval),
            )
        else:
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )

        return (
            instance_feature,
            anchor,
            self.cached_feature,
            self.cached_anchor,
            time_interval,
        )

    def update(self, instance_feature, anchor, confidence):
        if self.cached_feature is None:
            return instance_feature, anchor

        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        N = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk(
            confidence, N, instance_feature, anchor
        )
        selected_feature = torch.cat(
            [self.cached_feature, selected_feature], dim=1
        )
        selected_anchor = torch.cat(
            [self.cached_anchor, selected_anchor], dim=1
        )
        instance_feature = torch.where(
            self.mask[:, None, None], selected_feature, instance_feature
        )
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        self.confidence = torch.where(
            self.mask[:, None],
            self.confidence,
            self.confidence.new_tensor(0)
        )
        if self.instance_id is not None:
            self.instance_id = torch.where(
                self.mask[:, None],
                self.instance_id,
                self.instance_id.new_tensor(-1),
            )

        if num_dn > 0:
            instance_feature = torch.cat(
                [instance_feature, dn_instance_feature], dim=1
            )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        self.metas = metas
        confidence = confidence.max(dim=-1).values.sigmoid()
        if self.confidence is not None:
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
        self.temp_confidence = confidence

        (
            self.confidence,
            (self.cached_feature, self.cached_anchor),
        ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)

    def get_instance_id(self, confidence, anchor=None, threshold=None):
        confidence = confidence.max(dim=-1).values.sigmoid()
        instance_id = confidence.new_full(confidence.shape, -1).long()

        if (
            self.instance_id is not None
            and self.instance_id.shape[0] == instance_id.shape[0]
        ):
            instance_id[:, : self.instance_id.shape[1]] = self.instance_id

        mask = instance_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)
        num_new_instance = mask.sum()
        new_ids = torch.arange(num_new_instance).to(instance_id) + self.prev_id
        instance_id[torch.where(mask)] = new_ids
        self.prev_id += num_new_instance
        self.update_instance_id(instance_id, confidence)
        return instance_id

    def update_instance_id(self, instance_id=None, confidence=None):
        if self.temp_confidence is None:
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence
        instance_id = topk(temp_conf, self.num_temp_instances, instance_id)[1][
            0
        ]
        instance_id = instance_id.squeeze(dim=-1)
        self.instance_id = F.pad(
            instance_id,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )