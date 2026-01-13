import torch
import torch.nn as nn
import numpy as np

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
)

from projects.mmdet3d_plugin.core.box3d import *
from ..blocks import linear_relu_ln, SmoothNet

__all__ = [
    "SparseBox3DRefinementModule",
    "SparseBox3DKeyPointsGenerator",
    "SparseBox3DEncoder",
]


@POSITIONAL_ENCODING.register_module()
class SparseBox3DEncoder(BaseModule):
    def __init__(
        self,
        embed_dims,
        vel_dims=3,
        mode="add",
        output_fc=True,
        in_loops=1,
        out_loops=2,
    ):
        super().__init__()
        assert mode in ["add", "cat"]
        self.embed_dims = embed_dims
        self.vel_dims = vel_dims
        self.mode = mode

        def embedding_layer(input_dims, output_dims):
            return nn.Sequential(
                *linear_relu_ln(output_dims, in_loops, out_loops, input_dims)
            )

        if not isinstance(embed_dims, (list, tuple)):
            embed_dims = [embed_dims] * 5
        self.pos_fc = embedding_layer(3, embed_dims[0])
        self.size_fc = embedding_layer(3, embed_dims[1])
        self.yaw_fc = embedding_layer(2, embed_dims[2])
        if vel_dims > 0:
            self.vel_fc = embedding_layer(self.vel_dims, embed_dims[3])
        if output_fc:
            self.output_fc = embedding_layer(embed_dims[-1], embed_dims[-1])
        else:
            self.output_fc = None

    def forward(self, box_3d: torch.Tensor):
        pos_feat = self.pos_fc(box_3d[..., [X, Y, Z]])
        size_feat = self.size_fc(box_3d[..., [W, L, H]])
        yaw_feat = self.yaw_fc(box_3d[..., [SIN_YAW, COS_YAW]])
        if self.mode == "add":
            output = pos_feat + size_feat + yaw_feat
        elif self.mode == "cat":
            output = torch.cat([pos_feat, size_feat, yaw_feat], dim=-1)

        if self.vel_dims > 0:
            vel_feat = self.vel_fc(box_3d[..., VX : VX + self.vel_dims])
            if self.mode == "add":
                output = output + vel_feat
            elif self.mode == "cat":
                output = torch.cat([output, vel_feat], dim=-1)
        if self.output_fc is not None:
            output = self.output_fc(output)
        return output


@PLUGIN_LAYERS.register_module()
class SparseBox3DRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        output_dim=11,
        num_cls=10,
        normalize_yaw=False,
        refine_yaw=False,
        with_cls_branch=True,
        with_quality_estimation=False,
    ):
        super(SparseBox3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.num_cls = num_cls
        self.normalize_yaw = normalize_yaw
        self.refine_yaw = refine_yaw

        self.refine_state = [X, Y, Z, W, L, H]
        if self.refine_yaw:
            self.refine_state += [SIN_YAW, COS_YAW]

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )
        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, self.num_cls),
            )
        self.with_quality_estimation = with_quality_estimation
        if with_quality_estimation:
            self.quality_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, 2),
            )

    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor = 1.0,
        return_cls=True,
    ):
        feature = instance_feature + anchor_embed
        output = self.layers(feature)
        output[..., self.refine_state] = (
            output[..., self.refine_state] + anchor[..., self.refine_state]
        )
        if self.normalize_yaw:
            output[..., [SIN_YAW, COS_YAW]] = torch.nn.functional.normalize(
                output[..., [SIN_YAW, COS_YAW]], dim=-1
            )
        if self.output_dim > 8:
            if not isinstance(time_interval, torch.Tensor):
                time_interval = instance_feature.new_tensor(time_interval)
            translation = torch.transpose(output[..., VX:], 0, -1)
            velocity = torch.transpose(translation / time_interval, 0, -1)
            output[..., VX:] = velocity + anchor[..., VX:]

        if return_cls:
            assert self.with_cls_branch, "Without classification layers !!!"
            cls = self.cls_layers(instance_feature)
        else:
            cls = None
        if return_cls and self.with_quality_estimation:
            quality = self.quality_layers(feature)
        else:
            quality = None
        return output, cls, quality


@PLUGIN_LAYERS.register_module()
class SparseBox3DKeyPointsGenerator(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_learnable_pts=0,
        fix_scale=None,
        motion_models=None,
        with_acc_estimation=False,
        with_yaw_rate_estimation=False,
        acc_mode='smoothnet', #smoothnet, gaussian
        yaw_rate_mode='smoothnet', #smoothnet, gaussian
        fix_z=False,
        judge_reverse_yaw=False,
        temporal_window_size=1,
        acc_range=0.05,
        yaw_rate_range=0.05
    ):
        super(SparseBox3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = nn.Parameter(
            torch.tensor(fix_scale), requires_grad=False
        )
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = Linear(self.embed_dims, num_learnable_pts * 3)
        
        # multi motion hypothesis cfg
        self.fix_z = fix_z
        self.motion_models = motion_models
        self.judge_reverse_yaw = judge_reverse_yaw
        self.num_motion_model = len(self.motion_models)
        
        # predict acc and yaw rate
        self.with_acc_estimation = with_acc_estimation
        self.with_yaw_rate_estimation = with_yaw_rate_estimation
        self.acc_mode = acc_mode
        self.yaw_rate_mode = yaw_rate_mode
        
        if self.with_acc_estimation and self.acc_mode == 'smoothnet':
            self.acc_layers = SmoothNet(window_size=temporal_window_size, 
                                        output_size=1,
                                        state_size=1,
                                        feat_size=embed_dims)
        self.acc_range = acc_range
        
        if self.with_yaw_rate_estimation and self.yaw_rate_mode == 'smoothnet':
            self.yaw_rate_layers = SmoothNet(window_size=temporal_window_size, 
                                             output_size=1,
                                             state_size=1,
                                             feat_size=embed_dims)
        self.yaw_rate_range = yaw_rate_range

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)
    
    def acc_decode(self, x, feat):
        acc = torch.zeros_like(x[..., [X]])[..., None]
        acc = self.acc_layers(acc, feat[..., None]).squeeze(-1)
        return acc

    def yaw_rate_decode(self, x, feat):
        yaw_rate = torch.zeros_like(x[..., [X]])[..., None]
        yaw_rate = self.yaw_rate_layers(yaw_rate, feat[..., None]).squeeze(-1)
        return yaw_rate

    def pred_acc(self, anchor, feature):
        if self.with_acc_estimation:
            if self.acc_mode == 'smoothnet':
                cached_acc = self.acc_decode(anchor, feature)
            elif self.acc_mode == 'gaussian':
                cached_acc = torch.randn(size=anchor[..., [X]].shape, device=anchor.device) * self.acc_range
            else:
                raise NotImplementedError()
        else:
            cached_acc = None
        
        return cached_acc
    
    def pred_yaw_rate(self, anchor, feature):
        if self.with_yaw_rate_estimation:
            if self.yaw_rate_mode == 'smoothnet':
                cached_yaw_rate = self.yaw_rate_decode(anchor, feature)
            elif self.yaw_rate_mode == 'gaussian':
                cached_yaw_rate = torch.randn(size=anchor[..., [X]].shape, device=anchor.device) * self.yaw_rate_range
            else:
                raise NotImplementedError()
        else:
            cached_yaw_rate = None
        
        return cached_yaw_rate

    def forward(
        self,
        anchor,
        instance_feature=None,
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        bs, num_anchor = anchor.shape[:2]
        size = anchor[..., None, [W, L, H]].exp()
        key_points = self.fix_scale * size
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (
                self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3)
                .sigmoid()
                - 0.5
            )
            key_points = torch.cat(
                [key_points, learnable_scale * size], dim=-2
            )

        rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3])

        rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 2, 2] = 1

        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None]
        ).squeeze(-1)
        key_points = key_points + anchor[..., None, [X, Y, Z]]

        if (
            cur_timestamp is None
            or temp_timestamps is None
            or T_cur2temp_list is None
            or len(temp_timestamps) == 0
        ):
            return key_points

        temp_key_points_list = []
        velocity = anchor[..., VX:]
        for i, t_time in enumerate(temp_timestamps):
            time_interval = cur_timestamp - t_time
            translation = (
                velocity
                * time_interval.to(dtype=velocity.dtype)[:, None, None]
            )
            temp_key_points = key_points - translation[:, :, None]
            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)
            temp_key_points = (
                T_cur2temp[:, None, None, :3]
                @ torch.cat(
                    [
                        temp_key_points,
                        torch.ones_like(temp_key_points[..., :1]),
                    ],
                    dim=-1,
                ).unsqueeze(-1)
            )
            temp_key_points = temp_key_points.squeeze(-1)
            temp_key_points_list.append(temp_key_points)
        return key_points, temp_key_points_list

    def multi_motion_model_transform(self, time_interval,
                                        vel, center, yaw,
                                        acc=None, yaw_rate=None, 
                                        time_mask=None,
                                        instance_feature=None):
        """transform tracklet/detection from the source (history) timestamp to the target timestamp
        Note that, the base coordinate system (source timestamp frame) has not changed

        Args:
            time_interval: time interval from the source timestamp to the target timestamp [bs, ]
            vel: bbbox's velocity (in history frame), [bs, num_q, vel_dim]
            center: bbox's geometry center (in history frame), [bs, num_q, 3]
            yaw: heading angle (in history frame), [bs, num_q, 2 (sin_yaw, cos_yaw)]
            acc: overall acceleration `vector`. Defaults to None. [bs, num_q, 1]
            yaw_rate: over yaw (heading angle) rate. Defaults to None. [bs, num_q, 1]
            time_mask: the mask to filter invalid temporal instance (cross-scene), true for valid. [bs, ]
            instance_feature: the feature of instances, [bs, num_q, embed_dim]

        Returns:
            the predict geometry center, vel, yaw, acc, yaw_rate, [bs, num_q, num_model, state_dim]
        """
        centers, vels, yaws, accs, yaw_rates = [], [], [], None, None
        if acc is not None and yaw_rate is not None:
            accs, yaw_rates = [], []
        
        # delta_t = target_time - source_time, >0 forward pass, <0 backward pass
        delta_t = -time_interval
        
        # normalize yaw
        yaw_vector_length = torch.norm(yaw, p=2, dim=-1, keepdim=True)
        norm_yaw = nn.functional.normalize(yaw, dim=-1)
        
        for motion_i, motion_type in enumerate(self.motion_models):
            assert motion_type in MOTION_MODEL_TYPES, 'Unsupport motion model.'
            if motion_type == 'CV':
                translation = vel.transpose(0, -1) * delta_t
                translation = translation.transpose(0, -1)
                tmp_center = center + translation
                tmp_yaw = yaw
                tmp_velo = vel
            elif motion_type == 'CA':
                # velocity and yaw are decoupled, which are the main difference from CTRA/CTRV
                if acc is None or yaw_rate is None: continue
                acc_xyz = torch.cat([acc * norm_yaw[..., [1, 0]], torch.zeros_like(acc)], dim=-1)

                translation = vel.transpose(0, -1) * delta_t + 0.5 * acc_xyz.transpose(0, -1) * delta_t ** 2
                translation = translation.transpose(0, -1)
                tmp_center = center + translation

                translation_velo = acc_xyz.transpose(0, -1) * delta_t
                translation_velo = translation_velo.transpose(0, -1)
                tmp_velo = vel + translation_velo
                
                translation_yaw = yaw_rate.transpose(0, -1) * delta_t
                translation_yaw = translation_yaw.transpose(0, -1)
                overall_yaw = torch.atan2(yaw[..., 0], yaw[..., 1])[..., None]
                tmp_yaw = overall_yaw + translation_yaw
                tmp_yaw = torch.cat([torch.sin(tmp_yaw), torch.cos(tmp_yaw)], dim=-1) * yaw_vector_length
            elif motion_type == 'STATIC':
                tmp_center, tmp_yaw = center, yaw
                tmp_velo = vel * torch.zeros_like(vel)
            elif motion_type == 'CTRA':
                if acc is None or yaw_rate is None: continue
                
                # prevent 0-overflow
                yaw_rate = torch.where((yaw_rate < 0.01) & (yaw_rate > 0), 
                                       torch.tensor(0.01, dtype=yaw_rate.dtype, device=yaw_rate.device), yaw_rate)
                yaw_rate = torch.where((yaw_rate > -0.01) & (yaw_rate <= 0), 
                                       torch.tensor(-0.01, dtype=yaw_rate.dtype, device=yaw_rate.device), yaw_rate)
                
                translation_yaw = yaw_rate.transpose(0, -1) * delta_t
                translation_yaw = translation_yaw.transpose(0, -1)
                overall_yaw = torch.atan2(yaw[..., 0], yaw[..., 1])[..., None]
                tmp_yaw = overall_yaw + translation_yaw
                tmp_yaw = torch.cat([torch.sin(tmp_yaw), torch.cos(tmp_yaw)], dim=-1) 

                # determine the direction of movement (forward < 60 degree)
                velo_xy = vel[..., :2]
                overall_velo = torch.norm(velo_xy, p=2, dim=-1, keepdim=True)
                
                if self.judge_reverse_yaw:
                    is_backward = nn.functional.cosine_similarity(yaw, velo_xy, dim=-1)[..., None] < 0.5
                    overall_velo[is_backward] = -overall_velo[is_backward]
                
                translation_velo = acc.transpose(0, -1) * delta_t
                translation_velo = translation_velo.transpose(0, -1)
                tmp_velo = overall_velo + translation_velo
                
                ry_rate_inv_square = 1.0 / (yaw_rate * yaw_rate)
                
                tmp_x = center[..., 0, None] + ry_rate_inv_square * (tmp_velo * yaw_rate * tmp_yaw[..., 0, None] + acc * tmp_yaw[..., 1, None] - overall_velo * yaw_rate * norm_yaw[..., 0, None] - acc * norm_yaw[..., 1, None])
                tmp_y = center[..., 1, None] + ry_rate_inv_square * (-tmp_velo * yaw_rate * tmp_yaw[..., 1, None] + acc * tmp_yaw[..., 0, None] + overall_velo * yaw_rate * norm_yaw[..., 1, None] - acc * norm_yaw[..., 0, None])
                
                if not self.fix_z:
                    translation_z = vel[..., 2, None].transpose(0, -1) * delta_t
                    translation_z = translation_z.transpose(0, -1)
                    tmp_z = center[..., 2, None] + translation_z
                else:
                    tmp_z = center[..., 2, None]
                tmp_center = torch.cat([tmp_x, tmp_y, tmp_z], dim=-1)
                tmp_velo = torch.cat([tmp_velo * tmp_yaw[..., [1, 0]], vel[..., 2, None]], dim=-1)
                tmp_yaw = tmp_yaw * yaw_vector_length
            
            elif motion_type == 'CTRV':
                if yaw_rate is None: continue
                # prevent 0-overflow
                yaw_rate = torch.where((yaw_rate < 0.01) & (yaw_rate > 0), 
                                       torch.tensor(0.01, dtype=yaw_rate.dtype, device=yaw_rate.device), yaw_rate)
                yaw_rate = torch.where((yaw_rate > -0.01) & (yaw_rate <= 0), 
                                       torch.tensor(-0.01, dtype=yaw_rate.dtype, device=yaw_rate.device), yaw_rate)
                
                translation_yaw = yaw_rate.transpose(0, -1) * delta_t
                translation_yaw = translation_yaw.transpose(0, -1)
                overall_yaw = torch.atan2(yaw[..., 0], yaw[..., 1])[..., None]
                tmp_yaw = overall_yaw + translation_yaw
                tmp_yaw = torch.cat([torch.sin(tmp_yaw), torch.cos(tmp_yaw)], dim=-1)
                
                # determine the direction of movement (forward < 60 degree)
                velo_xy = vel[..., :2]
                overall_velo = torch.norm(velo_xy, p=2, dim=-1, keepdim=True)
                
                if self.judge_reverse_yaw:
                    is_backward = nn.functional.cosine_similarity(yaw, velo_xy, dim=-1)[..., None] < 0.5
                    overall_velo[is_backward] = -overall_velo[is_backward]
                
                tmp_velo = overall_velo
                
                ry_rate_inv = 1.0 / (yaw_rate)
                tmp_x = center[..., 0, None] + ry_rate_inv * (overall_velo * tmp_yaw[..., 0, None] - overall_velo * norm_yaw[..., 0, None])
                tmp_y = center[..., 1, None] + ry_rate_inv * (-overall_velo * tmp_yaw[..., 1, None] + overall_velo * norm_yaw[..., 1, None])
                
                if not self.fix_z:
                    translation_z = vel[..., 2, None].transpose(0, -1) * delta_t
                    translation_z = translation_z.transpose(0, -1)
                    tmp_z = center[..., 2, None] + translation_z
                else:
                    tmp_z = center[..., 2, None]
                tmp_center = torch.cat([tmp_x, tmp_y, tmp_z], dim=-1)
                tmp_velo = torch.cat([tmp_velo * tmp_yaw[..., [1, 0]], vel[..., 2, None]], dim=-1)
                tmp_yaw = tmp_yaw * yaw_vector_length
            
            centers.append(tmp_center[None, ...])
            vels.append(tmp_velo[None, ...])
            yaws.append(tmp_yaw[None, ...])
            
            if acc is not None and yaw_rate is not None:
                accs.append(acc[None, ...])
                yaw_rates.append(yaw_rate[None, ...])
        
        centers = torch.cat(centers, dim=0).permute(1, 2, 0, 3)
        vels = torch.cat(vels, dim=0).permute(1, 2, 0, 3)
        yaws = torch.cat(yaws, dim=0).permute(1, 2, 0, 3)
        if acc is not None and yaw_rate is not None:
            accs = torch.cat(accs, dim=0).permute(1, 2, 0, 3)
            yaw_rates = torch.cat(yaw_rates, dim=0).permute(1, 2, 0, 3)
        
        return centers, vels, yaws, accs, yaw_rates


    def multi_motion_anchor_projection(
        self,
        anchor,
        T_src2dst_list,
        acc=None,
        yaw_rate=None,
        src_timestamp=None,
        dst_timestamps=None,
        time_intervals=None,
        time_mask=None,
        instance_feature=None,
    ):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            vel = anchor[..., VX:]
            vel_dim = vel.shape[-1]
            T_src2dst = torch.unsqueeze(
                T_src2dst_list[i].to(dtype=anchor.dtype), dim=1
            )

            center = anchor[..., [X, Y, Z]]
            if time_intervals is not None:
                time_interval = time_intervals[i]
            elif src_timestamp is not None and dst_timestamps is not None:
                time_interval = (src_timestamp - dst_timestamps[i]).to(
                    dtype=vel.dtype
                )
            else:
                time_interval = None
            if time_interval is not None:
                multi_center, multi_vel, multi_yaw, _, _ = self.multi_motion_model_transform(time_interval, 
                                                                                             vel, center, 
                                                                                             yaw=anchor[..., [SIN_YAW, COS_YAW]],
                                                                                             acc=acc, yaw_rate=yaw_rate,
                                                                                             time_mask=time_mask,
                                                                                             instance_feature=instance_feature)
            multi_center = (
                torch.matmul(
                    T_src2dst[..., None, :3, :3], multi_center[..., None]
                ).squeeze(dim=-1)
                + T_src2dst[..., None, :3, 3]
            )
            size = anchor[..., None, [W, L, H]].tile((1, 1, self.num_motion_model, 1))
            # multi_yaw: [0: sin, 1: cos] -> [0: cos, 1: sin]
            multi_yaw = torch.matmul(
                T_src2dst[..., None, :2, :2], multi_yaw[..., [1, 0], None],
            ).squeeze(-1)
            # yaw: [0: cos, 1: sin] -> [0: sin, 1: cos]
            multi_yaw = multi_yaw[..., [1, 0]]
            multi_vel = torch.matmul(
                T_src2dst[..., None, :vel_dim, :vel_dim], multi_vel[..., None]
            ).squeeze(-1)
            # [bs, num_q, num_model, state_dim]

            dst_anchor = torch.cat([multi_center, size, multi_yaw, multi_vel], dim=-1)
            dst_anchors.append(dst_anchor)
        
        return dst_anchors

    @staticmethod
    def anchor_projection(
        anchor,
        T_src2dst_list,
        src_timestamp=None,
        dst_timestamps=None,
        time_intervals=None,
    ):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            vel = anchor[..., VX:]
            vel_dim = vel.shape[-1]
            T_src2dst = torch.unsqueeze(
                T_src2dst_list[i].to(dtype=anchor.dtype), dim=1
            )

            center = anchor[..., [X, Y, Z]]
            if time_intervals is not None:
                time_interval = time_intervals[i]
            elif src_timestamp is not None and dst_timestamps is not None:
                time_interval = (src_timestamp - dst_timestamps[i]).to(
                    dtype=vel.dtype
                )
            else:
                time_interval = None
            if time_interval is not None:
                translation = vel.transpose(0, -1) * time_interval
                translation = translation.transpose(0, -1)
                center = center - translation
            center = (
                torch.matmul(
                    T_src2dst[..., :3, :3], center[..., None]
                ).squeeze(dim=-1)
                + T_src2dst[..., :3, 3]
            )
            size = anchor[..., [W, L, H]]
            yaw = torch.matmul(
                T_src2dst[..., :2, :2],
                anchor[..., [COS_YAW, SIN_YAW], None],
            ).squeeze(-1)
            yaw = yaw[..., [1,0]]
            vel = torch.matmul(
                T_src2dst[..., :vel_dim, :vel_dim], vel[..., None]
            ).squeeze(-1)
            dst_anchor = torch.cat([center, size, yaw, vel], dim=-1)
            dst_anchors.append(dst_anchor)
        return dst_anchors

    @staticmethod
    def distance(anchor):
        return torch.norm(anchor[..., :2], p=2, dim=-1)
