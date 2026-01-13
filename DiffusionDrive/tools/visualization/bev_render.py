import os
import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from diffusers.schedulers import DDIMScheduler
import torch
CMD_LIST = ['Turn Right', 'Turn Left', 'Go Straight']
COLOR_VECTORS = ['cornflowerblue', 'royalblue', 'slategrey']
SCORE_THRESH = 0.3
MAP_SCORE_THRESH = 0.3
color_mapping = np.asarray([
    [0, 0, 0],
    [255, 179, 0],
    [128, 62, 117],
    [255, 104, 0],
    [166, 189, 215],
    [193, 0, 32],
    [206, 162, 98],
    [129, 112, 102],
    [0, 125, 52],
    [246, 118, 142],
    [0, 83, 138],
    [255, 122, 92],
    [83, 55, 122],
    [255, 142, 0],
    [179, 40, 81],
    [244, 200, 0],
    [127, 24, 13],
    [147, 170, 0],
    [89, 51, 21],
    [241, 58, 19],
    [35, 44, 22],
    [112, 224, 255],
    [70, 184, 160],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [0, 255, 235],
    [255, 0, 235],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 255, 204],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [255, 214, 0],
    [25, 194, 194],
    [92, 0, 255],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
]) / 255


class BEVRender:
    def __init__(
        self, 
        plot_choices,
        out_dir,
        xlim = 40,
        ylim = 40,
    ):
        self.plot_choices = plot_choices
        self.xlim = xlim
        self.ylim = ylim
        self.gt_dir = os.path.join(out_dir, "bev_gt")
        self.pred_dir = os.path.join(out_dir, "bev_pred")
        os.makedirs(self.gt_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)
        # self.vocabulary = np.load('/home/users/bencheng.liao/PlanWrapper/data/kmeans/kmeans_plan_4096.npy')
        self.vocabulary = np.load('/home/users/bencheng.liao/PlanWrapper/data/kmeans/kmeans_plan_6.npy')
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

    def reset_canvas(self):
        plt.close()
        self.fig, self.axes = plt.subplots(1, 1, figsize=(10, 20))
        self.axes.set_xlim(-17, 17)
        self.axes.set_ylim(-32, 32)
        self.axes.axis('off')

    def render(
        self,
        data, 
        result,
        index,
    ):
        self.reset_canvas()
        self.draw_detection_gt(data)
        # self.draw_motion_gt(data)
        self.draw_map_gt(data)
        # self.draw_planning_gt(data)
        # self.draw_planning_anchor(data)
        self.draw_diffusing_gt(data)
        # self.draw_denoised_anchor(data)
        # self.draw_planning_vocabulary(data)
        # self.draw_planning_anchor_v2(data)
        # self.draw_single_anchor(data)
        self._render_sdc_car()
        # self._render_command(data)
        # self._render_legend()
        save_path_gt = os.path.join(self.gt_dir, str(index).zfill(4) + '.png')
        self.save_fig(save_path_gt)
        import ipdb; ipdb.set_trace()


        return save_path_gt

    def save_fig(self, filename):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(filename,format="png",dpi=800)

    def draw_planning_anchor(self, data):
        if not self.plot_choices['planning']:
            return
        # import ipdb; ipdb.set_trace()
        # Add zero as starting point for each trajectory
        vocabulary = self.vocabulary.reshape(18, 6, 2)
        plan_vocabulary = np.concatenate((
            np.zeros((vocabulary.shape[0], 1, vocabulary.shape[-1])), 
            vocabulary), 
            axis=1
        )
        
        # Use different colormaps for variety
        colormaps = [
            # Sequential colormaps
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu',
            'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            # Diverging colormaps
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
            # Cyclic colormaps
            'twilight', 'twilight_shifted', 'hsv',
            # Qualitative colormaps
            'Pastel1', 'Pastel2', 'Paired', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c'
        ]
        
        # Draw each trajectory with a different colormap
        # for i, traj in enumerate(plan_vocabulary):
        i = 2
        traj = plan_vocabulary[i]
        colormap = colormaps[i % len(colormaps)]
        # Use lower opacity (0.3) to avoid overwhelming the visualization
        self._render_traj(
            traj, 
            traj_score=1.0,
            colormap=colormap, 
            dot_size=50,
            points_per_step=40,

        )

    def draw_detection_gt(self, data):
        if not self.plot_choices['det']:
            return

        for i in range(data['gt_labels_3d'].shape[0]):
            label = data['gt_labels_3d'][i]
            if label == -1: 
                continue
            color = color_mapping[i % len(color_mapping)]

            # draw corners
            corners = box3d_to_corners(data['gt_bboxes_3d'])[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

            # draw line to indicate forward direction
            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

    def draw_detection_pred(self, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['det'] and "boxes_3d" in result):
            return

        bboxes = result['boxes_3d']
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH: 
                continue
            color = color_mapping[result['instance_ids'][i] % len(color_mapping)]

            # draw corners
            corners = box3d_to_corners(bboxes)[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

            # draw line to indicate forward direction
            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

    def draw_track_pred(self, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['track'] and "anchor_queue" in result):
            return
        
        temp_bboxes = result["anchor_queue"]
        period = result["period"]
        bboxes = result['boxes_3d']
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH: 
                continue
            color = color_mapping[result['instance_ids'][i] % len(color_mapping)]
            center = bboxes[i, :3]
            centers = [center]
            for j in range(period[i]):
                # draw corners
                corners = box3d_to_corners(temp_bboxes[:, -1-j])[i, [0, 3, 7, 4, 0]]
                x = corners[:, 0]
                y = corners[:, 1]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle='-')

                # draw line to indicate forward direction
                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                x = [forward_center[0], center[0]]
                y = [forward_center[1], center[1]]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle='-')
                centers.append(center)

            centers = np.stack(centers)
            xs = centers[:, 0]
            ys = centers[:, 1]
            self.axes.plot(xs, ys, color=color, linewidth=2, linestyle='-')

    def draw_motion_gt(self, data):
        if not self.plot_choices['motion']:
            return

        for i in range(data['gt_labels_3d'].shape[0]):
            label = data['gt_labels_3d'][i]
            if label == -1: 
                continue
            color = color_mapping[i % len(color_mapping)]
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25

            center = data['gt_bboxes_3d'][i, :2]
            masks = data['gt_agent_fut_masks'][i].astype(bool)
            if masks[0] == 0:
                continue
            trajs = data['gt_agent_fut_trajs'][i][masks]
            trajs = trajs.cumsum(axis=0) + center
            trajs = np.concatenate([center.reshape(1, 2), trajs], axis=0)
            
            self._render_traj(trajs, traj_score=1.0,
                            colormap='winter', dot_size=dot_size)

    def draw_motion_pred(self, result, top_k=3):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['motion'] and "trajs_3d" in result):
            return
        
        bboxes = result['boxes_3d']
        labels = result['labels_3d']
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH: 
                continue
            label = labels[i]
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25

            traj_score = result['trajs_score'][i].numpy()
            traj = result['trajs_3d'][i].numpy()
            num_modes = len(traj_score)
            center = bboxes[i, :2][None, None].repeat(num_modes, 1, 1).numpy()
            traj = np.concatenate([center, traj], axis=1)

            sorted_ind = np.argsort(traj_score)[::-1]
            sorted_traj = traj[sorted_ind, :, :2]
            sorted_score = traj_score[sorted_ind]
            norm_score = np.exp(sorted_score[0])

            for j in range(top_k - 1, -1, -1):
                viz_traj = sorted_traj[j]
                traj_score = np.exp(sorted_score[j])/norm_score
                self._render_traj(viz_traj, traj_score=traj_score,
                                colormap='winter', dot_size=dot_size)
    
    def draw_map_gt(self, data):
        if not self.plot_choices['map']:
            return
        vectors = data['map_infos']
        for label, vector_list in vectors.items():
            color = COLOR_VECTORS[label]
            for vector in vector_list:
                pts = vector[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                self.axes.plot(x, y, color=color, linewidth=3, marker='o', linestyle='-', markersize=7)

    def draw_map_pred(self, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['map'] and "vectors" in result):
            return

        for i in range(result['scores'].shape[0]):
            score = result['scores'][i]
            if  score < MAP_SCORE_THRESH:
                continue
            color = COLOR_VECTORS[result['labels'][i]]
            pts = result['vectors'][i]
            x = pts[:, 0]
            y = pts[:, 1]
            plt.plot(x, y, color=color, linewidth=3, marker='o', linestyle='-', markersize=7)

    def draw_planning_gt(self, data):
        if not self.plot_choices['planning']:
            return

        # draw planning gt
        masks = data['gt_ego_fut_masks'].astype(bool)
        if masks[0] != 0:
            plan_traj = data['gt_ego_fut_trajs'][masks]
            cmd = data['gt_ego_fut_cmd']
            plan_traj[abs(plan_traj) < 0.01] = 0.0
            plan_traj = plan_traj.cumsum(axis=0)
            plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
            self._render_traj(plan_traj, traj_score=1.0,
                colormap='autumn', dot_size=50)
    def normalize_ego_fut_trajs(self, gt_ego_fut_trajs):
        # bs, ego_fut_ts, _ = gt_ego_fut_trajs.shape
        odo_info_fut_x = gt_ego_fut_trajs[..., 0:1]
        odo_info_fut_y = gt_ego_fut_trajs[..., 1:2]

        odo_info_fut_x = odo_info_fut_x / 3
        odo_info_fut_x = odo_info_fut_x.clamp(-1, 1)
        odo_info_fut_y = (odo_info_fut_y+0.5) / 8.1
        odo_info_fut_y = odo_info_fut_y.clamp(0, 1)
        odo_info_fut_y = odo_info_fut_y * 2 - 1
        odo_info_fut = torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)
        # odo_info_fut = odo_info_fut.reshape(-1,self.ego_fut_ts, 2)
        return odo_info_fut
    def denormalize_ego_fut_trajs(self, noisy_traj_points):
        # bs, ego_fut_ts, _ = noisy_traj_points.shape
        odo_info_fut_x = noisy_traj_points[..., 0:1]
        odo_info_fut_y = noisy_traj_points[..., 1:2]

        odo_info_fut_x = odo_info_fut_x * 3
        # odo_info_fut_x = odo_info_fut_x.clamp(-1, 1)
        odo_info_fut_y = (odo_info_fut_y+1) / 2 * 8.1 - 0.5
        # odo_info_fut_y = odo_info_fut_y.clamp(-1, 1)
        odo_info_fut = torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)
        return odo_info_fut
    def draw_diffusing_gt(self,data):
        if not self.plot_choices['planning']:
            return
        masks = data['gt_ego_fut_masks'].astype(bool)
        # import ipdb; ipdb.set_trace()
        plan_traj = data['gt_ego_fut_trajs'][masks]
        plan_traj = torch.from_numpy(plan_traj)
        odo_info_fut = self.normalize_ego_fut_trajs(plan_traj)
        odo_info_fut = odo_info_fut.unsqueeze(0)
        odo_info_fut = odo_info_fut.repeat(200,1,1)
        noise = torch.randn(odo_info_fut.shape)

        timesteps = 400
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=noise.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(noise.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(odo_info_fut.shape[0])
        noisy_traj_points = self.diffusion_scheduler.add_noise(
            original_samples=odo_info_fut,
            noise=noise,
            timesteps=timesteps,
        ).float()

        denorm_traj = self.denormalize_ego_fut_trajs(noisy_traj_points)
        denorm_traj = denorm_traj.numpy()
        # import ipdb; ipdb.set_trace()
        plan_traj = denorm_traj.cumsum(axis=1)
        plan_traj = np.concatenate((np.zeros((plan_traj.shape[0],1, plan_traj.shape[2])), plan_traj), axis=1)
        for i, traj in enumerate(plan_traj):
            self._render_traj(traj, traj_score=1.0,
                colormap='autumn', dot_size=500, points_per_step=3)
          


    def draw_planning_vocabulary_v2(self, data):
        if not self.plot_choices['planning']:
            return

        # Add zero as starting point for each trajectory
        plan_vocabulary = np.concatenate((
            np.zeros((self.vocabulary.shape[0], 1, self.vocabulary.shape[-1])), 
            self.vocabulary), 
            axis=1
        )
        
        # Use a wide variety of colormaps for better distinction
        colormaps = [
            # Sequential colormaps
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu',
            'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            # Diverging colormaps
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
            # Cyclic colormaps
            'twilight', 'twilight_shifted', 'hsv',
            # Qualitative colormaps
            'Pastel1', 'Pastel2', 'Paired', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c'
        ]
        
        # Group trajectories by similarity to reduce visual clutter
        # We'll sample every Nth trajectory to show a representative subset
        sampling_rate = 32  # Show every 32nd trajectory
        sampled_indices = range(0, len(plan_vocabulary), sampling_rate)
        
        for i, traj_idx in enumerate(sampled_indices):
            traj = plan_vocabulary[traj_idx]
            colormap = colormaps[i % len(colormaps)]
            
            # Vary opacity based on trajectory length to highlight different patterns
            traj_length = np.linalg.norm(traj[-1] - traj[0])
            opacity = min(0.8, max(0.2, traj_length / 40.0))  # Scale opacity with length
            
            # Vary dot size based on trajectory position
            # Further trajectories get smaller dots
            max_dist = np.max(np.abs(traj))
            dot_size = max(10, min(30, 40 - max_dist))
            
            self._render_traj(
                traj, 
                traj_score=opacity,
                colormap=colormap, 
                dot_size=dot_size,
                points_per_step=10  # Reduced points for cleaner visualization
            )

    def draw_planning_vocabulary(self, data):
        if not self.plot_choices['planning']:
            return

        # Add zero as starting point for each trajectory
        plan_vocabulary = np.concatenate((
            np.zeros((self.vocabulary.shape[0], 1, self.vocabulary.shape[-1])), 
            self.vocabulary), 
            axis=1
        )
        
        # Use different colormaps for variety
        colormaps = [
            # Sequential colormaps
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu',
            'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            # Diverging colormaps
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
            # Cyclic colormaps
            'twilight', 'twilight_shifted', 'hsv',
            # Qualitative colormaps
            'Pastel1', 'Pastel2', 'Paired', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c'
        ]
        
        # Draw each trajectory with a different colormap
        for i, traj in enumerate(plan_vocabulary):
            colormap = colormaps[i % len(colormaps)]
            # Use lower opacity (0.3) to avoid overwhelming the visualization
            self._render_traj(
                traj, 
                traj_score=0.6,
                colormap=colormap, 
                dot_size=25
            )
    def draw_planning_anchor(self, data):
        if not self.plot_choices['planning']:
            return
        # import ipdb; ipdb.set_trace()
        # Add zero as starting point for each trajectory
        vocabulary = self.vocabulary.reshape(18, 6, 2)
        plan_vocabulary = np.concatenate((
            np.zeros((vocabulary.shape[0], 1, vocabulary.shape[-1])), 
            vocabulary), 
            axis=1
        )
        
        # Use different colormaps for variety
        colormaps = [
            # Sequential colormaps
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu',
            'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            # Diverging colormaps
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
            # Cyclic colormaps
            'twilight', 'twilight_shifted', 'hsv',
            # Qualitative colormaps
            'Pastel1', 'Pastel2', 'Paired', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c'
        ]
        
        # Draw each trajectory with a different colormap
        for i, traj in enumerate(plan_vocabulary):
            colormap = colormaps[i % len(colormaps)]
            # Use lower opacity (0.3) to avoid overwhelming the visualization
            self._render_traj(
                traj, 
                traj_score=1.0,
                colormap=colormap, 
                dot_size=50,
                points_per_step=40,

            )

    def draw_planning_anchor_v2(self, data):
        if not self.plot_choices['planning']:
            return

        # Add zero as starting point for each trajectory
        vocabulary = self.vocabulary.reshape(18, 6, 2)
        plan_vocabulary = np.concatenate((
            np.zeros((vocabulary.shape[0], 1, vocabulary.shape[-1])), 
            vocabulary), 
            axis=1
        )
        
        # Use different colormaps for variety
        colormaps = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            # ... (keep other colormaps)
        ]
        
        # Parameters for noise generation
        num_noise_trajectories = 100  # Number of noise trajectories per anchor
        noise_std = 2.0  # Standard deviation of noise
        noise_opacity = 0.5  # Opacity for noise points
        
        # Draw each anchor trajectory and its noise
        for i, anchor_traj in enumerate(plan_vocabulary):
            colormap = colormaps[i % len(colormaps)]
            
            # Generate noise trajectories
            for _ in range(num_noise_trajectories):
                # Create noise that increases with distance from start
                noise_scale = np.linspace(0, 1, len(anchor_traj))[:, np.newaxis]
                noise = np.random.normal(0, noise_std, anchor_traj.shape) * noise_scale
                noise_traj = anchor_traj + noise
                
                # Draw noise trajectory with lower opacity
                self._render_traj(
                    noise_traj,
                    traj_score=noise_opacity,
                    colormap=colormap,
                    dot_size=40,
                    points_per_step=5
                )
            
            # Draw the main anchor trajectory on top
            self._render_traj(
                anchor_traj,
                traj_score=1.0,
                colormap=colormap,
                dot_size=50,
                points_per_step=20
            )
            line_color = matplotlib.colormaps[colormap](0.8)[:3]  # Using 0.8 to get a slightly darker shade
            self.axes.plot(
                anchor_traj[:, 0],  # x coordinates
                anchor_traj[:, 1],  # y coordinates
                color=line_color,
                linewidth=8,
                linestyle='-',
                zorder=3  # Make sure line appears above the noise points
            )

    def draw_diffusing_anchor(self,data):
        if not self.plot_choices['planning']:
            return
        vocabulary = self.vocabulary.reshape(18, 6, 2)
        plan_vocabulary = np.concatenate((
            np.zeros((vocabulary.shape[0], 1, vocabulary.shape[-1])), 
            vocabulary), 
            axis=1
        )
        colormaps = [
            # Sequential colormaps
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu',
            'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            # Diverging colormaps
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
            # Cyclic colormaps
            'twilight', 'twilight_shifted', 'hsv',
            # Qualitative colormaps
            'Pastel1', 'Pastel2', 'Paired', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c'
        ]
        # Parameters for noise generation
        num_noise_trajectories = 200  # Number of noise trajectories per anchor
        noise_std = 0.5  # or 2.0 for laerger # Standard deviation of noise
        noise_opacity = 1.0  # Opacity for noise points
        for i, anchor_traj in enumerate(plan_vocabulary):
            colormap = colormaps[i % len(colormaps)]
            
            # Generate noise trajectories
            for _ in range(num_noise_trajectories):
                # Create noise that increases with distance from start
                noise_scale = np.linspace(0, 2, len(anchor_traj))[:, np.newaxis]
                noise = np.random.normal(0, noise_std, anchor_traj.shape) * noise_scale
                noise_traj = anchor_traj + noise
                
                # Draw noise trajectory with lower opacity
                self._render_traj(
                    noise_traj,
                    traj_score=noise_opacity,
                    colormap=colormap,
                    dot_size=40,
                    points_per_step=5
                )

    def draw_denoised_anchor(self,data):
        if not self.plot_choices['planning']:
            return
        vocabulary = self.vocabulary.reshape(18, 6, 2)
        plan_vocabulary = np.concatenate((
            np.zeros((vocabulary.shape[0], 1, vocabulary.shape[-1])), 
            vocabulary), 
            axis=1
        )
        masks = data['gt_ego_fut_masks'].astype(bool)
        if masks[0] != 0:
            gt_plan_traj = data['gt_ego_fut_trajs'][masks]
            cmd = data['gt_ego_fut_cmd']
            gt_plan_traj[abs(gt_plan_traj) < 0.01] = 0.0
            gt_plan_traj = gt_plan_traj.cumsum(axis=0)
            gt_plan_traj = np.concatenate((np.zeros((1, gt_plan_traj.shape[1])), gt_plan_traj), axis=0)
        
        colormaps = [
            # Sequential colormaps
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu',
            'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            # Diverging colormaps
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
            # Cyclic colormaps
            'twilight', 'twilight_shifted', 'hsv',
            # Qualitative colormaps
            'Pastel1', 'Pastel2', 'Paired', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c'
        ]
        # Parameters for noise generation
        num_noise_trajectories = 200  # Number of noise trajectories per anchor
        noise_std = 0.5  # Standard deviation of noise
        
        for i, anchor_traj in enumerate(plan_vocabulary):
            colormap = colormaps[i % len(colormaps)]
            
            # Calculate opacity based on distance between anchor and ground truth
            if masks[0] != 0:
                dist = np.linalg.norm(anchor_traj - gt_plan_traj, axis=1).mean()
                # Set opacity to 1.0 for closest anchor, 0.5 for others
                if i == np.argmin([np.linalg.norm(traj - gt_plan_traj, axis=1).mean() for traj in plan_vocabulary]):
                    anchor_opacity = 1.0
                    zorder = 3
                    colormap = 'autumn'
                else:
                    anchor_opacity = 0.6
                    zorder = 2
            else:
                anchor_opacity = 0.3  # Default opacity if no ground truth
            
            # Generate noise trajectories
            for _ in range(num_noise_trajectories):
                # Create noise that increases with distance from start
                noise_scale = np.linspace(0, 2, len(anchor_traj))[:, np.newaxis]
                noise = np.random.normal(0, noise_std, anchor_traj.shape) * noise_scale
                noise_traj = anchor_traj + noise
                
                # Draw noise trajectory with anchor's opacity
                self._render_traj(
                    noise_traj,
                    traj_score=anchor_opacity,
                    colormap=colormap,
                    dot_size=40,
                    points_per_step=5,
                    zorder = zorder
                )

    def draw_planning_pred(self, data, result, top_k=3):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['planning'] and "planning" in result):
            return

        if self.plot_choices['track'] and "ego_anchor_queue" in result:
            ego_temp_bboxes = result["ego_anchor_queue"]
            ego_period = result["ego_period"]
            for j in range(ego_period[0]):
                # draw corners
                corners = box3d_to_corners(ego_temp_bboxes[:, -1-j])[0, [0, 3, 7, 4, 0]]
                x = corners[:, 0]
                y = corners[:, 1]
                self.axes.plot(x, y, color='mediumseagreen', linewidth=2, linestyle='-')

                # draw line to indicate forward direction
                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                x = [forward_center[0], center[0]]
                y = [forward_center[1], center[1]]
                self.axes.plot(x, y, color='mediumseagreen', linewidth=2, linestyle='-')
        # import ipdb; ipdb.set_trace()
        plan_trajs = result['planning'].cpu().numpy()
        num_cmd = len(CMD_LIST)
        num_mode = plan_trajs.shape[1]
        plan_trajs = np.concatenate((np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2)
        plan_score = result['planning_score'].cpu().numpy()

        cmd = data['gt_ego_fut_cmd'].argmax()
        plan_trajs = plan_trajs[cmd]
        plan_score = plan_score[cmd]

        sorted_ind = np.argsort(plan_score)[::-1]
        sorted_traj = plan_trajs[sorted_ind, :, :2]
        sorted_score = plan_score[sorted_ind]
        norm_score = np.exp(sorted_score[0])

        for j in range(top_k - 1, -1, -1):
            viz_traj = sorted_traj[j]
            traj_score = np.exp(sorted_score[j]) / norm_score
            self._render_traj(viz_traj, traj_score=traj_score,
                            colormap='autumn', dot_size=50)

    def _render_traj(
        self, 
        future_traj, 
        traj_score=1, 
        colormap='winter', 
        points_per_step=20, 
        dot_size=25,
        zorder = 2
    ):
        total_steps = (len(future_traj) - 1) * points_per_step + 1
        dot_colors = matplotlib.colormaps[colormap](
            np.linspace(0, 1, total_steps))[:, :3]
        dot_colors = dot_colors * traj_score + \
            (1 - traj_score) * np.ones_like(dot_colors)
        total_xy = np.zeros((total_steps, 2))
        for i in range(total_steps - 1):
            unit_vec = future_traj[i // points_per_step +
                                   1] - future_traj[i // points_per_step]
            total_xy[i] = (i / points_per_step - i // points_per_step) * \
                unit_vec + future_traj[i // points_per_step]
        total_xy[-1] = future_traj[-1]
        self.axes.scatter(
            total_xy[:, 0], total_xy[:, 1], c=dot_colors, s=dot_size,zorder=zorder)

    def _render_sdc_car(self):
        sdc_car_png = cv2.imread('resources/sdc_car.png')
        sdc_car_png = cv2.cvtColor(sdc_car_png, cv2.COLOR_BGR2RGB)
        im = self.axes.imshow(sdc_car_png, extent=(-1, 1, -2, 2))
        im.set_zorder(3)

    def _render_legend(self):
        legend = cv2.imread('resources/legend.png')
        legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
        self.axes.imshow(legend, extent=(15, 40, -40, -30))

    def _render_command(self, data):
        cmd = data['gt_ego_fut_cmd'].argmax()
        self.axes.text(-38, -38, CMD_LIST[cmd], fontsize=60)