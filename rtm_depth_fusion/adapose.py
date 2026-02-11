import torch as tch
import torch.nn as nn
import typing as ty

from .modules import PCLSampler, PointNet, Preprocessor


class RTMPoseToAdaPose(nn.Module):
    def __init__(self, n_joints=17 + 2, lstm=False):
        super().__init__()

        self.n_joints = n_joints
        self.lstm = lstm
        self.pre = Preprocessor()
        self.sampler = PCLSampler()
        self.encoder = PointNet()

        if self.lstm:
            # TODO
            raise NotImplementedError("LSTM TODO")
            # self.pre_fc = nn.LSTM()
        else:
            self.pre_fc = nn.Linear(1024 + (3 * self.n_joints), 256)

        self.fc = nn.Linear(256, self.n_joints)

    def forward(
        self,
        depth: tch.Tensor,
        simcc_x: tch.Tensor,
        simcc_y: tch.Tensor,
        simcc_z: tch.Tensor,
        K_inv_dense: tch.Tensor,
        bypass_root_center: ty.Optional[tch.Tensor] = None,
    ):
        B, _, H, W = depth.shape
        # DepthAI will supply mm, we want to train and output m
        depth = ( depth / 1000.0 ).to(dtype=simcc_x.dtype)
        coco_main_pcl_norm, pose_init_centered, rays_19, rays_all, pred_torso_root, inf_torso_root, uv_conf, u_px, v_px, zd_prior = (
            self.pre(depth, simcc_x, simcc_y, simcc_z, K_inv_dense, bypass_root_center)
        )
        dbg_px_uv = tch.cat([u_px, v_px], dim=-1)
        if not self.lstm:
            # Create time frame dimension
            coco_main_pcl_norm = coco_main_pcl_norm.unsqueeze(1)
            t_step = 1
        pcl_cleaned = self.sampler(coco_main_pcl_norm)
        # TODO double check the encoder with fresh brain
        pcl_embedding = self.encoder(pcl_cleaned)
        pre_fc = self.pre_fc(
            tch.cat([pcl_embedding, pose_init_centered.view(B, -1)], dim=-1)
        )
        z_len_residual = self.fc(pre_fc) * 1.5

        pred_z = inf_torso_root[:, :, 2] + z_len_residual
        coco_main_metric_xyz = rays_19 * pred_z.unsqueeze(-1)

        assert bool(tch.all(zd_prior[:, 91, 0] == zd_prior[:, 9, 0]))

        # TODO These could be merged into one clean set to reduce nodes by half
        left_wrist_d, left_mcp_d = tch.norm(coco_main_metric_xyz[:, 9:10], dim=-1, keepdim=True), tch.norm(coco_main_metric_xyz[:, 17:18], dim=-1, keepdim=True)
        left_hand_metric_over_px = tch.abs(left_wrist_d - left_mcp_d) / tch.abs(zd_prior[:, 9:10] - zd_prior[:, 100:101])

        left_wrist_to_kp_euclidean_dist_in_px = zd_prior[:, 91:112] - zd_prior[:, 91:92] # 21
        left_wrist_to_kp_euclidean_dist_in_metric = ( left_wrist_to_kp_euclidean_dist_in_px * left_hand_metric_over_px ) + left_wrist_d

        # Normalized rays magnitude == 1, direction is preserved, so when you multiply these by D(NOT Z) you get real XYZ with magnitude D
        left_rays_norm = rays_all[:, 91:112, :] / tch.norm(rays_all[:, 91:112, :], dim =-1, keepdim=True)
        left_hand_metric_xyz = left_rays_norm * left_wrist_to_kp_euclidean_dist_in_metric

        # assert tch.allclose(left_hand_metric_xyz[:, 0], coco_main_metric_xyz[:, 9])

        right_wrist_d, right_mcp_d = tch.norm(coco_main_metric_xyz[:, 10:11], dim=-1, keepdim=True), tch.norm(coco_main_metric_xyz[:, 18:19], dim=-1, keepdim=True)
        right_hand_metric_over_px = tch.abs(right_wrist_d - right_mcp_d) / (zd_prior[:, 10:11] - zd_prior[:, 121:122])

        right_wrist_to_kp_euclidean_dist_in_px = zd_prior[:, 112:133] - zd_prior[:, 112:113] # 21
        right_wrist_to_kp_euclidean_dist_in_metric = ( right_wrist_to_kp_euclidean_dist_in_px * right_hand_metric_over_px ) + right_wrist_d

        # Normalized rays magnitude == 1, direction is preserved, so when you multiply these by D(NOT Z) you get real XYZ with magnitude D
        right_rays_norm = rays_all[:, 112:133, :] / tch.norm(rays_all[:, 112:133, :], dim =-1, keepdim=True)
        right_hand_metric_xyz = right_rays_norm * right_wrist_to_kp_euclidean_dist_in_metric

        # assert tch.allclose(right_hand_metric_xyz[:, 0], coco_main_metric_xyz[:, 10])

        # TODO IDC
        face_kp = rays_all[:, 23:91] * tch.mean(coco_main_metric_xyz[:, :5, 2:3]) # 68
        l_foot_kp = rays_all[:, 17:20] * coco_main_metric_xyz[:, 15:16, 2:3]
        r_foot_kp = rays_all[:, 20:23] * coco_main_metric_xyz[:, 16:17, 2:3]

        coco_metric_xyz = tch.cat([coco_main_metric_xyz[:, :17], l_foot_kp, r_foot_kp, face_kp, left_hand_metric_xyz, right_hand_metric_xyz], dim=1)

        return coco_metric_xyz, pred_torso_root, uv_conf, z_len_residual, dbg_px_uv, zd_prior
