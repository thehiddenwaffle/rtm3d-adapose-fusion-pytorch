import torch as tch
import torch.nn as nn
import typing as ty

from modules import PCLSampler, PointNet, Preprocessor


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
        K_inv: tch.Tensor,
        bypass_root_center: ty.Optional[tch.Tensor] = None,
    ):
        B, _, H, W = depth.shape
        coco_main_pcl_norm, pose_init_centered, pred_torso_root, inf_torso_root, _ = (
            self.pre(depth, simcc_x, simcc_y, simcc_z, K_inv, bypass_root_center)
        )
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
        z_len_residual = self.fc(pre_fc)

        coco_main_metric_xyz = pose_init_centered.clone() + inf_torso_root
        coco_main_metric_xyz[:, :, 2] = coco_main_metric_xyz[:, :, 2] + z_len_residual

        # TODO norm based(?) cluster lifting

        return coco_main_metric_xyz, pred_torso_root
