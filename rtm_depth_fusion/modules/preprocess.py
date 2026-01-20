import typing as ty

import torch as tch
import torch.nn as nn
import torch.nn.functional as F

from rtm_depth_fusion.utils import extract_pcl_patches


class RootZGuesser(nn.Module):
    def __init__(self, enc_feats: ty.Tuple[int, ...] = (3, 64, 128, 256, 512)):
        super().__init__()

        self.conv_blocks = nn.ModuleList(
            nn.ModuleList(
                [
                    nn.Conv1d(enc_feats[i], enc_feats[i + 1], 3, padding=1),
                    nn.Conv1d(enc_feats[i + 1], enc_feats[i + 1], 3, padding=1),
                ]
            )
            for i in range(len(enc_feats) - 1)
        )

        self.fc = nn.Linear(enc_feats[-1], 1)

    def forward(self, point_cloud: tch.Tensor) -> tch.Tensor:
        """
        point_cloud: [B, N, 3]
        return:      [B, 1]
        """
        x = point_cloud.transpose(1, 2)

        for i, (conv1, conv2) in enumerate(self.conv_blocks):
            y = F.relu(conv1(x))
            z = conv2(y)
            # ReLU residual for all but the last block
            if i < len(self.conv_blocks) - 1:
                x = F.relu(y + z)
            else:
                x = z

        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.fc(x).view(-1, 1)

    def load_tf_weights(self, tf_weights):
        for i, (conv1, conv2) in enumerate(self.conv_blocks, start=1):
            # conv1
            w = tch.from_numpy(tf_weights.get_tensor(f"znet/conv{i * 2 - 1}/weights"))
            b = tch.from_numpy(tf_weights.get_tensor(f"znet/conv{i * 2 - 1}/biases"))
            # TF: [K, Cin, Cout]
            conv1.weight.copy_(w.permute(2, 1, 0).contiguous())
            conv1.bias.copy_(b)

            # conv2
            w = tch.from_numpy(tf_weights.get_tensor(f"znet/conv{i * 2}/weights"))
            b = tch.from_numpy(tf_weights.get_tensor(f"znet/conv{i * 2}/biases"))
            conv2.weight.copy_(w.permute(2, 1, 0).contiguous())
            conv2.bias.copy_(b)

            # ---- copy final fc ----
        fc_w = tch.from_numpy(tf_weights.get_tensor("znet/tfc1/weights"))
        fc_b = tch.from_numpy(tf_weights.get_tensor("znet/tfc1/biases"))
        self.fc.weight.copy_(fc_w.permute(1, 0).contiguous())
        self.fc.bias.copy_(fc_b)


class Preprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.z_guesser = RootZGuesser()

    @staticmethod
    def _expectation_1d(
        prob: tch.Tensor, sharpening_factor=100.0
    ) -> ty.Tuple[tch.Tensor, tch.Tensor, float]:
        """
        Compute soft-argmax expectation along the last axis.

        Args:
            prob: [B, K, N]  – probability distribution per keypoint.
        Returns:
            expectation: [B, K, 1]  – expected bin index.
        """
        prob_sharp = prob * tch.tensor(
            [sharpening_factor], device=prob.device, dtype=prob.dtype
        )
        # https://github.com/pytorch/pytorch/issues/136918
        bins = tch.arange(
            float(prob_sharp.shape[-1]), device=prob_sharp.device, dtype=prob_sharp.dtype
        )  # [N]
        prob_sharp = F.softmax(prob_sharp, dim=-1)
        expectation = (prob_sharp * bins).sum(-1, keepdim=True)  # [B, K, 1]
        conf, _indexes = tch.max(prob, dim=-1)
        return expectation, conf.clamp(0.01, 1.0), tch.tensor([bins.shape[0]], device=prob.device)

    def forward(
        self,
        depth: tch.Tensor,
        simcc_x: tch.Tensor,
        simcc_y: tch.Tensor,
        simcc_z: tch.Tensor,
        K_inv: tch.Tensor,
        bypass_root_center: ty.Optional[tch.Tensor] = None,
    ):
        # Enforce consistent hand scale at the cost of wrist inaccuracy, if we can't see the hand we don't really care where the wrist is anyway
        simcc_x[:, 9] = simcc_x[:, 91]
        simcc_y[:, 9] = simcc_y[:, 91]
        simcc_z[:, 9] = simcc_z[:, 91]
        simcc_x[:, 10] = simcc_x[:, 112]
        simcc_y[:, 10] = simcc_y[:, 112]
        simcc_z[:, 10] = simcc_z[:, 112]

        u_bin, u_conf, u_len = self._expectation_1d(simcc_x)  # [B, K, 1]
        v_bin, v_conf, v_len = self._expectation_1d(simcc_y)  # [B, K, 1]
        # TODO does using Z(D) give any performance improvements? or maybe just go back to RTMPose2D?
        z_bin, z_conf, z_len = self._expectation_1d(simcc_z)  # [B, K, 1]

        u_px = u_bin / 2.0  # [B, K, 1]
        v_px = v_bin / 2.0  # [B, K, 1]
        uvz = tch.cat(
            [u_px, v_px, tch.ones(u_px.shape, device=u_px.device, dtype=u_px.dtype)],
            dim=-1,
        )  # [B, K, 3]

        rays = tch.bmm(K_inv, uvz.transpose(2, 1)).transpose(1, 2)  # [B,K,3]

        torso_derived_from = tch.tensor([5, 6, 11, 12], device=uvz.device)
        torso_root = tch.mean(uvz[:, torso_derived_from, :], dim=1, keepdim=True)

        coco_main_kps = tch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 100, 121],
            device=uvz.device,
        )

        get_patches_for = tch.cat([uvz[:, coco_main_kps, :], torso_root], dim=1)
        all_pcl, _ = extract_pcl_patches(
            depth, get_patches_for[:, :, :2], K_inv
        )  # [B, 19 + 1, P, 3]

        torso_root_pcl = all_pcl[:, -1, :, :].squeeze(1)  # [B, P, 3]
        coco_main_pcl = all_pcl[:, :-1, :, :]  # [B, 19, P, 3]

        torso_root_mean = tch.mean(torso_root_pcl, dim=1, keepdim=True)
        torso_root_pcl_centered = torso_root_pcl - torso_root_mean
        z_root_offset = self.z_guesser(torso_root_pcl_centered)

        pred_torso_root = torso_root_mean.clone()
        pred_torso_root[:, :, 2] = pred_torso_root[:, :, 2] + z_root_offset
        torso_for_rest = pred_torso_root.clone()

        if bypass_root_center is not None:
            torso_for_rest = torso_root_mean.clone()
            torso_for_rest[:, :, 2] = bypass_root_center.to(
                device=torso_root_mean.device
            )[:, 2:]

        pose_init_centered = (
            rays[:, coco_main_kps, :] * torso_for_rest[:, :, 2:]
        ) - torso_for_rest

        # TODO these are the original box params but I want to re-tune them
        coco_main_pcl_norm = (coco_main_pcl - torso_for_rest.unsqueeze(1)) / tch.tensor(
            [1.8, 2, 1.5], device=coco_main_pcl.device, dtype=coco_main_pcl.dtype
        )

        return (
            coco_main_pcl_norm,
            pose_init_centered,
            rays[:, coco_main_kps, :],
            rays,
            pred_torso_root,
            torso_for_rest,
            tch.stack([u_conf, v_conf], dim=-1),
            u_px, v_px, z_bin / z_len,
        )
