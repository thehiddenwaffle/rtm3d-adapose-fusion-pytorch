#!/usr/bin/env python3
"""
Combine the RTMPose SimCC ONNX model with the DepthFusionNet ONNX post-processor.

The script maps the RTMPose SimCC outputs (simcc_x/y/z) to the similarly named
inputs in the depth fusion network and keeps the remaining depth fusion inputs
(`depth` and `camera_K_inv`) exposed so they can be provided at runtime.
"""

from __future__ import annotations

import argparse
from typing import Iterable

import onnx
import torch as tch
from onnx import TensorProto, compose
from onnx.compose import add_prefix

from rtm_depth_fusion import RTMPoseToAdaPose

EXPECTED_SIMCC_DTYPE = TensorProto.FLOAT16
EXPECTED_SIMCC_OUTPUT_SHAPES = {
    "simcc_x": [1, 133, 576],
    "simcc_y": [1, 133, 768],
    "simcc_z": [1, 133, 576],
}


def export_depth_fusion_net_to_onnx(model_path, output_onnx_path="_ada_post.onnx"):
    device = tch.device("cuda" if tch.cuda.is_available() else "cpu")

    post_model = RTMPoseToAdaPose()

    post_model.load_state_dict(tch.load(model_path, map_location=device)["model"])
    post_model.to(device)
    post_model.eval()

    batch_size = 1
    num_keypoints = 133
    depth_height, depth_width = 384, 288

    # Create example inputs
    depth = tch.randn(batch_size, 1, depth_height, depth_width, device=device)  # Depth map
    simcc_x = tch.randn(batch_size, num_keypoints, depth_width * 2, device=device)  # SimCC X heatmap
    simcc_y = tch.randn(batch_size, num_keypoints, depth_height * 2, device=device)  # SimCC Y heatmap
    simcc_z = tch.randn(batch_size, num_keypoints, depth_width * 2, device=device)  # SimCC Z heatmap
    camera_K_inv = tch.tensor([[[7.1048e-04, 0.0000e+00, 1.0357e-01],
                                [0.0000e+00, 1.2039e-03, -6.6613e-02],
                                [0.0000e+00, 0.0000e+00, 1.0000e+00]]],
                              device=device)  # Intrinsic inverse matrix(Note must be modified to be the crop of the input)
    camera_K_inv_squashed = tch.stack(
        (
            camera_K_inv[..., [0, 1], [0, 1]],  # [1/fx, 1/fy]
            camera_K_inv[..., [0, 1], 2],  # [-cx/fx, -cy/fy]
        ),
        dim=1,
    )

    # Export the model
    tch.onnx.export(
        post_model.half(),
        (depth.half(), simcc_x.half(), simcc_y.half(), simcc_z.half(), camera_K_inv_squashed.half()),
        output_onnx_path,
        export_params=True,
        # external_data=True,
        opset_version=11,
        input_names=["depth", "simcc_x", "simcc_y", "simcc_z", "camera_K_inv"],
        output_names=["kps_xyz", "dbg_torso_root_center_pred", "dbg_kp_pix_confidence", "dbg_kp_z_pred",
                      "dbg_px_coords", "dbg_z_prior"],
        # dynamo=True,
    )
    return onnx.load(output_onnx_path)


def combine_models(
        pose_path: str,
        ckpt_path: str,
        output_path: str,
        pose_output_names: Iterable[str],
) -> None:
    pose_model = onnx.load(pose_path)
    # If it's this easy why is there even an error on mismatch????????
    # pose_model.ir_version = 10

    depth_model = export_depth_fusion_net_to_onnx(ckpt_path)

    io_map = list(zip(pose_output_names, ("simcc_x", "simcc_y", "simcc_z")))

    merged = compose.merge_models(
        add_prefix(pose_model, "rtm_", rename_edges=False, rename_inputs=False, rename_outputs=False),
        add_prefix(depth_model, "ada_", rename_edges=False, rename_inputs=False, rename_outputs=False),
        io_map=io_map,
    )
    onnx.save(merged, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse RTMPose SimCC outputs with the depth fusion."
    )
    parser.add_argument(
        "--pose",
        default="./end2end_mmdeploy_rtmpose3d_subbed_ops.onnx",
        help="Path to the RTMPose SimCC ONNX file.",
    )
    parser.add_argument(
        "--ada-weights",
        default="../../ckpts_rtm_ada/epoch_026.pt",
        help="Path to the ada checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="rtm_ada_fusion.onnx",
        help="Path to write the combined ONNX graph.",
    )
    parser.add_argument(
        "--simcc-names",
        default=["output", "1338", "1340"],
        help="3 Names of the RTMPose output tensor that represents SimCC X.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combine_models(
        pose_path=args.pose,
        ckpt_path=args.ada_weights,
        output_path=args.output,
        pose_output_names=args.simcc_names,
    )
    print(f"Combined ONNX model written to {args.output}")


if __name__ == "__main__":
    main()
