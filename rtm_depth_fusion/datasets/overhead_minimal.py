import json
import os.path as osp
import typing as ty

import numpy as np
import torch as tch
import torch.nn.functional as F
from torch.utils.data import Dataset

DEFAULT_IMAGE_SIZE = (288, 384)  # (W, H) — portrait, matches EgoBody convention

# overhead keypoint index k → RTMPose3D 133-joint simcc index
_SIMCC_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 100, 121]

_N_X = 576   # W * 2 = 288 * 2
_N_Y = 768   # H * 2 = 384 * 2
_N_Z = 576   # matches EgoBody z bins


class OverheadItem(ty.NamedTuple):
    valid_entry: tch.Tensor
    depth: tch.Tensor
    K: tch.Tensor
    K_inv: tch.Tensor
    kps19_cam: tch.Tensor
    simcc_x: tch.Tensor
    simcc_y: tch.Tensor
    simcc_z: tch.Tensor
    image_id: ty.List[int]
    ann_id: ty.List[int]

    @staticmethod
    def collate(items: ty.List["OverheadItem"]) -> "OverheadItem":
        batch = [i for i in items if bool(tch.all(i.valid_entry))]
        if not batch:
            return _invalid_overhead
        out: ty.Dict[str, ty.Any] = {}
        for key in OverheadItem._fields:
            vals = [getattr(b, key) for b in batch]
            if OverheadItem.__annotations__[key] is tch.Tensor:
                out[key] = tch.cat(vals, dim=0)
            elif OverheadItem.__annotations__[key] is ty.List[int]:
                out[key] = sum(vals, [])
            else:
                raise NotImplementedError(f"un-collatable field {key}")
        return OverheadItem(**out)

    def to(self, *args, **kwargs) -> "OverheadItem":
        out = {}
        for key in OverheadItem._fields:
            v = getattr(self, key)
            out[key] = v.to(*args, **kwargs) if isinstance(v, tch.Tensor) else v
        return OverheadItem(**out)


# noinspection PyArgumentList
_invalid_overhead = OverheadItem(
    **{k: None for k in OverheadItem._fields if k != "valid_entry"},
    valid_entry=tch.tensor([False]),
)


def _depth_resize(depth: tch.Tensor, h_out: int, w_out: int) -> tch.Tensor:
    *batch, h, w = depth.shape
    x = depth.reshape(-1, 1, h, w)
    x = F.interpolate(x, size=(h_out, w_out), mode="bilinear", align_corners=False)
    return x.reshape(*batch, h_out, w_out)


def _build_K(fx: float, fy: float, cx: float, cy: float) -> tch.Tensor:
    return tch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=tch.float32
    )


def _simcc_spike(center_bin: float, n_bins: int, sigma: float = 2.5) -> tch.Tensor:
    bins = tch.arange(n_bins, dtype=tch.float32)
    return tch.exp(-0.5 * ((bins - center_bin) / sigma) ** 2)


class OverheadMinDataset(Dataset):
    def __init__(
        self,
        root: str,
        img_size: ty.Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ):
        super().__init__()
        self.root = root
        self.img_size = img_size  # (W, H)

        with open(osp.join(root, "calibration.json")) as f:
            cal = json.load(f)
        self._fx = float(cal["fx"])
        self._fy = float(cal["fy"])
        self._cx = float(cal["cx"])
        self._cy = float(cal["cy"])

        with open(osp.join(root, "annotations_3d.json")) as f:
            coco = json.load(f)

        self._entries = [
            (ann["image_id"], ann["id"], ann)
            for ann in coco["annotations"]
            if not ann.get("excluded", False)
        ]

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> OverheadItem:
        image_id, ann_id, ann = self._entries[idx]

        x0, y0, w, h = ann["bbox"]
        if w < 15 or h < 15:
            return _invalid_overhead

        depth_np = np.load(
            osp.join(self.root, "depth", f"{image_id:06d}.npy")
        ).astype(np.float32)
        depth = tch.from_numpy(depth_np).unsqueeze(0)  # [1, H_src, W_src]

        W_out, H_out = self.img_size
        depth_crop = depth[
            :,
            round(y0) : round(y0 + h),
            round(x0) : round(x0 + w),
        ]
        depth_resized = _depth_resize(depth_crop, H_out, W_out).unsqueeze(0)  # [1,1,H,W]

        sx = W_out / w
        sy = H_out / h
        K = _build_K(
            fx=self._fx * sx,
            fy=self._fy * sy,
            cx=(self._cx - x0) * sx,
            cy=(self._cy - y0) * sy,
        )
        K_inv = tch.linalg.inv(K)

        kps3d = np.array(ann["keypoints_3d"], dtype=np.float32).reshape(19, 4)
        kps19_cam = tch.from_numpy(kps3d[:, :3] / 1000.0).unsqueeze(0)  # mm → m, [1,19,3]

        # Synthesize SimCC heatmaps from 2D annotations
        kps2d = np.array(ann["keypoints"], dtype=np.float32).reshape(19, 3)  # [x, y, v]
        kps2d_out = kps2d.copy()
        kps2d_out[:, 0] = (kps2d[:, 0] - x0) * sx   # x in output space
        kps2d_out[:, 1] = (kps2d[:, 1] - y0) * sy   # y in output space

        n_x = W_out * 2
        n_y = H_out * 2
        n_z = _N_Z

        simcc_x = tch.zeros(1, 133, n_x)
        simcc_y = tch.zeros(1, 133, n_y)
        simcc_z = tch.zeros(1, 133, n_z)

        # Per-sample random linear transform for Z (teaches general depth, not exact scale)
        z_scale = float(np.random.uniform(0.7, 1.4))
        z_offset = float(np.random.uniform(-n_z * 0.1, n_z * 0.1))

        for k, simcc_idx in enumerate(_SIMCC_IDX):
            x_out, y_out, vis = kps2d_out[k]
            if vis <= 0:
                continue
            simcc_x[0, simcc_idx] = _simcc_spike(x_out * 2, n_x)
            simcc_y[0, simcc_idx] = _simcc_spike(y_out * 2, n_y)

            z_mm = float(kps3d[k, 2])  # already mm from dataset
            bin_z = float(np.clip(z_mm / 5000.0 * n_z * z_scale + n_z / 2 + z_offset, 0, n_z - 1))
            simcc_z[0, simcc_idx] = _simcc_spike(bin_z, n_z)

        return OverheadItem(
            valid_entry=tch.tensor([True]),
            depth=depth_resized.to(tch.int16),
            K=K.unsqueeze(0),
            K_inv=K_inv.unsqueeze(0),
            kps19_cam=kps19_cam,
            simcc_x=simcc_x,
            simcc_y=simcc_y,
            simcc_z=simcc_z,
            image_id=[image_id],
            ann_id=[ann_id],
        )
