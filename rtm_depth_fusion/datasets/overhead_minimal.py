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

# Keypoint local indices excluded from bbox derivation (lower-leg / feet)
_BBOX_EXCLUDE = frozenset({13, 14, 15, 16})
_BBOX_KPS = np.array([i not in _BBOX_EXCLUDE for i in range(19)])


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


def _rotate_depth(depth: tch.Tensor, angle_deg: float, cx: float, cy: float,
                  W: int, H: int) -> tch.Tensor:
    """Rotate [1,1,H,W] depth around pixel (cx,cy) by angle_deg (true pixel-space rotation).

    affine_grid normalises x over W and y over H independently, so a naive rotation
    in that space shears when W != H.  Scaling the off-diagonal terms by ar=H/W
    and 1/ar corrects for the aspect ratio.
    """
    theta = np.radians(angle_deg)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    cx_n = 2.0 * cx / W - 1.0
    cy_n = 2.0 * cy / H - 1.0
    ar = float(H) / float(W)
    mat = tch.tensor([[c,       s * ar,  cx_n * (1 - c) - cy_n * s * ar],
                      [-s / ar, c,       cy_n * (1 - c) + cx_n * s / ar]],
                     dtype=tch.float32).unsqueeze(0)
    grid = F.affine_grid(mat, depth.shape, align_corners=False)
    return F.grid_sample(depth.float(), grid, mode="bilinear",
                         align_corners=False, padding_mode="zeros")


def _rotate_K(K_np: np.ndarray, angle_deg: float) -> tch.Tensor:
    """Return new 3×3 intrinsic matrix after in-plane rotation by angle_deg."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    cx, cy = K_np[0, 2], K_np[1, 2]
    H_img = np.array([[c, -s, cx * (1 - c) + cy * s],
                      [s,  c, cy * (1 - c) - cx * s],
                      [0,  0, 1]], dtype=np.float32)
    R3d_T = np.array([[c,  s, 0],
                      [-s, c, 0],
                      [0,  0, 1]], dtype=np.float32)
    return tch.from_numpy(H_img @ K_np @ R3d_T)


def _simcc_spike(center_bin: float, n_bins: int, sigma: float = 2.5) -> tch.Tensor:
    bins = tch.arange(n_bins, dtype=tch.float32)
    return tch.exp(-0.5 * ((bins - center_bin) / sigma) ** 2)


class OverheadMinDataset(Dataset):
    def __init__(
        self,
        root: str,
        img_size: ty.Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        augment: bool = True,
    ):
        super().__init__()
        self.root = root
        self.img_size = img_size  # (W, H)
        self.augment = augment

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

        depth_np = np.load(
            osp.join(self.root, "depth", f"{image_id:06d}.npy")
        ).astype(np.float32)
        H_full, W_full = depth_np.shape

        kps3d = np.array(ann["keypoints_3d"], dtype=np.float32).reshape(19, 4)
        kps2d = np.array(ann["keypoints"],    dtype=np.float32).reshape(19, 3)

        # derive bbox from visible keypoints, excluding lower-leg/feet (kps 13-16)
        vis = (kps2d[:, 2] > 0) & _BBOX_KPS
        if not vis.any():
            return _invalid_overhead
        px, py = kps2d[vis, 0], kps2d[vis, 1]
        span = max(float(px.max() - px.min()), float(py.max() - py.min()), 1.0)
        pad = span * 0.1
        x0 = float(np.clip(px.min() - pad, 0, W_full))
        y0 = float(np.clip(py.min() - pad, 0, H_full))
        x1 = float(np.clip(px.max() + pad, 0, W_full))
        y1 = float(np.clip(py.max() + pad, 0, H_full))
        w, h = x1 - x0, y1 - y0
        if w < 15 or h < 15:
            return _invalid_overhead

        depth = tch.from_numpy(depth_np).unsqueeze(0)  # [1, H_src, W_src]
        W_out, H_out = self.img_size

        K_full_np = np.array(
            [[self._fx, 0.0, self._cx],
             [0.0, self._fy, self._cy],
             [0.0,      0.0,      1.0]], dtype=np.float32)

        if self.augment:
            angle_deg = float(np.random.uniform(-80.0, 80.0))
            theta = np.radians(angle_deg)
            c, s = float(np.cos(theta)), float(np.sin(theta))

            # 1. rotate full depth frame around full-frame principal point
            depth_full = depth.unsqueeze(0)  # [1,1,H_full,W_full]
            depth = _rotate_depth(
                depth_full, angle_deg, self._cx, self._cy, W_full, H_full
            )[0]  # [1,H_full,W_full]

            # 2. rotate 3D keypoints (Z unchanged)
            kps3d_rot = kps3d.copy()
            kps3d_rot[:, 0] = kps3d[:, 0] * c - kps3d[:, 1] * s
            kps3d_rot[:, 1] = kps3d[:, 0] * s + kps3d[:, 1] * c
            kps3d = kps3d_rot

            # 3. rotate 2D keypoints in full-frame space
            kps2d_rot = kps2d.copy()
            u = kps2d[:, 0] - self._cx
            v = kps2d[:, 1] - self._cy
            kps2d_rot[:, 0] = self._cx + c * u - s * v
            kps2d_rot[:, 1] = self._cy + s * u + c * v
            kps2d = kps2d_rot

            # 4. tight bbox from rotated visible keypoints, excluding lower-leg/feet
            vis_r = (kps2d[:, 2] > 0) & _BBOX_KPS
            if not vis_r.any():
                return _invalid_overhead
            px_r, py_r = kps2d[vis_r, 0], kps2d[vis_r, 1]
            span_r = max(float(px_r.max() - px_r.min()), float(py_r.max() - py_r.min()), 1.0)
            pad_r  = span_r * 0.1
            rx0 = float(np.clip(px_r.min() - pad_r, 0, W_full))
            ry0 = float(np.clip(py_r.min() - pad_r, 0, H_full))
            rx1 = float(np.clip(px_r.max() + pad_r, 0, W_full))
            ry1 = float(np.clip(py_r.max() + pad_r, 0, H_full))
            rw  = max(rx1 - rx0, 1.0)
            rh  = max(ry1 - ry0, 1.0)

            # 5. rotate full-frame K
            K_rot_np = _rotate_K(K_full_np, angle_deg).numpy()
        else:
            rx0, ry0, rw, rh = x0, y0, w, h
            K_rot_np = K_full_np

        # crop rotated depth, resize to NN input size
        depth_crop    = depth[:, round(ry0):round(ry0 + rh), round(rx0):round(rx0 + rw)]
        depth_resized = _depth_resize(depth_crop, H_out, W_out).unsqueeze(0)  # [1,1,H,W]

        # K: crop+scale applied on top of (possibly rotated) full-frame K
        sx = W_out / rw
        sy = H_out / rh
        crop_scale = np.array(
            [[sx,  0.0, -rx0 * sx],
             [0.0, sy,  -ry0 * sy],
             [0.0, 0.0,       1.0]], dtype=np.float32)
        K     = tch.from_numpy(crop_scale @ K_rot_np)
        K_inv = tch.linalg.inv(K)

        # map (possibly rotated) full-frame 2D kps into output space;
        # clamp out-of-bounds to nearest edge (visibility preserved)
        kps2d_out = kps2d.copy()
        kps2d_out[:, 0] = np.clip((kps2d[:, 0] - rx0) * sx, 0, W_out - 1)
        kps2d_out[:, 1] = np.clip((kps2d[:, 1] - ry0) * sy, 0, H_out - 1)

        # Derive kps19_cam by back-projecting clamped output UV at each keypoint's depth.
        # OOB keypoints are smushed to the nearest boundary ray; Z is unchanged.
        K_inv_np = K_inv.numpy()
        vis2d = kps2d[:, 2] > 0                                     # [19]
        uvh = np.stack([kps2d_out[:, 0], kps2d_out[:, 1],
                        np.ones(19, dtype=np.float32)], axis=0)      # [3,19]
        rays = K_inv_np @ uvh                                        # [3,19]  (X/Z, Y/Z, 1)
        Z_m  = kps3d[:, 2] / 1000.0                                  # [19]  depth in metres
        kps19_cam_np = (rays * Z_m[np.newaxis, :]).T.astype(np.float32)  # [19,3]
        kps19_cam_np[~vis2d] = 0.0
        kps19_cam = tch.from_numpy(kps19_cam_np).unsqueeze(0)        # [1,19,3]

        # Synthesize SimCC heatmaps from 2D annotations

        n_x = W_out * 2
        n_y = H_out * 2
        n_z = _N_Z

        simcc_x = tch.zeros(1, 133, n_x)
        simcc_y = tch.zeros(1, 133, n_y)
        simcc_z = tch.zeros(1, 133, n_z)

        # Per-sample random linear transform for Z (teaches general depth, not exact scale)
        z_scale = float(np.random.uniform(1000, 1200))
        z_root = np.mean(kps3d[7:11, 2])

        for k, simcc_idx in enumerate(_SIMCC_IDX):
            x_out, y_out, vis = kps2d_out[k]
            if vis <= 0:
                continue
            simcc_x[0, simcc_idx] = _simcc_spike(x_out * 2, n_x)
            simcc_y[0, simcc_idx] = _simcc_spike(y_out * 2, n_y)

            z_mm = float(kps3d[k, 2])  # already mm from dataset
            bin_z = float(np.clip(((z_mm - z_root) / z_scale * n_z) + (n_z // 2), 0, n_z - 1))
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
