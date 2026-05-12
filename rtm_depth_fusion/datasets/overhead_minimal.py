import json
import math
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

# Left↔right symmetric joint pairs for horizontal flip.
# _KPS19_HFLIP_PAIRS: indices into the 19-joint array.
# _SIMCC133_HFLIP_PAIRS: corresponding indices in the 133-joint SimCC channel dim.
# Joints 17/18 map to simcc 100/121 (matching hand joints, right↔left).
_KPS19_HFLIP_PAIRS    = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
                          (11, 12), (13, 14), (15, 16), (17, 18)]
_SIMCC133_HFLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
                          (11, 12), (13, 14), (15, 16), (100, 121)]


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


def _swap_pairs(t: tch.Tensor, pairs: ty.List[ty.Tuple[int, int]], dim: int) -> tch.Tensor:
    t = t.clone()
    for a, b in pairs:
        sel_a = [slice(None)] * t.ndim
        sel_b = [slice(None)] * t.ndim
        sel_a[dim] = a
        sel_b[dim] = b
        tmp = t[tuple(sel_a)].clone()
        t[tuple(sel_a)] = t[tuple(sel_b)]
        t[tuple(sel_b)] = tmp
    return t


def _make_simcc(
    kps2d_out: np.ndarray,
    kps3d: np.ndarray,
    n_x: int,
    n_y: int,
    n_z: int,
    W_out: int,
    H_out: int,
) -> ty.Tuple[tch.Tensor, tch.Tensor, tch.Tensor]:
    simcc_x = tch.zeros(1, 133, n_x)
    simcc_y = tch.zeros(1, 133, n_y)
    simcc_z = tch.zeros(1, 133, n_z)

    z_scale = float(np.random.uniform(0.7, 1.4))
    z_offset = float(np.random.uniform(-n_z * 0.1, n_z * 0.1))

    for k, simcc_idx in enumerate(_SIMCC_IDX):
        x_out, y_out, vis = kps2d_out[k]
        if vis <= 0 or not (0 <= x_out < W_out and 0 <= y_out < H_out):
            continue
        simcc_x[0, simcc_idx] = _simcc_spike(x_out * 2, n_x)
        simcc_y[0, simcc_idx] = _simcc_spike(y_out * 2, n_y)

        z_mm = float(kps3d[k, 2])
        bin_z = float(np.clip(z_mm / 5000.0 * n_z * z_scale + n_z / 2 + z_offset, 0, n_z - 1))
        simcc_z[0, simcc_idx] = _simcc_spike(bin_z, n_z)

    return simcc_x, simcc_y, simcc_z


def _apply_hflip_geom(
    depth: tch.Tensor,
    kps2d_out: np.ndarray,
    kps19_cam: tch.Tensor,
    K: tch.Tensor,
    W_out: int,
) -> ty.Tuple[tch.Tensor, np.ndarray, tch.Tensor, tch.Tensor, tch.Tensor]:
    depth = tch.flip(depth, dims=[-1])

    kps2d_out = kps2d_out.copy()
    kps2d_out[:, 0] = W_out - 1 - kps2d_out[:, 0]
    for a, b in _KPS19_HFLIP_PAIRS:
        kps2d_out[[a, b]] = kps2d_out[[b, a]]

    kps19_cam = kps19_cam.clone()
    kps19_cam[..., 0] = -kps19_cam[..., 0]
    kps19_cam = _swap_pairs(kps19_cam, _KPS19_HFLIP_PAIRS, dim=-2)

    K = K.clone()
    K[0, 2] = W_out - 1 - K[0, 2]
    K_inv = tch.linalg.inv(K)

    return depth, kps2d_out, kps19_cam, K, K_inv


def _apply_rotation(
    depth: tch.Tensor,
    kps19_cam: tch.Tensor,
    K: tch.Tensor,
    angle_deg: float,
    W_out: int,
    H_out: int,
    kps_vis: np.ndarray,
) -> ty.Tuple[tch.Tensor, tch.Tensor, np.ndarray]:
    a = math.radians(angle_deg)
    cos_a = math.cos(a)
    sin_a = math.sin(a)

    # Rotate depth image around its centre using affine sampling.
    # theta maps output normalised coords → input normalised coords,
    # so content appears rotated CCW by `a` in pixel space (Y-down).
    theta = tch.tensor(
        [[cos_a, sin_a, 0.0], [-sin_a, cos_a, 0.0]], dtype=tch.float32
    ).unsqueeze(0)
    grid = F.affine_grid(theta, depth.shape, align_corners=False)
    depth = F.grid_sample(depth, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # Rotate 3D keypoints around the optical (Z) axis by the same angle.
    # R_z(a) @ p rotates X toward -Y (CCW in image), consistent with the depth rotation above.
    R_z = tch.tensor(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=tch.float32
    )
    kps19_cam = kps19_cam @ R_z.T  # [1, 19, 3]

    # Re-project rotated keypoints to pixel space to rebuild kps2d_out.
    pts = kps19_cam[0]  # [19, 3] metres
    z = pts[:, 2].clamp(min=1e-6)
    u = (K[0, 0] * pts[:, 0] / z + K[0, 2]).numpy()
    v = (K[1, 1] * pts[:, 1] / z + K[1, 2]).numpy()
    kps2d_out = np.stack([u, v, kps_vis.astype(np.float32)], axis=1)

    return depth, kps19_cam, kps2d_out


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
        kps19_cam = tch.from_numpy(kps3d[:, :3] / 1000.0).unsqueeze(0)  # mm→m, [1,19,3]

        kps2d = np.array(ann["keypoints"], dtype=np.float32).reshape(19, 3)  # [x, y, v]
        kps2d_out = kps2d.copy()
        kps2d_out[:, 0] = (kps2d[:, 0] - x0) * sx
        kps2d_out[:, 1] = (kps2d[:, 1] - y0) * sy

        # --- Augmentation ---
        did_hflip = False
        applied_angle = 0.0
        if self.augment and np.random.random() < 0.5:
            depth_resized, kps2d_out, kps19_cam, K, K_inv = _apply_hflip_geom(
                depth_resized, kps2d_out, kps19_cam, K, W_out
            )
            did_hflip = True

        if self.augment:
            angle = float(np.random.uniform(-90.0, 90.0))
            if abs(angle) > 0.5:
                depth_resized, kps19_cam, kps2d_out = _apply_rotation(
                    depth_resized, kps19_cam, K, angle, W_out, H_out, kps2d_out[:, 2]
                )
                applied_angle = angle

        # --- Generate SimCC heatmaps from final keypoint positions ---
        n_x = W_out * 2
        n_y = H_out * 2
        n_z = _N_Z

        simcc_x, simcc_y, simcc_z = _make_simcc(kps2d_out, kps3d, n_x, n_y, n_z, W_out, H_out)

        if did_hflip:
            simcc_x = tch.flip(simcc_x, dims=[-1])
            simcc_x = _swap_pairs(simcc_x, _SIMCC133_HFLIP_PAIRS, dim=1)
            simcc_y = _swap_pairs(simcc_y, _SIMCC133_HFLIP_PAIRS, dim=1)
            simcc_z = _swap_pairs(simcc_z, _SIMCC133_HFLIP_PAIRS, dim=1)

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
