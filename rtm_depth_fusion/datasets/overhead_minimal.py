import json
import os.path as osp
import typing as ty

import numpy as np
import torch as tch
import torch.nn.functional as F
from torch.utils.data import Dataset

DEFAULT_IMAGE_SIZE = (288, 384)


class OverheadItem(ty.NamedTuple):
    valid_entry: tch.Tensor
    depth: tch.Tensor
    K: tch.Tensor
    K_inv: tch.Tensor
    kps19_cam: tch.Tensor
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


class OverheadMinDataset(Dataset):
    def __init__(
        self,
        root: str,
        img_size: ty.Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ):
        super().__init__()
        self.root = root
        self.img_size = img_size  # (H, W)

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
        depth = tch.from_numpy(depth_np).unsqueeze(0)  # [1, H, W]

        H_out, W_out = self.img_size
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

        kps = np.array(ann["keypoints_3d"], dtype=np.float32).reshape(19, 4)
        kps19_cam = tch.from_numpy(kps[:, :3] / 1000.0).unsqueeze(0)  # mm → m, [1,19,3]

        return OverheadItem(
            valid_entry=tch.tensor([True]),
            depth=depth_resized.to(tch.int16),
            K=K.unsqueeze(0),
            K_inv=K_inv.unsqueeze(0),
            kps19_cam=kps19_cam,
            image_id=[image_id],
            ann_id=[ann_id],
        )
