import json
import os
import os.path as osp
import typing as ty

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch as tch

# noinspection PyPep8Naming
import torch.nn.functional as F
from torch.utils.data import Dataset

from rtm_depth_fusion.utils import depth_to_rgbz_image

DEFAULT_OUTPUT_ROOT = "kinect_coco133_corrected"
DEFAULT_DEPTH_SCALE = 1.0e-3
CAMERA_CANDIDATES = ["master", "sub_1", "sub_2", "sub_3", "sub_4"]
SUBJECT_LABELS = ["interactee", "camera_wearer"]
DEFAULT_DESIRED_DEPTH_STD = 1.0
DEFAULT_DESIRED_DEPTH_MEAN = 1.5
DEFAULT_IMAGE_SIZE = (288, 384)


# noinspection PyPep8Naming
class CameraCalibration(ty.NamedTuple):
    fx: float
    fy: float
    cx: float
    cy: float

    def as_tensor(
        self,
        *,
        device: ty.Optional[tch.device] = None,
        dtype: ty.Optional[tch.dtype] = None,
    ) -> tch.Tensor:
        """
        Returns the intrinsic camera matrix as a 3x3 torch tensor on the requested device/dtype.
        A cloned tensor is returned to ensure downstream code can edit it safely.
        """
        K = tch.tensor(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=dtype or tch.float32,
            device=device,
        )
        return K

    def inverse_tensor(
        self,
        *,
        device: ty.Optional[tch.device] = None,
        dtype: ty.Optional[tch.dtype] = None,
    ) -> tch.Tensor:
        """Returns K^{-1} computed via tch.linalg."""
        return tch.linalg.inv(self.as_tensor(device=device, dtype=dtype))

    def project(
        self,
        points_cam: tch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> ty.Tuple[tch.Tensor, tch.Tensor]:
        """
        Projects 3D camera-space points to pixel coordinates via matrix multiplication.

        Args:
            points_cam: (..., 3) tensor of XYZ points in the camera frame.
            eps: minimum positive depth threshold before treating as invalid.
        Returns:
            uv: (..., 2) tensor of pixel coordinates with NaNs for invalid points.
            valid: (...) boolean mask indicating finite positive depths.
        """
        if points_cam.shape[-1] != 3:
            raise ValueError("points_cam must have shape (..., 3)")

        device, dtype = points_cam.device, points_cam.dtype
        K = self.as_tensor(device=device, dtype=dtype)
        z = points_cam[..., 2:3]
        valid = tch.isfinite(z) & (z > eps)
        z_safe = tch.where(valid, z, tch.ones_like(z))

        normalized = tch.cat(
            [
                points_cam[..., :2] / z_safe,
                tch.ones_like(z_safe),
            ],
            dim=-1,
        )
        uv_h = normalized @ K.transpose(0, 1)
        uv = uv_h[..., :2]
        invalid_fill = tch.full_like(uv, float("nan"))
        uv = tch.where(valid.squeeze(-1).unsqueeze(-1), uv, invalid_fill)
        return uv, valid.squeeze(-1)

    @classmethod
    def from_file(
        cls, cam_params_dir: str, view: str, color: bool = True
    ) -> "CameraCalibration":
        """
        Loads camera parameters from a file and returns an instance of CameraCalibration.

        Args:
            cam_params_dir (str): Path to the directory containing camera parameter files.
            view (str): The view identifier (e.g., 'front', 'left', etc.).
            color (bool): Whether to load "Color.json" (default) or "IR.json".

        Returns:
            CameraCalibration: An instance of CameraCalibration with loaded parameters.
        """
        file_name = "Color.json" if color else "IR.json"
        json_path = osp.join(cam_params_dir, f"kinect_{view}", file_name)

        with open(json_path, "r") as file:
            cam_data = json.load(file)

        if "f" in cam_data and "c" in cam_data:
            fx, fy = cam_data["f"]
            cx, cy = cam_data["c"]
        else:
            raise KeyError(
                "Camera intrinsics must contain 'f' (focal lengths) and 'c' (principal point)."
            )

        return cls(fx, fy, cx, cy)


class RecordingMeta(ty.NamedTuple):
    start: int
    end: int
    interactee_idx: int
    camera_wearer_idx: int
    interactee_gender: str
    camera_wearer_gender: str


class DatasetIndexEntry(ty.NamedTuple):
    recording: str
    frame_id: str
    camera: str
    depth_path: str
    color_path: str
    npz_path: str


class CameraParams(ty.NamedTuple):
    color_cam: CameraCalibration
    ir_cam: CameraCalibration
    T_depth_to_color: np.ndarray


class ViewInfo(ty.NamedTuple):
    recording: str
    camera: str
    p: CameraParams
    T_master_to_color: ty.Optional[np.ndarray]


class EgoBodyItem(ty.NamedTuple):
    valid_entry: tch.Tensor  # single bool
    depth: tch.Tensor
    K: tch.Tensor
    K_inv: tch.Tensor
    kps133_cam: tch.Tensor
    simcc_x: tch.Tensor
    simcc_y: tch.Tensor
    simcc_z: tch.Tensor
    recording: ty.List[str]
    frame_id: ty.List[str]
    camera: ty.List[str]
    depth_path: ty.List[str]
    smplx_pkl_path: ty.List[str]
    subject_label: ty.List[str]

    @staticmethod
    def collate(items: ty.List["EgoBodyItem"]) -> "EgoBodyItem":
        batch = [i for i in items if bool(tch.all(i.valid_entry))]
        if not any(batch):
            return invalidEgo

        out: ty.Dict[str, ty.Any] = {}

        for key in EgoBodyItem._fields:
            items_for_key = [getattr(bi, key) for bi in batch]
            if EgoBodyItem.__annotations__[key] is tch.Tensor:
                out[key] = tch.cat(items_for_key, dim=0)
            elif EgoBodyItem.__annotations__[key] is ty.List[str]:
                out[key] = sum(items_for_key, [])
            else:
                raise NotImplementedError(f"un-collatable field was added {key}")
        return EgoBodyItem(**out)

    def to(self, *args, **kwargs):
        out = {}
        for key in EgoBodyItem._fields:
            value = getattr(self, key)
            if isinstance(value, tch.Tensor):
                out[key] = value.to(*args, **kwargs)
            else:
                out[key] = value
        return EgoBodyItem(**out)


# noinspection PyArgumentList
invalidEgo = EgoBodyItem(
    **{k: None for k in EgoBodyItem._fields if k != "valid_entry"},
    valid_entry=tch.tensor([False]),
)


# -------------------------------------------------------------------------
# JSON / SE(3) helpers
# -------------------------------------------------------------------------
def _load_json(path: str) -> ty.Dict:
    if not osp.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        return json.load(f)


# noinspection PyPep8Naming
def parse_transform(tf_json_path: str) -> np.ndarray:
    data = _load_json(tf_json_path)
    if "trans" in data:
        arr = np.asarray(data["trans"], dtype=np.float32)
        return arr.reshape(4, 4)
    if "matrix" in data:
        arr = np.asarray(data["matrix"], dtype=np.float32)
        return arr.reshape(4, 4)
    if "rotation" in data and "translation" in data:
        R = np.asarray(data["rotation"], dtype=np.float32).reshape(3, 3)
        t = np.asarray(data["translation"], dtype=np.float32).reshape(3)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    raise ValueError(f"Unsupported transform schema: {tf_json_path}")


# noinspection PyPep8Naming
def invert_SE3(T: np.ndarray) -> np.ndarray:
    if T.shape != (4, 4):
        raise ValueError("invert_SE3 expects a 4x4 matrix.")
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4, dtype=T.dtype)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def load_depth_to_color_transform(
    cam_params_dir: str, view: str
) -> ty.Optional[np.ndarray]:
    base = osp.join(cam_params_dir, f"kinect_{view}")
    ir_path = osp.join(base, "IR.json")

    def _parse(data: ty.Dict) -> ty.Optional[np.ndarray]:
        if (d := data.get("ext_depth2color", None)) is not None:
            arr = np.asarray(d, dtype=np.float32).reshape(4, 4)
            return arr
        if (d := data.get("ext_color2depth", None)) is not None:
            arr = np.asarray(d, dtype=np.float32).reshape(4, 4)
            return invert_SE3(arr)
        return None

    return _parse(_load_json(ir_path))


def load_extrinsic_mat(calib_root: str, recording: str, view: str) -> np.ndarray:
    if view == "master":
        return np.eye(4, dtype=np.float32)
    trans_dir = osp.join(calib_root, recording, "cal_trans")
    mapping = {
        "sub_1": "kinect_11to12_color.json",
        "sub_2": "kinect_13to12_color.json",
        "sub_3": "kinect_14to12_color.json",
        "sub_4": "kinect_15to12_color.json",
    }
    if view not in mapping:
        raise ValueError(f"Unsupported view '{view}'.")
    path = osp.join(trans_dir, mapping[view])
    if not osp.isfile(path):
        raise FileNotFoundError(path)
    data = _load_json(path)
    arr = np.asarray(data.get("trans"), dtype=np.float32)
    if arr.shape != (4, 4):
        arr = arr.reshape(4, 4)
    return invert_SE3(arr)


def _homogeneous(points: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    return np.concatenate([points, ones], axis=1)


# noinspection PyPep8Naming
def _apply_transform(points: np.ndarray, T: ty.Optional[np.ndarray]) -> np.ndarray:
    if T is None:
        return points.copy()
    pts_h = _homogeneous(points)
    transformed = (T @ pts_h.T).T
    if transformed.shape[1] == 4:
        w = transformed[:, 3:4]
        non_zero = np.abs(w) > 1e-8
        transformed[non_zero[:, 0]] /= w[non_zero[:, 0]]
    return transformed[:, :3]


# noinspection PyPep8Naming
def depth_resize_stretch(depth, W_out, H_out):
    # depth: (..., H, W)
    *batch, H, W = depth.shape
    x = depth.reshape(-1, 1, H, W)  # (B,1,H,W)
    x = F.interpolate(x, size=(H_out, W_out), mode="bilinear", align_corners=False)
    return x.reshape(*batch, H_out, W_out)


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
class EgoBodyDataset(Dataset):
    def __init__(
        self,
        release_root: str,
        split: str = "val",
        output_root: str = DEFAULT_OUTPUT_ROOT,
        img_size: ty.Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        depth_scale: float = DEFAULT_DEPTH_SCALE,
    ):
        super().__init__()
        self.release_root = release_root
        self.split = split
        self.img_size = img_size
        self.output_root = output_root
        self.depth_scale = float(depth_scale)

        self.depth_root = osp.join(release_root, "kinect_depth")
        self.color_root = osp.join(release_root, "kinect_color")
        self.cam_params_root = osp.join(release_root, "kinect_cam_params")
        self.calib_root = osp.join(release_root, "calibrations")
        self.output_abs = osp.join(release_root, output_root)

        self._load_recording_meta_table(release_root)
        self.camera_params_cache: ty.Dict[str, CameraParams] = {}
        self.entries: ty.List[DatasetIndexEntry] = self._build_index(split)
        self._view_cache: ty.Dict[ty.Tuple[str, str], ViewInfo] = {}

        if not self.entries:
            raise RuntimeError(f"No samples found for split='{split}'.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> EgoBodyItem:
        entry = self.entries[idx]
        recording = entry.recording
        frame_id = entry.frame_id
        npz_path = entry.npz_path
        depth_path = entry.depth_path
        color_path = entry.color_path
        camera = entry.camera

        if (
            not osp.exists(npz_path)
            or not osp.exists(depth_path)
            or not osp.exists(color_path)
        ):
            return invalidEgo

        view_info = self._get_view_info(recording, camera)

        # Depth frames are in eights of a mm for some reason: https://github.com/sanweiliti/EgoBody/blob/e3ecbb839d3d7d2fcd758cb311661105975840d1/release_vis_kinect_pcd.py#L75
        depth_raw = imageio.imread(depth_path).astype(np.float32) / 8.0
        depth_m = depth_raw * self.depth_scale
        depth_tensor = tch.from_numpy(depth_m).unsqueeze(0)  # [1,H,W]
        rgbz_but_only_z = depth_to_rgbz_image(
            depth_tensor.unsqueeze(0),
            view_info.p.ir_cam.inverse_tensor().unsqueeze(0),
            view_info.p.color_cam.as_tensor().unsqueeze(0),
            tch.from_numpy(view_info.p.T_depth_to_color).unsqueeze(0),
            1080,
            1920,
        )

        def one_subject(subject):
            points_master, confidence, simcc, roi = self._load_keypoints(
                npz_path, subject, camera
            )
            if roi is None:
                return invalidEgo
            x0, y0, scale_x, scale_y = (
                roi["x0"],
                roi["y0"],
                roi["scale_x"],
                roi["scale_y"],
            )
            width_crop, height_crop = (
                scale_x * self.img_size[0],
                scale_y * self.img_size[1],
            )

            if (
                width_crop < 15
                or height_crop < 15
                or not all(isinstance(i, np.ndarray) for i in simcc.values())
            ):
                return invalidEgo

            # TODO now that we're returning to color space need project whole depth map to XYZ, translate to color space, and then back to UVZ and do something about the depth holes

            # Convert from master coords to "this" camera(ident if this is master)
            pts_color = _apply_transform(points_master, view_info.T_master_to_color)
            points_cam_t = tch.from_numpy(pts_color).unsqueeze(0)

            cx_crop = float(view_info.p.color_cam.cx) - float(x0)
            cy_crop = float(view_info.p.color_cam.cy) - float(y0)

            depth_stretched = depth_resize_stretch(
                rgbz_but_only_z[
                    ...,
                    round(y0) : round(y0 + height_crop),
                    round(x0) : round(x0 + width_crop),
                ],
                int(self.img_size[0]),
                int(self.img_size[1]),
            )
            sx = float(self.img_size[0]) / width_crop
            sy = float(self.img_size[1]) / height_crop
            fx_scaled = view_info.p.color_cam.fx * sx  # fx'
            fy_scaled = view_info.p.color_cam.fy * sy  # fy'
            cx_scaled = float(cx_crop) * sx  # cx'' = sx * cx'
            cy_scaled = float(cy_crop) * sy  # cy'' = sy * cy'
            # noinspection PyPep8Naming
            color_resize_K = CameraCalibration(
                fx=fx_scaled, fy=fy_scaled, cx=cx_scaled, cy=cy_scaled
            )

            return EgoBodyItem(
                valid_entry=tch.tensor([True]),
                depth=depth_stretched,
                K=color_resize_K.as_tensor().unsqueeze(0),
                K_inv=color_resize_K.inverse_tensor().unsqueeze(0),
                kps133_cam=points_cam_t,
                simcc_x=tch.from_numpy(simcc["x"]),
                simcc_y=tch.from_numpy(simcc["y"]),
                simcc_z=tch.from_numpy(simcc["z"]),
                recording=[recording],
                frame_id=[frame_id],
                camera=[camera],
                depth_path=[depth_path],
                smplx_pkl_path=[npz_path],
                subject_label=[subject],
            )

        cam_wearer = one_subject("camera_wearer")
        interactee = one_subject("interactee")

        return EgoBodyItem.collate([cam_wearer, interactee])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_recording_meta_table(self, release_root: str):
        info_csv = osp.join(release_root, "data_info_release.csv")
        df = pd.read_csv(info_csv)
        meta: ty.Dict[str, RecordingMeta] = {}
        for _, row in df.iterrows():
            name = row.get("recording_name")
            if not isinstance(name, str):
                continue
            recording = name.strip()
            start = int(row.get("start_frame", 0))
            end = int(row.get("end_frame", start))
            fpv_raw = str(row.get("body_idx_fpv", "0 male")).split()
            interactee_idx = int(fpv_raw[0])
            interactee_gender = fpv_raw[1] if len(fpv_raw) > 1 else "neutral"
            other_col = "body_idx_0" if interactee_idx == 1 else "body_idx_1"
            camera_gender = str(row.get(other_col, "0 neutral")).split()
            camera_gender = camera_gender[1] if len(camera_gender) > 1 else "neutral"
            meta[recording] = RecordingMeta(
                start=start,
                end=end,
                interactee_idx=interactee_idx,
                camera_wearer_idx=1 - interactee_idx,
                interactee_gender=interactee_gender,
                camera_wearer_gender=camera_gender,
            )
        self.recording_meta = meta

    def _load_camera_params(self, view: str) -> CameraParams:
        if view in self.camera_params_cache:
            return self.camera_params_cache[view]
        # noinspection PyPep8Naming
        T_depth_to_color = load_depth_to_color_transform(self.cam_params_root, view)
        params = CameraParams(
            color_cam=CameraCalibration.from_file(self.cam_params_root, view, True),
            ir_cam=CameraCalibration.from_file(self.cam_params_root, view, False),
            T_depth_to_color=T_depth_to_color,
        )
        self.camera_params_cache[view] = params
        return params

    @staticmethod
    def _list_recordings_from_splits(
        release_root: str, splits: ty.Sequence[str]
    ) -> ty.List[str]:
        splits_csv = osp.join(release_root, "data_splits.csv")
        df = pd.read_csv(splits_csv)
        ordered: ty.List[str] = []
        seen = set()
        for split in splits:
            if split not in df.columns:
                continue
            for name in df[split]:
                if isinstance(name, str):
                    name = name.strip()
                if not name or not isinstance(name, str):
                    continue
                if name not in seen:
                    ordered.append(name)
                    seen.add(name)
        return ordered

    def _build_index(self, split: str) -> ty.List[DatasetIndexEntry]:
        def _list_corrected_frames(output_dir: str) -> ty.List[str]:
            if not osp.isdir(output_dir):
                return []
            frames = [
                f[:-4]
                for f in os.listdir(output_dir)
                if f.startswith("frame_") and f.endswith(".npz")
            ]
            frames.sort()
            return frames

        recordings = self._list_recordings_from_splits(self.release_root, [split])
        entries: ty.List[DatasetIndexEntry] = []
        for rec in recordings:
            meta = self.recording_meta.get(rec)
            out_dir = osp.join(self.output_abs, rec)
            frame_list = _list_corrected_frames(out_dir)
            if not frame_list:
                continue
            if meta and len(frame_list) != meta.end - meta.start + 1:
                continue

            # Build entries for all frame_id, camera, and subject
            for frame_id in frame_list:
                npz_path = osp.join(out_dir, f"{frame_id}.npz")
                for camera in CAMERA_CANDIDATES:
                    depth_path = osp.join(
                        self.depth_root, rec, camera, f"{frame_id}.png"
                    )
                    color_path = osp.join(
                        self.color_root, rec, camera, f"{frame_id}.jpg"
                    )
                    entries.append(
                        DatasetIndexEntry(
                            recording=rec,
                            frame_id=frame_id,
                            camera=camera,
                            depth_path=depth_path,
                            color_path=color_path,
                            npz_path=npz_path,
                        )
                    )

        entries.sort(key=lambda e: (e.recording, e.frame_id, e.camera))
        return entries

    def _get_view_info(self, recording: str, camera: str) -> ViewInfo:
        key = (recording, camera)
        if key in self._view_cache:
            return self._view_cache[key]
        cam_params = self._load_camera_params(camera)
        # noinspection PyPep8Naming
        T_master_to_color = load_extrinsic_mat(self.calib_root, recording, camera)
        info = ViewInfo(
            recording=recording,
            camera=camera,
            p=cam_params,
            T_master_to_color=T_master_to_color,
        )
        self._view_cache[key] = info
        return info

    @staticmethod
    def _load_keypoints(
        npz_path: str, subject: str, view: str
    ) -> ty.Tuple[np.ndarray, np.ndarray, ty.Dict[str, np.array], np.ndarray]:
        prefixes = [f"merged_{subject}", f"rtm_{subject}", f"smpl_{subject}"]

        # frame_payload[f"rtm_{label}_{view}_region"] = observation['root_and_scale']
        with np.load(npz_path, allow_pickle=True) as data:
            simcc = {
                axis: data.get(f"rtm_{subject}_{view}_simcc_{axis}", None)
                for axis in ("x", "y", "z")
            }
            root_scale = data.get(f"rtm_{subject}_{view}_region", None)
            # if any(e is None for e in simcc.values()):
            #     raise KeyError(f"Missing simcc maps for subject '{subject}' in view '{view}'. File: {npz_path}")
            for prefix in prefixes:
                pts_key = f"{prefix}_points"
                conf_key = f"{prefix}_confidence"
                if pts_key in data and conf_key in data:
                    pts = np.asarray(data[pts_key], dtype=np.float32)
                    conf = np.asarray(data[conf_key], dtype=np.float32)
                    return (
                        pts,
                        np.clip(conf, 0.0, 1.0),
                        simcc,
                        root_scale.item() if root_scale is not None else None,
                    )  # TODO don't love this
        raise KeyError(f"No keypoints found for subject '{subject}' in {npz_path}")
