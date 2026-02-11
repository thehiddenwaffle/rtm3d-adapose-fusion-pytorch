import torch as tch
import torch.nn.functional as F


# depth->3d(IR)->3d(Color)
def depth_to_rgbz_image_BAK(
    depth,  # [B,1,Hd,Wd]
    K_depth_inv,  # [B,3,3]  inverse intrinsics
    K_rgb,  # [B,3,3]  forward intrinsics
    T_depth_to_rgb,  # [B,4,4]  rigid transform
    Hrgb,
    Wrgb,
):
    B, _, Hd, Wd = depth.shape
    device = depth.device

    # --- build depth pixel grid ---
    y, x = tch.meshgrid(tch.arange(Hd, device=device), tch.arange(Wd, device=device))

    ones = tch.ones_like(x)
    pix = tch.stack([x, y, ones], dim=-1).float()  # [Hd,Wd,3]
    pix = pix.view(1, -1, 3).repeat(B, 1, 1)  # [B,N,3]
    depth_flat = depth.view(B, -1, 1)

    # --- backproject to 3D in depth camera ---
    X = (K_depth_inv @ pix.transpose(1, 2)).transpose(1, 2)
    X = X * depth_flat  # [B,N,3]

    # --- transform to RGB camera frame ---
    X_h = tch.cat([X, tch.ones_like(X[..., :1])], dim=-1)  # [B,N,4]
    Xr = (T_depth_to_rgb @ X_h.transpose(1, 2)).transpose(1, 2)[..., :3]

    Z = Xr[..., 2:3]
    valid = Z[..., 0] > 0

    # --- project to RGB pixels ---
    proj = (K_rgb @ Xr.transpose(1, 2)).transpose(1, 2)
    uv = proj[..., :2] / proj[..., 2:3]

    u = uv[..., 0].round().long()
    v = uv[..., 1].round().long()

    # --- bounds mask ---
    in_bounds = (u >= 0) & (u < Wrgb) & (v >= 0) & (v < Hrgb) & valid

    # --- z-buffer splatting ---
    rgbz = tch.full((B, Hrgb, Wrgb), float("inf"), device=device)

    for b in range(B):
        ub = u[b][in_bounds[b]]
        vb = v[b][in_bounds[b]]
        zb = Z[b][in_bounds[b], 0]

        idx = vb * Wrgb + ub
        flat = rgbz[b].view(-1)

        flat.index_put_((idx,), zb, accumulate=False)
        # keep nearest depth
        flat[idx] = tch.minimum(flat[idx], zb)

    rgbz[rgbz == float("inf")] = 0
    return rgbz.unsqueeze(1)


def depth_to_rgbz_image(
    depth,  # [B,1,Hd,Wd]
    K_depth_inv,  # [B,3,3]
    K_rgb,  # [B,3,3]
    T_depth_to_rgb,  # [B,4,4]
    Hrgb,
    Wrgb,
):
    B, _, Hd, Wd = depth.shape
    device = depth.device

    # --- create RGB pixel grid ---
    y_rgb, x_rgb = tch.meshgrid(
        tch.arange(Hrgb, device=device), tch.arange(Wrgb, device=device)
    )
    ones = tch.ones_like(x_rgb)
    rgb_pix = tch.stack([x_rgb, y_rgb, ones], dim=-1).float()  # [Hrgb,Wrgb,3]
    rgb_pix = rgb_pix.view(1, Hrgb * Wrgb, 3).repeat(B, 1, 1)  # [B,Hrgb*Wrgb,3]

    # --- project RGB pixels into depth camera frame ---
    K_rgb_inv = tch.inverse(K_rgb)  # [B,3,3]
    rays = (K_rgb_inv @ rgb_pix.transpose(1, 2)).transpose(1, 2)  # [B,Hrgb*Wrgb,3]
    rays_h = tch.cat([rays, tch.ones_like(rays[..., :1])], dim=-1)  # [B,Hrgb*Wrgb,4]

    # --- transform rays to depth camera frame ---
    T_rgb_to_depth = tch.inverse(T_depth_to_rgb)
    X_depth = (T_rgb_to_depth @ rays_h.transpose(1, 2)).transpose(1, 2)[
        ..., :3
    ]  # [B,Hrgb*Wrgb,3]

    # --- convert 3D points to depth pixel coordinates ---
    uv_depth = (K_depth_inv.inverse() @ X_depth.transpose(1, 2)).transpose(
        1, 2
    )  # [B,Hrgb*Wrgb,3]
    u = uv_depth[..., 0] / uv_depth[..., 2]
    v = uv_depth[..., 1] / uv_depth[..., 2]

    # --- normalize to [-1,1] for grid_sample ---
    u_norm = 2 * (u / (Wd - 1)) - 1
    v_norm = 2 * (v / (Hd - 1)) - 1
    grid = tch.stack([u_norm, v_norm], dim=-1)  # [B,Hrgb*Wrgb,2]
    grid = grid.view(B, Hrgb, Wrgb, 2)  # now safe

    # --- sample depth map using nearest neighbor ---
    rgbz = F.grid_sample(
        depth, grid, mode="nearest", padding_mode="zeros", align_corners=True
    )

    return rgbz  # [B,1,Hrgb,Wrgb]


def extract_pcl_patches(depth, uv_px, intrinsic_inv_K=None, K_inv_dense=None, window=3):
    """
    depth: [B, 1, H, W]
    uv_px: [B, K, 2] pixel coords (float or int, in pixel units)
    intrinsic_K: [B, 3, 3] camera intrinsics
        [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]]
    window: radius; 1 -> 3x3, 2 -> 5x5
    returns: [B, K, (2 * window + 1)^2, 3]  # XYZ in camera coords
    """
    B, _, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    K_pts = uv_px.shape[1]

    # -------- clamp UV together so the (2w+1)x(2w+1) patch stays inside --------
    uv_px[..., 0] = uv_px[..., 0].clamp(min=window, max=W - 1 - window)
    uv_px[..., 1] = uv_px[..., 1].clamp(min=window, max=H - 1 - window)
    uv_center_int = uv_px.round().to(tch.long)  # [B, K, 2]

    # -------- build 2D offsets for the (2w+1)x(2w+1) window --------
    offsets_1d = tch.arange(-window, window + 1, device=device)
    dy, dx = tch.meshgrid(offsets_1d, offsets_1d)  # [2w+1, 2w+1]
    dx = dx.reshape(-1)  # [P]
    dy = dy.reshape(-1)  # [P]
    # offsets_2d[..., 0] = Δu (x), offsets_2d[..., 1] = Δv (y)
    offsets_2d = tch.stack([dx, dy], dim=-1)  # [P, 2]
    P = offsets_2d.shape[0]

    # Broadcast center + offsets: [B, K, 1, 2] + [1, 1, P, 2] -> [B, K, P, 2]
    uv_idx = uv_center_int.unsqueeze(2) + offsets_2d.view(1, 1, P, 2)  # [B, K, P, 2]

    # Final safety clamp for indices (in case of numerical edge issues)
    # min_idx = tch.tensor([0, 0], device=device, dtype=tch.long)
    # max_idx = tch.tensor([W - 1, H - 1], device=device, dtype=tch.long)
    # uv_idx = uv_idx.clamp(min=min_idx, max=max_idx)        # [B, K, P, 2]

    # Split for indexing: u_idx (x), v_idx (y)
    u_idx = uv_idx[..., 0]  # [B, K, P]
    v_idx = uv_idx[..., 1]  # [B, K, P]

    # Gather depth patches
    depth_flat = depth.view(B, H * W)  # [B, H*W]
    lin_idx = (v_idx * W + u_idx).view(B, -1)  # [B, K*P]
    patch_depth = depth_flat.gather(1, lin_idx)  # [B, K*P]
    patch_depth = patch_depth.view(B, K_pts, P)  # [B, K, P]

    # Build homogeneous pixel coords [u, v, 1] for each patch element
    uv_float = uv_idx.view(B, -1, 2).to(dtype)  # [B, K*P, 2]
    ones = tch.ones(
        (B, uv_float.shape[1], 1), device=device, dtype=dtype
    )  # [B, K*P, 1]
    pix_h = tch.cat([uv_float, ones], dim=-1)  # [B, K*P, 3]

    if K_inv_dense is not None and intrinsic_inv_K is None:
        cam_dirs = tch.cat([uv_float * K_inv_dense[..., 0:1, :] + K_inv_dense[..., 1:2, :],ones], dim=-1)
    else:
        cam_dirs = tch.matmul(intrinsic_inv_K.unsqueeze(1), pix_h.unsqueeze(-1)).squeeze(
            -1
        )  # [B, K*P, 3]

    # Scale by depth to get XYZ
    z_flat = patch_depth.view(B, K_pts * P, 1)  # [B, K*P, 1]
    points_flat = cam_dirs * z_flat  # [B, K*P, 3]

    # Reshape back to [B, K, P, 3]
    pcl_patches = points_flat.view(B, K_pts, P, 3)  # [B, K, P, 3]

    return pcl_patches, points_flat
