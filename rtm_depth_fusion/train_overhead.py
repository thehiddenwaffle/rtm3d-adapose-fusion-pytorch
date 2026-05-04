import argparse as ap
import math
import os
import typing as ty

import torch as tch
import torch.nn as nn
from torch.utils.data import DataLoader

from rtm_depth_fusion import RTMPoseToAdaPose
from rtm_depth_fusion.datasets.overhead_minimal import OverheadMinDataset, OverheadItem

# Shoulder + elbow indices in the 19-keypoint overhead annotation set
_ARM_KPS = tch.tensor([5, 6, 7, 8])


def load_ckpt(
        path: str,
        model: nn.Module,
        optimizer: ty.Optional[tch.optim.Optimizer] = None,
        scaler: ty.Optional[tch.cuda.amp.GradScaler] = None,
        map_location: str = "cpu",
) -> ty.Dict[str, ty.Any]:
    ckpt = tch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optim" in ckpt:
        optimizer.load_state_dict(ckpt["optim"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def huber_loss(err, delta=0.9, reduction="mean"):
    abs_error = tch.abs(err)
    quadratic = tch.minimum(abs_error, tch.tensor(delta, device=err.device))
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    if reduction == "mean":
        return loss.mean(dim=-1, keepdim=True)
    elif reduction == "sum":
        return loss.sum(dim=-1, keepdim=True)
    else:
        return loss


def root_z_loss_fn(pred_root_z, kps19_cam, conf):
    arm_kps = _ARM_KPS.to(kps19_cam.device)
    root_gt = tch.mean(kps19_cam[:, arm_kps, :], dim=1)
    diff = pred_root_z[:, :, 2:].squeeze(1) - root_gt[:, 2:]
    w = tch.min(tch.mean(conf[:, arm_kps, :], dim=1) ** 2.0, dim=-1, keepdim=True).values
    return huber_loss(w * diff, reduction="")


def arm_delta_d_loss_fn(pred_coco_metric_xyz, kps19_cam, conf):
    arm_kps = _ARM_KPS.to(kps19_cam.device)
    diff = pred_coco_metric_xyz[:, arm_kps, 2] - kps19_cam[:, arm_kps, 2]
    w = tch.clamp(2.0 * (tch.min(conf[:, arm_kps, :], dim=-1).values - 0.25), min=0.05, max=0.99)
    weighted_diff = (w * diff).sum() / w.sum()
    return huber_loss(weighted_diff)


def train_one_epoch(
        model: RTMPoseToAdaPose,
        loader: DataLoader,
        optimizer: tch.optim.Optimizer,
        scaler: tch.cuda.amp.GradScaler,
        args: ap.Namespace,
        epoch: int,
) -> ty.Dict[str, float]:
    model.train()
    if not args.train_ada_layers:
        model.sampler.requires_grad_(False)
        model.encoder.requires_grad_(False)

    tot_loss, tot_root_z, tot_arm_d, n = 0.0, 0.0, 0.0, 0

    for batch in loader:
        batch: OverheadItem = ty.cast(OverheadItem, batch).to(
            device=args.device, non_blocking=True
        )

        optimizer.zero_grad()

        with tch.cuda.amp.autocast(enabled=args.amp):
            camera_K_inv_squashed = tch.stack(
                (
                    batch.K_inv[..., [0, 1], [0, 1]],
                    batch.K_inv[..., [0, 1], 2],
                ),
                dim=1,
            )

            pred_coco_metric_xyz, pred_root_z, uv_conf, _, _, _ = model(
                batch.depth,
                batch.simcc_x,
                batch.simcc_y,
                batch.simcc_z,
                camera_K_inv_squashed,
            )

            loss_z = root_z_loss_fn(pred_root_z, batch.kps19_cam, uv_conf).mean()
            loss_arm = arm_delta_d_loss_fn(pred_coco_metric_xyz, batch.kps19_cam, uv_conf).mean()

            loss = loss_z + loss_arm

        scaler.scale(loss).backward()

        if args.grad_clip and args.grad_clip > 0:
            scaler.unscale_(optimizer)
            tch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        tot_loss += float(loss.detach().cpu())
        tot_root_z += float(loss_z.detach().cpu())
        tot_arm_d += float(loss_arm.detach().cpu())
        n += 1

        if (n + 1) % args.log_every == 0:
            print(
                f"[train e{epoch:03d} it{n:05d}/{len(loader)}] "
                f"loss={tot_loss / n:.6f} root_z={tot_root_z / n:.6f} arm_d={tot_arm_d / n:.6f}"
            )

    return {
        "loss": tot_loss / max(n, 1),
        "arm_d": tot_arm_d / max(n, 1),
        "root_z": tot_root_z / max(n, 1),
    }


def save_ckpt(path: str,
              model: nn.Module,
              optimizer: tch.optim.Optimizer,
              scaler: tch.cuda.amp.GradScaler,
              epoch: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    tch.save(payload, path)


def main():
    DEFAULT_LR = 1e-4
    parser = ap.ArgumentParser("Train overhead minimal")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/media/thwdpc/extrastorage/PycharmProjects/dai-frame-stitch/datasets/overhead_live",
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7654321)
    parser.add_argument("--ckpt-dir", type=str, default="./ckpts_overhead")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--train-ada-layers", action="store_true", default=False)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    tch.manual_seed(args.seed)
    tch.cuda.manual_seed_all(args.seed)

    model = RTMPoseToAdaPose()
    model.to(args.device)

    optimizer = tch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = tch.cuda.amp.GradScaler(enabled=args.amp)
    scheduler = tch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0

    if args.resume:
        opt_or_not = None
        if math.isclose(args.lr, DEFAULT_LR):
            opt_or_not = optimizer
        ckpt = load_ckpt(args.resume, model, opt_or_not, scaler, map_location=args.device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from {args.resume} @ epoch={start_epoch}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    train_ds = OverheadMinDataset(args.dataset_root)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda b: OverheadItem.collate(b),
    )

    for epoch in range(start_epoch, args.epochs):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, args, epoch)
        scheduler.step()
        save_ckpt(
            os.path.join(args.ckpt_dir, f"epoch_{epoch:03d}.pt"),
            model, optimizer, scaler, epoch,
        )


if __name__ == "__main__":
    main()
