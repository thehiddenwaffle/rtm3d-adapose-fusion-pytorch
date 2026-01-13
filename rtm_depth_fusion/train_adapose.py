import argparse as ap
import math
import os
import typing as ty

import torch as tch
import torch.nn as nn
from torch.utils.data import DataLoader

from rtm_depth_fusion import RTMPoseToAdaPose
from rtm_depth_fusion.datasets import EgoBodyDataset, EgoBodyItem


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


# My torch version is too far behind
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


def root_z_loss_fn(pred_root_z, kps133_cam):
    torso_derived_from = tch.tensor([5, 6, 11, 12], device=kps133_cam.device)
    root_gt = tch.mean(kps133_cam[:, torso_derived_from, :], dim=1)
    return huber_loss(pred_root_z[:, :, 2:].squeeze(1) - root_gt[:, 2:], reduction="")


def joint19_z_loss_fn(pred_coco_main_metric_xyz, kps133_cam, conf):
    main_19 = tch.arange(0, 19, device=kps133_cam.device)
    diff = pred_coco_main_metric_xyz[:, :, 2] - kps133_cam[:, main_19, 2]
    # Anything > .75 is 1, anything <.25 is fully suppressed
    w = tch.clamp(2.0 * (tch.min(conf, dim=-1, keepdim=True).values - 0.25), min=0.05, max=0.99)
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
        # model.pre.z_guesser.requires_grad_(False)
        model.sampler.requires_grad_(False)
        model.encoder.requires_grad_(False)

    tot_loss, tot_root_z, tot_joint19_z, n = 0.0, 0.0, 0.0, 0

    for batch in loader:
        batch_is_ego: EgoBodyItem = ty.cast(EgoBodyItem, batch).to(
            device=args.device, non_blocking=True
        )

        optimizer.zero_grad()

        with tch.cuda.amp.autocast(enabled=args.amp):
            torso_derived_from = tch.tensor([5, 6, 11, 12], device=batch.kps133_cam.device)
            bypass_z_root = tch.mean(batch.kps133_cam[:, torso_derived_from, :], dim=1)
            pred_coco_main_metric_xyz, pred_root_z, uv_conf = model(
                batch_is_ego.depth,
                batch_is_ego.simcc_x,
                batch_is_ego.simcc_y,
                batch_is_ego.simcc_z,
                batch_is_ego.K_inv,
                bypass_z_root,
            )

            loss_z = root_z_loss_fn(pred_root_z, batch_is_ego.kps133_cam).mean()
            # TODO conf scaling
            loss_jt = joint19_z_loss_fn(
                pred_coco_main_metric_xyz, batch_is_ego.kps133_cam, uv_conf
            ).mean()

            loss = loss_z + loss_jt

        scaler.scale(loss).backward()

        if args.grad_clip and args.grad_clip > 0:
            scaler.unscale_(optimizer)
            tch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        tot_loss += float(loss.detach().cpu())
        tot_root_z += float(loss_z.detach().cpu())
        tot_joint19_z += float(loss_jt.detach().cpu())
        n += 1

        if (n + 1) % args.log_every == 0:
            print(
                f"[train e{epoch:03d} it{n:05d}/{len(loader)}] "
                f"loss={tot_loss / n:.6f} root_z={tot_root_z / n:.6f} tot_joint19_z={tot_joint19_z / n:.6f}"
            )
    # except Exception as e:
    #     print("err "+str(e))
    return {
        "loss": tot_loss / max(n, 1),
        "z19": tot_joint19_z / max(n, 1),
        "root_z": tot_root_z / max(n, 1),
    }


def main():
    DEFAULT_LR = 3e-4
    parser = ap.ArgumentParser("Train TinyCenterScale")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/media/thwdpc/MassStorage/EgoBody",
        help="path to train data (dataset-specific)",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--no-amp", dest="amp", action="store_false", default=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7654321)
    parser.add_argument("--ckpt-dir", type=str, default="ckpts_rtm_ada")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--init-from-tf", type=str, default="adapose_weights/density_weights_model"
    )
    parser.add_argument("--train-ada-layers", action="store_true", default=False)
    parser.add_argument("--log-every", type=int, default=4)
    args = parser.parse_args()

    tch.manual_seed(args.seed)
    tch.cuda.manual_seed_all(args.seed)

    model = RTMPoseToAdaPose()
    model.to(args.device)

    optimizer = tch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = tch.cuda.amp.GradScaler(enabled=args.amp)
    scheduler = tch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val = float("inf")

    if args.resume:
        opt_or_not = None
        if math.isclose(args.lr, DEFAULT_LR):
            opt_or_not = optimizer
        ckpt = load_ckpt(
            args.resume, model, opt_or_not, scaler, map_location=args.device
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(
            f"Resumed from {args.resume} @ epoch={start_epoch} best_val={best_val:.6f}, optimizer {'reset for lr' if opt_or_not is None else 'loaded from dict'}"
        )
    elif os.path.exists(args.init_from_tf):
        import tensorflow as tf

        with tch.no_grad():
            weights = tf.train.load_checkpoint(args.init_from_tf)
            model.pre.z_guesser.load_tf_weights(weights)
            model.sampler.load_tf_weights(weights)
            model.encoder.load_tf_weights(weights)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    train_ds = EgoBodyDataset(args.dataset_root, "train")
    val_ds = EgoBodyDataset(args.dataset_root, "val")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda b: EgoBodyItem.collate(b),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda b: EgoBodyItem.collate(b),
    )

    for epoch in range(start_epoch, args.epochs):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, args, epoch)
        # va = validate(model, val_loader, cfg)
        scheduler.step()


if __name__ == "__main__":
    main()
