from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

from vggt.models.vggt import VGGT

@torch.no_grad()
def pad_to_batch(imgs: List[torch.Tensor]) -> tuple[torch.Tensor, List[tuple[int, int]]]:
    max_h = max(t.shape[1] for t in imgs)
    max_w = max(t.shape[2] for t in imgs)
    padded, shapes = [], []
    for img in imgs:
        h, w = img.shape[1:]
        shapes.append((h, w))
        padded.append(F.pad(img, (0, max_w - w, 0, max_h - h)))
    return torch.stack(padded), shapes

@torch.no_grad()
def run_batch(
    model: torch.nn.Module,
    batch_rgb: List[torch.Tensor],
    *,
    device: str,
    fp16: bool,
) -> List[torch.Tensor]:
    imgs_pad, shapes = pad_to_batch(batch_rgb)
    if fp16:
        imgs_pad = imgs_pad.half()

    preds = model(imgs_pad.to(device))
    if "world_points" not in preds:
        raise RuntimeError("Checkpoint missing 'world_points' key")

    world = preds["world_points"]
    if world.dim() == 4 and world.shape[1] == 3:
        xyz = world
    elif world.dim() == 4 and world.shape[-1] == 3:
        xyz = world.permute(0, 3, 1, 2).contiguous()
    elif world.dim() == 5 and world.shape[1] == 1 and world.shape[-1] == 3:
        xyz = world.squeeze(1).permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"Unexpected shape for world_points: {world.shape}")
    

    rgbxyz_pad = torch.cat([imgs_pad.cpu(), xyz.cpu()], dim=1)

    out: List[torch.Tensor] = []
    for i, (h, w) in enumerate(shapes):
        out.append(rgbxyz_pad[i, :, :h, :w])
    return out

def infer_video(
    model: torch.nn.Module,
    video: str | Path,
    *,
    device: str,
    batch_size: int,
    resize_edge: int | None,
    patch_size: int,
    fp16: bool,
    dump_first: str | None,
) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    frames, batch = [], []
    dumped = False
    with tqdm(total=total, unit="f", desc="VGGT") as bar:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            h0, w0 = bgr.shape[:2]
            if resize_edge and max(h0, w0) > resize_edge:
                s = resize_edge / max(h0, w0)
                bgr = cv2.resize(bgr, (int(w0 * s), int(h0 * s)))
            pad_h = (-bgr.shape[0]) % patch_size
            pad_w = (-bgr.shape[1]) % patch_size
            if pad_h or pad_w:
                bgr = cv2.copyMakeBorder(
                    bgr, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
                )

            if dump_first and not dumped:
                cv2.imwrite(dump_first, bgr)
                dumped = True

            rgb = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255
            batch.append(rgb)

            if len(batch) == batch_size:
                frames.extend(
                    run_batch(model, batch_rgb=batch, device=device, fp16=fp16)
                )
                batch.clear()
            bar.update(1)

        if batch:
            frames.extend(run_batch(model, batch_rgb=batch, device=device, fp16=fp16))

    cap.release()
    return torch.stack(frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--resize", type=int, default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--dump_frame", default=None)
    ap.add_argument("--img_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--embed_dim", type=int, default=1024)
    ap.add_argument("--save_rgbxyz", default="rgbxyz_tensor.pt")
    args = ap.parse_args()

    ckpt = Path(args.weights)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    model = VGGT(
        img_size=args.img_size, patch_size=args.patch_size, embed_dim=args.embed_dim
    ).to(args.device)
    if args.fp16:
        model = model.half()

    state = torch.load(ckpt, map_location=args.device)
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()

    tensor = infer_video(
        model,
        args.video,
        device=args.device,
        batch_size=args.batch_size,
        resize_edge=args.resize,
        patch_size=args.patch_size,
        fp16=args.fp16,
        dump_first=args.dump_frame,
    )

    z_min = tensor[0, 5].min().item()
    z_max = tensor[0, 5].max().item()
    print(f"Z‑range of the first frame: {z_min:.2f} … {z_max:.2f} m")

    Path(args.save_rgbxyz).parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, args.save_rgbxyz)
    print("✓ Saved →", args.save_rgbxyz)


if __name__ == "__main__":
    main()
