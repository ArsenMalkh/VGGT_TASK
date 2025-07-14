#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

from vggt.models.vggt import VGGT

def pad_to_batch(imgs: List[torch.Tensor]) -> tuple[torch.Tensor, List[tuple[int, int]]]:
    max_h = max(t.shape[1] for t in imgs)
    max_w = max(t.shape[2] for t in imgs)
    padded, shapes = [], []
    for img in imgs:
        h, w = img.shape[1:]
        shapes.append((h, w))
        padded.append(F.pad(img, (0, max_w - w, 0, max_h - h)))
    return torch.stack(padded), shapes

def run_batch(model: torch.nn.Module, batch_rgb: List[torch.Tensor], *, device: str, fp16: bool) -> List[torch.Tensor]:
    imgs_pad, shapes = pad_to_batch(batch_rgb)
    if fp16:
        imgs_pad = imgs_pad.half()
    with torch.inference_mode():
        preds = model(imgs_pad.to(device))
    if "world_points" not in preds:
        raise RuntimeError("Checkpoint missing 'world_points'")
    height = preds["world_points"][:, 0, ..., 2].cpu()
    if fp16:
        height = height.half()
    rgbh = torch.cat([imgs_pad.cpu(), height.unsqueeze(1)], dim=1)
    out: List[torch.Tensor] = []
    for i, (h, w) in enumerate(shapes):
        out.append(rgbh[i, :, :h, :w])
    return out

def infer_video(model: torch.nn.Module, video: str | Path, *, device: str, batch_size: int, resize_edge: int | None, patch_size: int, fp16: bool, dump_first: str | None) -> torch.Tensor:
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
            h, w = bgr.shape[:2]
            if resize_edge and max(h, w) > resize_edge:
                scale = resize_edge / max(h, w)
                bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
            pad_h = (-bgr.shape[0]) % patch_size
            pad_w = (-bgr.shape[1]) % patch_size
            if pad_h or pad_w:
                bgr = cv2.copyMakeBorder(bgr, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            if dump_first and not dumped:
                cv2.imwrite(dump_first, bgr)
                dumped = True
            rgb_np = bgr[:, :, ::-1].copy()
            rgb = torch.from_numpy(rgb_np).permute(2, 0, 1).float().div_(255)
            batch.append(rgb)
            if len(batch) == batch_size:
                frames.extend(run_batch(model, batch_rgb=batch, device=device, fp16=fp16))
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
    ap.add_argument("--save_rgbd", default="rgbd_tensor.pt")
    args = ap.parse_args()

    ckpt = Path(args.weights)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    model = VGGT(img_size=args.img_size, patch_size=args.patch_size, embed_dim=args.embed_dim).to(args.device)
    if args.fp16:
        model = model.half()

    state = torch.load(ckpt, map_location=args.device)
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()

    tensor = infer_video(model, args.video, device=args.device, batch_size=args.batch_size,
                         resize_edge=args.resize, patch_size=args.patch_size,
                         fp16=args.fp16, dump_first=args.dump_frame)

    h_min, h_max = tensor[0, 3].min().item(), tensor[0, 3].max().item()
    print(f"Height range 1st frame: {h_min:.2f} … {h_max:.2f} m")

    Path(args.save_rgbd).parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, args.save_rgbd)
    print("✓ Saved →", args.save_rgbd)

if __name__ == "__main__":
    main()

