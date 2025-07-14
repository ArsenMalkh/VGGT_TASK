#!/usr/bin/env python3

from pathlib import Path
import argparse
import torch
import numpy as np

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgbh", default="rgbd_tensor.pt")
    ap.add_argument("--hmat", default="H.npy")
    ap.add_argument("--out", default="tensor_court.pt")
    args = ap.parse_args()

    rgbh = torch.load(args.rgbh)
    T, C, H, W = rgbh.shape
    assert C == 4

    xs, ys = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    grid = torch.stack([xs, ys, torch.ones_like(xs)], 0).float()
    XY = torch.from_numpy(np.load(args.hmat)).float() @ grid.view(3, -1)
    XY = (XY[:2] / XY[2]).T.view(H, W, 2)
    XY = XY.unsqueeze(0).repeat(T, 1, 1, 1)

    rgbh_hw = rgbh.permute(0, 2, 3, 1).contiguous()
    tensor_out = torch.cat([rgbh_hw, XY], dim=-1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor_out, args.out)
    print("✓", args.out, tensor_out.shape)

if __name__ == "__main__":
    main()

