from pathlib import Path
import argparse
import numpy as np
import torch


#Estimate the court plane (z = ax + by + d) from the first frame.
def estimate_court_plane(xyz: torch.Tensor) -> torch.Tensor:
    H, W = xyz.shape[1:]
    coords = xyz.permute(1, 2, 0).reshape(-1, 3)
    z_q20 = torch.quantile(coords[:, 2], 0.20)
    plane_pts = coords[coords[:, 2] <= z_q20]
    A = torch.cat([plane_pts[:, :2], torch.ones_like(plane_pts[:, :1])], dim=1)
    params, *_ = torch.linalg.lstsq(A, plane_pts[:, 2:3])
    return params.squeeze().float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgbxyz", default="rgbxyz_tensor.pt")
    ap.add_argument("--hmat", default="H.npy")
    ap.add_argument("--out", default="tensor_court.pt")
    args = ap.parse_args()

    rgbxyz = torch.load(args.rgbxyz)
    T, _, H, W = rgbxyz.shape
    xyz = rgbxyz[:, 3:]

    a, b, d = estimate_court_plane(xyz[0])
    normal = torch.tensor([a, b, -1.0])
    norm_normal = normal.norm()

    # Compute orthogonal height above court plane for each point
    xyz_flat = xyz.permute(0, 2, 3, 1)
    height = (xyz_flat @ normal).add(d) / norm_normal
    height = height.clamp_min(0.0)

    # Compute homography-based court coordinates (X_court, Y_court)
    xs, ys = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    grid = torch.stack([xs, ys, torch.ones_like(xs)], 0).float()
    XY = torch.from_numpy(np.load(args.hmat)).float() @ grid.view(3, -1)
    XY = (XY[:2] / XY[2]).T.view(H, W, 2)
    XY = XY.unsqueeze(0).repeat(T, 1, 1, 1)

    rgb = rgbxyz[:, :3].permute(0, 2, 3, 1)
    depth = height.unsqueeze(-1)
    tensor_out = torch.cat([rgb, depth, XY], dim=-1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor_out, args.out)
    print("✓", args.out, tensor_out.shape)

if __name__ == "__main__":
    main()