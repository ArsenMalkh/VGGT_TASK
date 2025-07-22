import argparse
import torch
import plotly.graph_objects as go

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor", default="tensor_court.pt")
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()

    tensor = torch.load(args.tensor)
    frame = tensor[args.frame]

    height = frame[..., 3].flatten().numpy()
    xy = frame[..., 4:].reshape(-1, 2).numpy()

    mask = height > 0.0
    x = xy[mask, 0]
    y = xy[mask, 1]
    z = height[mask]

    fig = go.Figure(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=2, opacity=0.7),
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Court X (m)",
            yaxis_title="Court Y (m)",
            zaxis_title="Height (m)",
            aspectmode="data",
        ),
    )
    fig.show()

if __name__ == "__main__":
    main()
