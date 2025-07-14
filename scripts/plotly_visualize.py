import torch
import plotly.graph_objects as go

tensor = torch.load("tensor_court.pt")
frame = 0

height = tensor[frame, ..., 3].flatten().numpy()
xy = tensor[frame, ..., 4:].reshape(-1, 2).numpy()

mask = height > 0

x = xy[mask, 0]
y = xy[mask, 1]
z = height[mask]

fig = go.Figure(go.Scatter3d(
    x=x, y=y, z=z,
    mode="markers",
    marker=dict(size=2, opacity=0.7)
))
fig.update_layout(scene=dict(
    xaxis_title="X court (m)",
    yaxis_title="Y court (m)",
    zaxis_title="Height (m)",
    aspectmode="data"
))
fig.show()

