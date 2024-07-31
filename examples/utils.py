from __future__ import annotations
from typing import Generator, Sequence, Optional
import os
from pathlib import Path
import contextlib

import numpy as np
import torch
from torch import Tensor
from pytorch3d import transforms as T
import plotly.express as px
import plotly.graph_objects as go


@contextlib.contextmanager
def working_directory(path: str | Path) -> Generator:
    prev_cwd = Path.cwd()
    os.chdir(str(Path(path).resolve()))
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def get_all_named_cmaps() -> list[str]:
    named_colorscales = px.colors.named_colorscales()
    base_cmaps = ["cividis", "sunset", "turbo", "thermal"]
    base_cmaps_idx = [named_colorscales.index(x) for x in base_cmaps]
    for idx in base_cmaps_idx:
        named_colorscales.pop(idx)
    named_colorscales = base_cmaps + named_colorscales
    return named_colorscales


def interactive_plot(
    data: np.ndarray | Sequence[np.ndarray] | Tensor | Sequence[Tensor],
    labels: Optional[str | Sequence[str]] = None,
    point_size: int = 3,
    opacity: float = 0.8,
    color: Optional[np.ndarray] = None,
    cmap: Optional[str] = None,
    constraint_x: bool = False,
    constraint_y: bool = False,
    constraint_z: bool = False,
    return_fig: bool = False,
    width: int = 500,
    height: int = 500,
) -> Optional[go.Figure]:
    """ Interactive plot of point cloud(s) based on Plotly. Can display N pointcloud(s).

    Args:
        pointcloud (Union[np.ndarray, tuple[np.ndarray]]): If a list or tuple is passed,
        each element will be displayed with its own name and colormap.

    Raises:
        ValueError: If too few or too many pointclouds are passed to the function.
    """
    if not isinstance(data, (list, tuple)):
        data = [data]
    if isinstance(data, tuple):
        data = list(data)
    for i, x in enumerate(data):
        if isinstance(x, Tensor) and x.is_cuda:
            data[i] = x.cpu()
    N = len(data)
    if labels is None:
        labels = [f"pointcloud {i + 1}" for i in range(N)]
    if not isinstance(labels, (list, tuple)):
        labels = (labels, )
    if labels is not None and not len(data) == len(labels):
        raise ValueError(f"You gave {len(data)} pointclouds but {len(labels)} labels.")
    all_cmaps = get_all_named_cmaps()
    if isinstance(cmap, str):
        cmaps = N * [cmap]
    elif not (isinstance(cmap, (tuple, list)) and len(cmap) == len(data)):
        cmaps = all_cmaps[:N]
    else:
        cmaps = cmap
    traces = list()
    for pointcloud, label, cmap in zip(data, labels, cmaps):
        x, y, z = pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2]
        c = color if color is not None else z
        marker_kwargs = dict(size=point_size, opacity=opacity, color=c, colorscale=cmap)
        scatter_kwargs = dict(visible=True, mode='markers', name=label, marker=marker_kwargs)
        traces.append(go.Scatter3d(x=x, y=y, z=z, **scatter_kwargs))
    layout = dict(
        width=width, height=height,
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]), margin=dict(t=50)
    )
    if constraint_x:
        layout['scene'] = dict(xaxis=dict(nticks=4, range=[-1, 1]))
    if constraint_y:
        layout['scene'] = dict(yaxis=dict(nticks=4, range=[-1, 1]))
    if constraint_z:
        layout['scene'] = dict(zaxis=dict(nticks=4, range=[-1, 1]))
    fig = go.Figure(data=traces)
    fig.update_layout(**layout)
    if return_fig:
        return fig
    fig.show()


def so3_relative_angle(R1: Tensor, R2: Tensor) -> Tensor:
    """ Geodesic distance in SO(3). R1 & R2 are batches of rotation matrices (B, 3, 3). """
    cos_theta = (torch.einsum('bij,bij->b', R1, R2) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1, 1)
    return torch.acos(cos_theta) * 180 / torch.pi


@torch.no_grad()
def compute_pairwise_errors(R_abs_hat: Tensor, R_abs_gt: Tensor) -> Tensor:
    """ Multiviews registration errors. """
    M_abs_hat_T = T.Transform3d().rotate(R_abs_hat).inverse()
    M_abs_gt = T.Transform3d().rotate(R_abs_gt)
    N = len(R_abs_gt)
    D = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            a = M_abs_gt[i].compose(M_abs_hat_T[i]).get_matrix()[:, :3, :3]
            b = M_abs_gt[j].compose(M_abs_hat_T[j]).get_matrix()[:, :3, :3]
            D[i, j] = so3_relative_angle(a, b)
    return D


def get_errors_index(R_abs_hat: Tensor, R_abs_gt: Tensor, threshold: int = 15) -> Tensor:
    Ea = compute_pairwise_errors(R_abs_hat, R_abs_gt)
    maximal_clique_idx = ((Ea <= threshold).sum(dim=0)).argmax()  # row of the maximal clique
    errors_idx = (Ea[maximal_clique_idx] > threshold).argwhere().squeeze(dim=1)
    return errors_idx
