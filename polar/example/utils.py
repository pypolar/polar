from __future__ import annotations
from typing import Generator, Sequence, Optional
from pathlib import Path
from random import randint
import os
from pathlib import Path
import contextlib

import numpy as np
import torch
from torch import Tensor
from pytorch3d import transforms as T
import plotly.express as px
import plotly.graph_objects as go
from ..so3 import so3_relative_angle


# _________________________________________________________________________________________________________ #

def load_sample_data(*keys: str) -> list[Tensor | float] | dict:
    """ Accepted keys:
        'all', 'config', 'R_abs_gt', 'template', 'anisotropy', 'cropped_noisy', 'outliers'
        - If 'all' or len(keys) > 3:
            return the whole data dict
        - If ('anisotropy', 'cropped_noisy', 'outliers') in keys:
            return (degraded views, degradation value, R_abs_gt) (+ optionally config, template)

    Returns:
        TODO
    """
    here = Path(__file__).parent
    data = torch.load(here / 'data.pt')
    if 'all' in keys or len(keys) > 3:
        return data
    degradations = ('anisotropy', 'cropped_noisy', 'outliers')
    if len(keys) == 0:
        i = randint(0, 2)
        keys = (degradations[i],)
    msg = "Only one degradations can be specified. Call `load_sample_data` several times or pass 'all'."
    assert sum([k in degradations for k in keys]) <= 1, msg
    return_values = list()
    views = None
    for k in degradations:
        if k in keys:
            views = data['views'][k]
            value = data['config'][k]
    if views is not None:
        return_values.append(views)
        return_values.append(value)
        return_values.append(data['R_abs_gt'])
    for key in ('config', 'template'):
        if key in keys:
            return_values.append(data[key])
    return return_values


# _________________________________________________________________________________________________________ #

@contextlib.contextmanager
def working_directory(path: str | Path) -> Generator:
    prev_cwd = Path.cwd()
    os.chdir(str(Path(path).resolve()))
    try:
        yield
    finally:
        os.chdir(prev_cwd)


# _________________________________________________________________________________________________________ #

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
        data (np.ndarray | Sequence[np.ndarray] | Tensor | Sequence[Tensor]): _description_
        labels (Optional[str  |  Sequence[str]], optional): _description_. Defaults to None.
        point_size (int, optional): _description_. Defaults to 3.
        opacity (float, optional): _description_. Defaults to 0.8.
        color (Optional[np.ndarray], optional): _description_. Defaults to None.
        cmap (Optional[str], optional): _description_. Defaults to None.
        constraint_x (bool, optional): _description_. Defaults to False.
        constraint_y (bool, optional): _description_. Defaults to False.
        constraint_z (bool, optional): _description_. Defaults to False.
        return_fig (bool, optional): _description_. Defaults to False.
        width (int, optional): _description_. Defaults to 500.
        height (int, optional): _description_. Defaults to 500.

    Raises:
        ValueError: _description_

    Returns:
        Optional[go.Figure]: _description_
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


# _________________________________________________________________________________________________________ #

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
