from __future__ import annotations
from typing import TypedDict, cast
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import pytorch3d.transforms as T


# +---------------------------------------------------------------------------------------------+ #
# |                                   S0(3) GEODESIC DISTANCE                                   | #
# +---------------------------------------------------------------------------------------------+ #

def so3_relative_angle(R1: Tensor, R2: Tensor) -> Tensor:
    r""" Geodesic distance in the manifold of rotations SO(3):
    $\operatorname{arccos}\left(\frac{\mathrm{tr}(R_1 R_2^\top) - 1}{2}\right)$

    Args:
        R1 (Tensor): Batch of rotation matrices $(B, 3, 3)$.
        R2 (Tensor): Batch of rotation matrices $(B, 3, 3)$.

    Returns:
        Distances: Batch of distances $(B,)$ in $[0, \pi]$.
    """
    cos_theta = (torch.einsum('bij,bij->b', R1, R2) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1, 1)
    return torch.acos(cos_theta) * 180 / torch.pi


# +---------------------------------------------------------------------------------------------+ #
# |                                       SO(3) SAMPLING                                        | #
# +---------------------------------------------------------------------------------------------+ #

def super_fibonacci_spirals(n: int | float, double: bool = False) -> Tensor:
    """ Generate $n$ uniformly spaced rotations.<br>
    [Alexa M., Super-Fibonacci Spirals: Fast, Low-Discrepancy Sampling of SO3, CVPR 2022.](https://openaccess.thecvf.com/content/CVPR2022/papers/Alexa_Super-Fibonacci_Spirals_Fast_Low-Discrepancy_Sampling_of_SO3_CVPR_2022_paper.pdf)

    Args:
        n (int | float): Number of samples to generate.
        double (bool, optional):
            If `True`, generated samples are of type `double`. Defaults to `False`.

    Returns:
        Rotations: Batch of generated rotations $(n, 3, 3)$.
    """
    n = int(n)
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041
    Q = np.empty(shape=(n, 4), dtype=np.float64 if double else np.float32)
    i = np.arange(n)
    s = i + 0.5
    r = np.sqrt(s / n)
    R = np.sqrt(1.0 - s / n)
    alpha = 2.0 * np.pi * s / phi
    beta = 2.0 * np.pi * s / psi
    Q[i, 0] = r * np.sin(alpha)
    Q[i, 1] = r * np.cos(alpha)
    Q[i, 2] = R * np.sin(beta)
    Q[i, 3] = R * np.cos(beta)
    matrices = T.quaternion_to_matrix(torch.tensor(Q))
    angles = T.matrix_to_euler_angles(matrices, 'ZXZ')
    return angles


# +---------------------------------------------------------------------------------------------+ #
# |                            SO(3) KNN GRAPH & LOCAL MINIMA SEARCH                            | #
# +---------------------------------------------------------------------------------------------+ #

class Graph(TypedDict):
    indices: Tensor
    relative_angles: Tensor


def make_so3_knn_graph(so3: Tensor, K: int = 8) -> Graph:
    """ Note that each rotation is NOT its own neighbor. """
    so3_knn_graph = dict()
    R2 = T.euler_angles_to_matrix(so3, "ZXZ")
    for i, angle in enumerate(so3):
        so3_knn_graph[i] = dict()
        R1 = T.euler_angles_to_matrix(angle, "ZXZ").repeat(len(so3), 1, 1)
        relative_angles = so3_relative_angle(R1, R2)
        values, indices = relative_angles.sort()
        so3_knn_graph[i] = dict(indices=indices[1:K+1], relative_angles=values[1:K+1])
    return cast(Graph, so3_knn_graph)


def save_so3_knn_graph(so3_knn_graph: Graph, L: int, K: int, output_dir: str = ".cache/") -> None:
    """ From a dict of pairs: idx: {'indices': tensor(indices), 'angles': tensor(angles},
        I create two tensors Indices and Angles (by stacking in order), because it's much more
        lightweight this way.
    """
    stack = lambda x: torch.stack([so3_knn_graph[k][x] for k in sorted(so3_knn_graph.keys())])
    indices_table = stack('indices')
    angles_table = stack('relative_angles')
    here = Path(__file__).resolve().parent
    output_path = here / output_dir
    output_path.mkdir(exist_ok=True)
    filename = f"so3_knn_lookup_table_indices_L={L}_K={K}.pt"
    torch.save(indices_table, output_path / filename)
    filename = f"so3_knn_lookup_table_angles_L={L}_K={K}.pt"
    torch.save(angles_table, output_path / filename)


def make_and_save_so3_knn_lookup_table(L: int, K: int, output_dir: str = ".cache/") -> None:
    here = Path(__file__).resolve().parent
    output_path = here / output_dir
    output_path.mkdir(exist_ok=True)
    filename = f"so3_knn_lookup_table_L={L}_K={K}.pt"
    output_path = output_path / filename
    so3 = super_fibonacci_spirals(L)
    so3_knn_graph = make_so3_knn_graph(so3, K)
    save_so3_knn_graph(so3_knn_graph, L, K, output_dir)


def get_so3_knn_graph(L: int | float, K: int, verbose: bool = False, output_dir: str = ".cache/", ) -> Graph:
    """ Get a k nearest neighbors graph over rotations. Load it from disk if it exists, create it otherwise.

    !!! Warning
        This function will search for existing graphs in `output_dir.` If a new graph must be created,
        it may takes quite a long time, depending on values of `L` and `K`.


    Args:
        L (int | float): Number of nodes (rotations) in the graph.
        K (int): Number of neighbors.
        verbose (bool, optional): If `True`, display infos about the graph retrieving. Defaults to `False`.
        output_dir (str, optional): Where to look for or store the graph. Defaults to `".cache/"`.

    Returns:
        Graph: $SO(3)$ knn graph. Keys: `indices`, `relative_angles`
    """
    L = int(L)
    here = Path(__file__).resolve().parent
    output_path = here / output_dir
    filename = f"so3_knn_lookup_table_indices_L={L}_K={K}.pt"
    if not (output_path / filename).exists() and verbose:
        print((
            "Cached SO(3) Lookup Table not found. "
            "It will be created on the fly and saved for later use. "
            "This will occur extra runtime."
        ))
        make_and_save_so3_knn_lookup_table(L, K, output_dir)
    elif verbose:
        print("    - Using cached SO(3) knn-graph.")
    path = output_path / f"so3_knn_lookup_table_indices_L={L}_K={K}.pt"
    indices = torch.load(path)
    path = output_path / f"so3_knn_lookup_table_angles_L={L}_K={K}.pt"
    angles = torch.load(path)
    knn_graph = {i: {'indices': indices[i],
                     'relative_angles': angles[i]} for i in range(len(indices))}
    return cast(Graph, knn_graph)


def single_view_find_so3_local_minima_from_knn_graph(so3_knn_graph: Graph, values: Tensor) -> Tensor:
    """ Use a knn SO(3) graph (from [polar.so3.get_so3_knn_graph][]) of $L$ rotations to find local minima in 
    one single set of $L$ `values` associated to these rotations.

    Args:
        so3_knn_graph (Graph): Graph where nodes are rotations. Each node is connected to its k nearest neighbors
            (according to the [polar.so3.so3_relative_angle][] distance.)
        values (Tensor): Values in which to find local minima $(L,)$.

    Returns:
        LocalMinima: Tensor of local minima indices in range $[0, L]$.
    """
    neighbors = torch.stack([n['indices'] for n in so3_knn_graph.values()])  # type: ignore
    local_minima_indices = (values[:, None] <= values[neighbors]).all(dim=1).argwhere().squeeze()
    return local_minima_indices[values[local_minima_indices].argsort()]


def parallel_find_so3_local_minima_from_knn_graph(so3_knn_graph: Graph, values: Tensor) -> list[Tensor]:
    """ All views in parallel: values is a tensor (num_views, so3_sampling_size). """
    neighbors = torch.stack([n['indices'] for n in so3_knn_graph.values()])  # type: ignore
    # boolean (N, L) =   (N, L, 1)     <=       (L, K)         (all)
    local_minima = (values[:, :, None] <= values[:, neighbors]).all(dim=2)
    local_minima_indices = [l.argwhere().squeeze(dim=1) for l in local_minima]
    local_minima_indices_sorted = [l[v[l].argsort()] for l, v in zip(local_minima_indices, values)]
    return local_minima_indices_sorted


def find_so3_local_minima_from_knn_graph(so3_knn_graph: Graph, values: Tensor, parallel: bool) -> list[Tensor]:
    """ Use a knn SO(3) graph (from [polar.so3.get_so3_knn_graph][]) of $L$ rotations to find local minima in 
    $N$ sets of $L$ `values` associated to these rotations.

    Args:
        so3_knn_graph (Graph): Graph where nodes are rotations. Each node is connected to its k nearest neighbors
            (according to the [polar.so3.so3_relative_angle][] distance.)
        values (Tensor): Values in which to find local minima $(N, L)$.
        parallel (bool): If `True`, local minima in each of the `N` sets of `L` distances are searched in parallel.
            Otherwise, run $N$ sequential searches.

    Returns:
        LocalMinima: List of $N$ local minima indices in range $[0, L]$.  
    """
    if parallel:
        return parallel_find_so3_local_minima_from_knn_graph(so3_knn_graph, values)
    return [single_view_find_so3_local_minima_from_knn_graph(so3_knn_graph, v) for v in values]
