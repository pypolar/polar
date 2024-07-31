"""
    In all the following:
    - N = num objects (num views)
    - P = num points after decoding (i.e. d(z) is (P, 3))
    - D = latent dimension
    - L = number of rotation uniformly sampled in SO(3)
    - K = number of neighbors in SO(3) K-NN graph
"""

from __future__ import annotations
from pathlib import Path
from functools import wraps
from time import perf_counter
from abc import abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Callable, Sequence, Any, cast
from typing_extensions import TypeAlias
from tqdm.auto import tqdm

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions import MultivariateNormal
from torch_geometric import compile
from torch_geometric.data import Batch
from pytorch3d import transforms as T
from pytorch3d.loss import chamfer_distance

from .network import PointNetAE, AE_BASELINE_WEIGHTS
from . import so3


Views: TypeAlias = "Tensor | Sequence[Tensor]"


# +---------------------------------------------------------------------------------------------+ #
# |                                       SMALL UTILITIES                                       | #
# +---------------------------------------------------------------------------------------------+ #

def exists(x: Any) -> bool:
    return x is not None


def repeat(x: Tensor, k: int) -> Tensor:
    return x.repeat(k, *(x.ndim * (1,)))


def flatten(x: Tensor) -> Tensor:
    return x.flatten(start_dim=0, end_dim=1)


def unflatten(x: Tensor, sizes: tuple[int, int]) -> Tensor:
    return x.unflatten(0, sizes)


def d_C(X: Views, Y: Views, **kwargs) -> Views:
    cd = lambda x, y: chamfer_distance(x, y, batch_reduction=None, **kwargs)[0]
    if isinstance(X, Tensor) and isinstance(Y, Tensor):
        return cast(Tensor, cd(X, Y))
    distances = [cd(x[None, ...], y[None, ...]) for x, y in zip(X, Y)]
    distances = [cast(Tensor, d).squeeze() for d in distances]
    return distances


def pointwise_single_directional_chamfer(X: Views, Y: Views) -> Views:
    return d_C(X, Y, single_directional=True, point_reduction=None)


def center_max_norm(views: Views) -> Views:
    # center
    views = [v - v.mean(dim=0) for v in views]
    # max norm
    max_norm = max([v.norm(dim=1).max() for v in views])  # type: ignore
    views = [v / max_norm for v in views]
    return views


def timeit(verbose: bool = False) -> Callable:
    def timeit_inner(function: Callable) -> Callable:
        @wraps(function)
        def timeit_wrapper(*args, **kwargs) -> Any:
            tic = perf_counter()
            result = function(*args, **kwargs)
            toc = perf_counter()
            if verbose:
                print(f'[TIMER: Function {function.__name__}(): {toc - tic:.4f}s]')
            return result
        return timeit_wrapper
    return timeit_inner


# +---------------------------------------------------------------------------------------------+ #
# |                                          STRUCTURES                                         | #
# +---------------------------------------------------------------------------------------------+ #

# __________________________________ 1. AE wrappers & interface _________________________________ #

class AbstractAE(torch.nn.Module):

    @abstractmethod
    def encode(self, x: Batch | Views) -> Tensor:
        pass

    @abstractmethod
    def decode(self, x: Batch | Tensor) -> Tensor:
        pass


class FrozenAE:

    def __init__(self, ae: AbstractAE, batch_size: int) -> None:
        self.ae = ae.eval()
        self.freeze()
        self.batch_size = batch_size

    def freeze(self) -> None:
        for p in self.ae.parameters():
            p.requires_grad_(False)

    def split(self, X: Views | Batch) -> list[list[Tensor]]:
        indices = torch.arange(len(X)).split(self.batch_size)
        return [[X[i] for i in idx] for idx in indices]

    def encode(self, X: Batch | Views) -> Tensor:
        if isinstance(X, Tensor) and X.ndim == 2:
            X = X.unsqueeze(0)
        z = [self.ae.encode(batch) for batch in self.split(X)]
        return torch.vstack(z).squeeze()

    def decode(self, z: Tensor) -> Tensor:
        if z.ndim == 1:
            z = z.unsqueeze(0)
        x_hat = [self.ae.decode(batch) for batch in z.split(self.batch_size)]
        return torch.vstack(x_hat).squeeze()


@torch.no_grad()
def check_ae_interface(model_init_ae_method: Callable) -> Callable:
    @wraps(model_init_ae_method)
    def wrapper(self):
        ae = self.base_ae.eval().to(self.device)
        class_name = self.base_ae.__class__.__name__
        x = [torch.randn(1024, 3), torch.randn(512, 3)]
        x = list(map(lambda x: x.to(self.device, non_blocking=True), x))
        try:
            z = ae.encode(x)
        except AttributeError:
            print((
                f"The provided AutoEncoder ({class_name}) doesn't implement an encode() method. "
                "Exiting."
            ))
            return
        except RuntimeError:
            print((
                "There was a RuntimeError when testing the encode() method of the provided "
                f"AutoEncoder ({class_name}). Please check that your AutoEncoder "
                "encode() method accepts a list of tensors of shape (n_i, 3). Exiting."
            ))
            return
        if not isinstance(z, Tensor):
            print((
                f"Your AutoEncoder ({class_name}) encode() method returned {type(z)}. "
                "It must return a single tensor of shape (batch_size, latent_dim). "
                "Exiting."
            ))
            return
        try:
            x_hat = ae.decode(z)
        except AttributeError:
            print((
                f"The provided AutoEncoder ({class_name}) doesn't implement a decode() method. "
                "Exiting."
            ))
            return
        except RuntimeError:
            print((
                "There was a RuntimeError when testing the decode() method of the provided "
                f"AutoEncoder ({class_name}). Please check that your AutoEncoder decode() method "
                "accepts a single input tensor of shape (batch_size, latent_dim). Exiting."
            ))
            return
        if not isinstance(x_hat, Tensor):
            print((
                f"Your AutoEncoder ({class_name}) decode() method returned {type(x_hat)}. It must "
                "return a a single tensor representing the reconstructed point cloud(s). Exiting."
            ))
            return
        return model_init_ae_method(self)
    return wrapper


# ____________________________________ 2. POLAR's components ____________________________________ #

@dataclass
class Config:
    batch_size: int
    lr: float
    patience: int
    max_optim_steps: int
    last_joint_fit: bool
    loss: str
    fast_approx: bool
    estimate_translation: bool
    estimate_scaling: bool
    L: int | float
    K: int
    R_eps: int
    topk: int
    parallel_minima_search: bool
    sigmas: float | tuple[float, float, float] | None
    keep_ratio: float | None
    outliers_ratio: float | None
    density_radius: float | None
    density_weight: float | None
    log_params: bool
    verbose: bool
    pbar: bool
    time: bool


@dataclass
class SO3:
    sampling: Tensor
    transforms: T.Rotate
    graph: so3.Graph


@dataclass
class Params:
    z: Tensor
    alpha: Tensor
    beta: Tensor
    theta: Tensor

    @property
    def sim3(self) -> tuple[Tensor, Tensor, Tensor]:
        return (self.alpha, self.beta, self.theta)

    def to_transforms(self) -> T.Transform3d:
        alpha, beta, theta = self.sim3
        R = T.euler_angles_to_matrix(theta, 'ZXZ')
        t = T.Transform3d(device=self.z.device, dtype=self.z.dtype)
        return t.scale(alpha).rotate(R).translate(beta)

    def clone(self) -> Params:
        z, alpha, beta, theta = map(lambda x: x.clone(), (self.z, *self.sim3))
        return Params(z, alpha, beta, theta)

    def detach(self) -> Params:
        z, alpha, beta, theta = map(lambda x: x.detach(), (self.z, *self.sim3))
        return Params(z, alpha, beta, theta)

    def index_update_(self, new_params: Params, indices: Tensor | Sequence[int]) -> Params:
        """ Acts inplace. """
        if not isinstance(indices, Tensor):
            indices = torch.tensor(indices).long().to(self.z.device)
        for i, idx in enumerate(indices):
            self.alpha[idx] = new_params.alpha[i].clone()
            self.beta[idx] = new_params.beta[i].clone()
            self.theta[idx] = new_params.theta[i].clone()
        return self

    def requires_grad_(self) -> Params:
        """ Acts inplace. """
        for k in ('z', 'alpha', 'beta', 'theta'):
            getattr(self, k).requires_grad_()
        return self


@dataclass
class Memory:
    best_losses: Tensor
    losses: Tensor | list[Tensor] = field(default_factory=lambda: list())
    alpha: Tensor | list[Tensor] = field(default_factory=lambda: list())
    beta: Tensor | list[Tensor] = field(default_factory=lambda: list())
    theta: Tensor | list[Tensor] = field(default_factory=lambda: list())
    z: Tensor | list[Tensor] = field(default_factory=lambda: list())
    keep_ratio: Tensor | list[Tensor] = field(default_factory=lambda: list())
    lr: Tensor | list[Tensor] = field(default_factory=lambda: list())
    steps: int = 0
    multistart: bool = False
    improvements: Tensor = torch.empty(0)


# _______________________________________________________________________________________________ #

class EarlyStopping:

    """ Early stopping mechanism with patience on moving average. """

    def __init__(self, model: POLAR) -> None:
        self.best_loss = torch.inf
        self.losses = list()
        self.moving_avg = list()
        self.no_improve_count = 0
        self.converged = False
        self.patience = model.patience
        self.i = 0

    def step(self, last_loss: Tensor) -> None:
        last_loss_value: float = last_loss.detach().item()
        self.losses.append(last_loss_value)
        self.i += 1
        if self.i <= self.patience:
            return
        window = self.losses[self.i - self.patience : self.i]
        avg = sum(window) / len(window)
        self.moving_avg.append(avg)
        diff = avg - self.best_loss
        if diff >= 0 or abs(diff) <= 0.001 * avg:
            self.no_improve_count += 1
        else:
            self.best_loss = min(last_loss_value, self.best_loss)
            self.no_improve_count = 0
        self.converged = self.no_improve_count == self.patience


# +---------------------------------------------------------------------------------------------+ #
# |                                      MAIN MODEL CLASS                                       | #
# +---------------------------------------------------------------------------------------------+ #

class POLAR:

    """ Constructor of the POLAR estimator class. """

    def __init__(
        self,
        # base
        ae: AbstractAE | None = None,
        batch_size: int = 2048,
        # fit
        lr: float = 1e-2,
        patience: int = 100,
        max_optim_steps: int = 1_000,
        last_joint_fit: bool = True,
        loss: str = 'latent',
        # multistart
        fast_approx: bool = True,
        topk: int = 4,
        parallel_minima_search: bool = True,
        R_eps: int = 15,
        # SO(3)
        L: float = 5e4,
        K: int = 256,
        # motions
        estimate_translation: bool = True,
        estimate_scaling: bool = False,
        # degradations model
        sigmas: float | tuple[float, float, float] | None = None,
        keep_ratio: float | None = None,
        outliers_ratio: float | None = None,
        # regularization
        density_weight: float = 1e-2,
        density_radius: float = 0.1,
        # log
        log_params: bool = False,
        verbose: bool = True,
        time: bool = False,
        # progress bar
        pbar: bool = False
    ) -> None:
        r"""_summary_

        Args:
            ae (AbstractAE | None, optional):
                Accept any autoencoder that implements `encode()` and `decode()` methods, and encode a point cloud
                using a global descriptor (see [polar.network.model.PointNetAE][]). If `None`, load an instance of
                PointNetAE pretrained on ModelNet40. Defaults to `None`.
            batch_size (int, optional): Used throughout the whole optimization. Defaults to `2048`.
            lr (float, optional): Learning rate. Defaults to `1e-2`.
            patience (int, optional):
                For each joint or multistart fit, stop optimization after `max_optim_steps` if early stopping wasn't
                triggered. Defaults to `1_000`.
            max_optim_steps (int, optional):
                Each optimization (joint or parallel multistart) will stop after `max_optim_steps` if early stopping
                wasn't triggered. Defaults to 1_000.
            last_joint_fit (bool, optional):
                If `True`, run a final joint fit after the multistart converged. Defaults to `True`.
            loss (str, optional): 'latent' or 'ambient'.
                Criterion used for both joint and parallel multistart optimizations. For each view $\mathrm{X_i}$, if
                'latent', compute $\lVert \mathrm{e}(\rho_i \mathrm{d}(z)) - \mathrm{e}(\mathrm{X_i}) \rVert$, else
                compute $d_{\operatorname{CD}}(\rho_i \mathrm{d}(z), \mathrm{X_i})$, where $d_{\operatorname{CD}}$
                denotes the Chamfer distance. Defaults to 'latent'.
            fast_approx (bool, optional):
                If `True`, do not degrade the estimated template during the local minima search. Defaults to `True`.
            topk (int, optional): Number of local minima to try per view. Defaults to `4`.
            parallel_minima_search (bool, optional):
                If `True`, search local minima for each view in parallel. Faster, but may require a lot of VRAM if
                there are many views and / or `topk`. Defaults to `True`.
            R_eps (int, optional): Relative angle threshold to detect an escape from a local minima. Defaults to `15`.
            L (float, optional): _description_. Defaults to 5e4.
            K (int, optional): Number of neighbors of a rotation for the local minima search. Defaults to `256`.
            estimate_translation (bool, optional): If False, only estimate 3D rotations. Defaults to True.
            estimate_scaling (bool, optional):
                If `True`, estimate a scaling parameter per view, in addition to rotation and translation.
                Defaults to `False`.
            sigmas (float | tuple[float, float, float] | None, optional):
                Noise diagonal covariances. If a single float is provided, assume isotropic noise. Else, specify
                the variance for each axis. Defaults to None.
            keep_ratio (float | None, optional):
                Ratio of points that are not occluded in each view, in [0, 1]. Defaults to None.
            outliers_ratio (float | None, optional): Ratio of points to segment out, in [0, 1]. Defaults to `None`.
            density_weight (float, optional):
                If > 0, will apply a regularization term in the loss. Specify the weight of this ponderation.
                Defaults to 1e-2.
            density_radius (float, optional):
                When `density_weight` is > 0, a regularization term will be computed. This regularization is the
                average point density, ie the average number of point in balls of radius `density_radius`.
                Defaults to 0.1.
            log_params (bool, optional):
                If `True`, logs the evolution of estimated template of motions during the optimization. Slightly slow
                down the process. Defaults to False.
            verbose (bool, optional): If `True`, print optimization steps. Defaults to True.
            time (bool, optional):
                If `True`, log the execution time of each function (mostly for debugging). Defaults to False.
            pbar (bool, optional): If `True`, display a progress bar for each optimization loop. Defaults to False.
        """
        self.config = Config(batch_size, lr, patience, max_optim_steps, last_joint_fit, loss,
                             fast_approx, estimate_translation, estimate_scaling, L, K, R_eps, topk,
                             parallel_minima_search, sigmas, keep_ratio, outliers_ratio,
                             density_radius, density_weight, log_params, verbose, pbar, time)
        # self.register_config()
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.max_optim_steps = max_optim_steps
        self.last_joint_fit = last_joint_fit
        self.loss = loss
        self.fast_approx = fast_approx
        self.estimate_translation = estimate_translation
        self.estimate_scaling = estimate_scaling
        self.L = L
        self.K = K
        self.R_eps = R_eps
        self.topk = topk
        self.parallel_minima_search = parallel_minima_search
        self.sigmas = sigmas
        self.keep_ratio = keep_ratio
        self.outliers_ratio = outliers_ratio
        self.density_weight = density_weight
        self.density_radius = density_radius
        self.log_params = log_params
        self.verbose = verbose
        self.pbar = pbar
        self.time = time
        self.base_ae = self.setup_base_ae(ae)
        self.init_timer()

    # ________________________________________ Utilities ________________________________________ #

    # def register_config(self) -> None:
    #     for k, v in asdict(self.config).items():
    #         setattr(self, k, v)

    @property
    def ae_class_name(self) -> str:
        if not hasattr(self, 'ae'):
            return type(self.base_ae).__name__
        if type(self.ae.ae).__name__ == 'OptimizedModule':
            return type(self.ae.ae._orig_mod).__name__
        return type(self.ae.ae).__name__

    @property
    def transforms(self) -> T.Transform3d:
        return self.params.to_transforms()

    @property
    def template(self) -> Tensor:
        r""" Current estimate of the template: $\mathrm{d}(z) \in \mathbb{R}^{K \times 3}$.

        Returns:
            Tensor: Template point cloud (K, 3).
        """
        return self.ae.decode(self.params.z)

    @torch.no_grad()
    def distances_to_orbit_fast_approx(self, z: Tensor) -> Tensor:
        X_hat = self.so3.transforms.transform_points(self.ae.decode(z).detach())  # (L, P, 3)
        if self.loss == 'latent':
            distances = [d_C(repeat(x, len(X_hat)), X_hat) for x in self.X]
            distances = [cast(Tensor, d) for d in distances]
            return torch.stack(distances)  # (L, N)
        Z_hat = self.ae.encode(X_hat)  # (L, D)
        d = lambda z1, z2: torch.linalg.norm(z1 - z2, dim=2, ord=2)
        distances = [d(self.Z[:, None, :], z) for z in Z_hat.split(self.batch_size)]
        return torch.hstack(distances)  # (L, N)

    @torch.no_grad()
    def distances_to_orbit_slow_exact(self, z: Tensor, beta: Tensor, degrade: bool) -> Tensor:
        x_hat = self.ae.decode(z).detach()  # (P, 3)
        distances = list()
        iterable = zip(self.X, beta)
        iterable = tqdm(iterable, total=len(beta)) if self.pbar else iterable
        for x, t in iterable:  # N iterations
            x_hat_t = x_hat + t[None, :]
            X_hat_t = self.so3.transforms.transform_points(x_hat_t)  # (L, P, 3)
            X = repeat(x, len(X_hat_t))
            X_hat_t_crop, X_outliers = self.degrade(X_hat_t, X) if degrade else (X_hat_t, X)
            if self.loss == 'latent':
                Z_hat_crop = self.ae.encode(X_hat_t_crop)
                Z_outliers = self.ae.encode(X_outliers)
                d = torch.linalg.vector_norm(Z_hat_crop - Z_outliers, dim=1)
            else:
                d = d_C(X_hat_t_crop, X_outliers)
                if not isinstance(d, Tensor):
                    d = torch.stack([cast(Tensor, dd) for dd in d])
            distances.append(d)
        return torch.stack(distances)  # (L, N)

    def distances_to_orbit(self, z: Tensor, beta: Tensor, degrade: bool) -> Tensor:
        if self.fast_approx:
            return self.distances_to_orbit_fast_approx(z)
        return self.distances_to_orbit_slow_exact(z, beta, degrade)

    def get_mean_losses(self, multistart: bool = False) -> Tensor:
        """ Get the loss evolution during the optimization, averaged over all views.

        Args:
            multistart (bool, optional):
                If False, skip the loss evolution during parallel multistarts, only keeping joint
                fits evolution. Defaults to False.

        Returns:
            Tensor: Average loss evolution.
        """
        M = self.memories if multistart else filter(lambda m: not m.multistart, self.memories)
        return torch.hstack([cast(Tensor, m.losses).mean(dim=1) for m in M])

    def get_matrix(self, rotation_only: bool = False) -> Tensor:
        """ Get the matrix of estimated rigid motions.

        Args:
            rotation_only (bool, optional):
                If `True`, return only the rotation part of the estimated motions. Defaults to False.

        Returns:
            Tensor: Tensor of estimated rigid motions: (N, 3, 3) if `rotation_only` is `True`, else
                (N, 4, 4).
        """
        M = self.transforms.get_matrix().detach().cpu()
        return M[:, :3, :3] if rotation_only else M

    @torch.no_grad()
    def registrate(self) -> list[Tensor]:
        t_inv = self.transforms.inverse()
        return [t_inv[i].transform_points(self.X[i]) for i in range(len(t_inv))]

    # _____________________________________ 1. Initializers _____________________________________ #

    def init_timer(self) -> None:
        if not self.time:
            return
        for method in ('init_params', 'joint_fit', 'get_restart_angles'):
            exec(f'self.{method} = timeit(verbose=True)(self.{method})')

    def set_device(self, X: Views) -> torch.device:
        device = X[0].device
        if self.verbose:
            print(f'    - Using device: {device.type}.')
        return device

    def setup_base_ae(self, ae: AbstractAE | None) -> PointNetAE | AbstractAE:
        if exists(ae):
            return cast(AbstractAE, ae)
        temp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pae = PointNetAE.build().eval()
        pae.load_state_dict(torch.load(AE_BASELINE_WEIGHTS, map_location=temp_device)['network'])
        return pae

    # @check_ae_interface
    def init_ae(self) -> FrozenAE:
        compiled_ae = compile(self.base_ae.to(self.device))
        compiled_ae = cast(AbstractAE, compiled_ae)
        return FrozenAE(compiled_ae, self.batch_size)

    def init_data(self, X: Views) -> tuple[list[Tensor], Tensor]:
        X = [x.detach() for x in X]
        Z = self.ae.encode(X).detach()
        return X, Z

    def init_so3(self) -> SO3:
        so3_sampling = so3.super_fibonacci_spirals(self.L).to(self.device).detach()
        so3_graph = so3.get_so3_knn_graph(self.L, self.K, self.verbose)
        R = T.euler_angles_to_matrix(so3_sampling, 'ZXZ')
        transforms = T.Rotate(R=R, dtype=so3_sampling.dtype)
        return SO3(so3_sampling, transforms, so3_graph)

    def init_params(self) -> Params:
        # 1. init z: Z's Frechet mean
        pairwise_latent_distances = self.Z[None, :, :] - self.Z[:, None, :]
        frechet_mean = torch.linalg.norm(pairwise_latent_distances, dim=2).sum(dim=0).argmin()
        z = self.Z[frechet_mean].clone()
        # 2.1 init α (N, ): ones
        alpha = torch.ones(len(self.Z), device=self.device)
        # 2.2 init β (N, 3): zeros
        beta = torch.zeros(len(self.Z), 3, device=self.device)
        # 3. init θ (N, 3) (coarse exhaustive search):
        if self.verbose:
            print('    - Running coarse exhaustive search ...')
        distances = self.distances_to_orbit(z, beta, degrade=False)
        theta = self.so3.sampling[distances.argmin(dim=1)].clone()
        params = Params(z, alpha, beta, theta)
        params.requires_grad_()
        return params

    def init_noise(self) -> MultivariateNormal | None:
        if not exists(self.sigmas):
            return None
        if isinstance(self.sigmas, float):
            self.sigmas = 3 * (self.sigmas, )
        loc = torch.zeros(3).to(self.device)
        cov = torch.diag(torch.tensor(self.sigmas).pow(2)).to(self.device)
        return MultivariateNormal(loc, cov)

    def init(self, X: Views) -> None:
        if self.verbose:
            print(':: Initializing model ...')
        self.device: torch.device = self.set_device(X)
        self.ae: FrozenAE = self.init_ae()
        self.X, self.Z = self.init_data(X)
        self.so3: SO3 = self.init_so3()
        self.params: Params = self.init_params()
        self.noise: MultivariateNormal | None = self.init_noise()
        self.memories: list[Memory] = None  # type: ignore

    # _____________________________________ 2. Optimization _____________________________________ #

    def init_optimizer(self, param_names: list[str]) -> tuple[AdamW, ReduceLROnPlateau]:
        params: list[Tensor] = [getattr(self.params, name) for name in param_names]
        optimizer = AdamW(params, lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)
        return optimizer, scheduler

    def degrade(self, X_hat: Tensor, X: Views) -> tuple[Views, Views]:
        # 1. Noise
        if exists(self.noise):
            self.noise = cast(MultivariateNormal, self.noise)
            num_points = len(self.template)
            noise = self.noise.sample(sample_shape=torch.Size((num_points,)))
            X_hat_noisy = X_hat + noise[None, ...]
        else:
            X_hat_noisy = X_hat
        # 2. Crop (x_hat mask)
        if exists(self.keep_ratio):
            self.keep_ratio = cast(float, self.keep_ratio)
            with torch.no_grad():
                distances = pointwise_single_directional_chamfer(X_hat_noisy, X)
                thresholds = [d.quantile(self.keep_ratio) for d in distances]
            X_hat_crop = [x[d < t] for x, d, t in zip(X_hat_noisy, distances, thresholds)]
        else:
            X_hat_crop = X_hat
        # 3. Outliers (X mask)
        if exists(self.outliers_ratio):
            self.outliers_ratio = cast(float, self.outliers_ratio)
            with torch.no_grad():
                distances = pointwise_single_directional_chamfer(X, X_hat_noisy)
                thresholds = [d.quantile(1 - self.outliers_ratio) for d in distances]
            X_outliers = [x[d < t] for x, d, t in zip(X, distances, thresholds)]
        else:
            X_outliers = X
        return X_hat_crop, X_outliers

    def decode_transform_degrade_encode(self) -> tuple[Views, Views, Tensor | None]:
        # 1. Decode
        x_hat = self.ae.decode(self.params.z)            # (D, ) -> (P, 3)
        # 2. Transform
        X_hat = self.transforms.transform_points(x_hat)  # (P, 3) -> (N, P, 3) | αR_θ(X) + β
        # 3. Degrade
        X_hat_crop, X_outliers = self.degrade(X_hat, self.X)
        # 4. Regularize
        densities = None
        if exists(self.density_weight) and self.density_weight > 0:
            pairwise_distances = torch.cdist(x_hat, x_hat)
            densities = (pairwise_distances <= self.density_radius).sum(dim=1)
        if self.loss == 'ambient':
            return X_hat_crop, X_outliers, densities
        # 5. Encode if latent loss
        Z_hat_crop = self.ae.encode(X_hat_crop)
        Z_outliers = self.ae.encode(X_outliers)
        return Z_hat_crop, Z_outliers, densities

    def compute_latent_loss(self, Z_hat: Tensor, Z: Views) -> tuple[Tensor, Tensor]:
        losses = torch.linalg.norm(Z - Z_hat, dim=1, ord=2)
        return losses.mean(), losses.detach().cpu()

    def compute_ambient_loss(self, X_hat: Views, X: Views) -> tuple[Tensor, Tensor]:
        d_CD = lambda x, y: chamfer_distance(x, y, batch_reduction=None)[0]
        losses = [d_CD(x[None, ...], y[None, ...]) for x, y in zip(X_hat, X)]
        losses = [cast(Tensor, l) for l in losses]
        losses = torch.hstack(losses)
        return losses.mean(), losses.detach().cpu()

    def compute_loss(
        self, X_or_Z_hat: Views, X_or_Z: Views, densities: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        loss_fn = self.compute_ambient_loss if self.loss == 'ambient' else self.compute_latent_loss
        loss, losses = loss_fn(X_or_Z_hat, X_or_Z)  # type: ignore
        if exists(densities):
            densities = cast(Tensor, densities)
            loss = loss + self.density_weight * densities.float().std()
        return loss, losses.detach().cpu()

    def log(self, losses: Tensor, last_lr: float, memory: Memory) -> None:
        # type hint
        memory.lr = cast(list, memory.lr)
        memory.losses = cast(list, memory.losses)
        memory.z = cast(list, memory.z)
        # update best loss per object
        losses = losses.float().clone()
        memory.best_losses[losses < memory.best_losses] = losses[losses < memory.best_losses]
        # lr
        memory.lr.append(torch.as_tensor(last_lr))
        # store all in memory
        memory.losses.append(losses)
        memory.z.append(self.params.z.clone().detach().cpu())
        if self.log_params:
            for p in ('z', 'alpha', 'beta', 'theta'):
                getattr(memory, p).append(getattr(self.params, p).clone().detach().cpu())

    def fit_params_subset(self, iterable: tqdm | range, params: list[str]) -> None:
        optimizer, scheduler = self.init_optimizer(params)
        early_stopping = EarlyStopping(self)
        memory = Memory(torch.empty(len(self.Z)).fill_(torch.inf))
        for i in range(self.max_optim_steps):
            optimizer.zero_grad(set_to_none=True)
            loss, losses = self.compute_loss(*self.decode_transform_degrade_encode())
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            last_lr = optimizer.param_groups[0]['lr']
            self.log(losses, last_lr, memory)
            early_stopping.step(loss)
            if isinstance(iterable, tqdm):
                iterable.update()
                iterable.set_postfix(dict(loss=early_stopping.losses[-1], lr=last_lr))
            if early_stopping.converged:
                break
        # update memory
        memory.steps = i + 1
        keys = ['losses', 'lr']
        if self.log_params:
            keys.extend(['z', 'alpha', 'beta', 'theta'])
        for key in keys:
            setattr(memory, key, torch.stack(getattr(memory, key)))
        self.memories = [memory] if not exists(self.memories) else self.memories + [memory]

    def get_fit_params(self, z: bool) -> list[str]:
        params = ['theta']
        if self.estimate_translation:
            params.append('beta')
        if self.estimate_scaling:
            params.append('alpha')
        if z:
            params.append('z')
        return params

    def joint_fit(self, z: bool = True) -> None:
        if self.verbose:
            fit_type = 'joint' if z else 'multistart'
            print(f':: Running {len(self.Z)} parallel {fit_type} optimizations ...')
        iterable = range(self.max_optim_steps)
        iterable = tqdm(iterable) if self.pbar else iterable
        params = self.get_fit_params(z)
        self.fit_params_subset(iterable, params)

    # ______________________________________ 3. Multistart ______________________________________ #

    @torch.no_grad()
    def get_restart_angles(self) -> Tensor:
        if self.verbose:
            print(':: Computing restart angles ...')
        distances = self.distances_to_orbit(self.params.z, self.params.beta, degrade=True)
        local_minima = so3.find_so3_local_minima_from_knn_graph(self.so3.graph, distances,
                                                                self.parallel_minima_search)
        all_restart_angles = list()
        for l in local_minima:
            restart_angles = self.so3.sampling[l[:self.topk]]
            if len(restart_angles) < self.topk:  # repeat if not enough minima to stack
                n = self.topk - len(restart_angles)
                restart_angles = torch.vstack((repeat(restart_angles[0], n), restart_angles))
            all_restart_angles.append(restart_angles)
        return torch.stack(all_restart_angles)

    @torch.no_grad()
    def get_multistart_improvements(self, K: int, N: int) -> tuple[Params, Tensor]:
        alpha, beta, theta = map(lambda x: unflatten(x, (K, N)), self.params.sim3)
        restart_losses = self.memories[-1].best_losses.clone()
        restart_losses = unflatten(restart_losses, (K, N))
        best_minima = restart_losses.argmin(dim=0)
        restart_losses = restart_losses[best_minima, torch.arange(N)]
        # 1. Loss improvements
        loss_improvements = (restart_losses <= self.__best_losses).argwhere().squeeze(dim=1)
        get_best_param = lambda p: p[best_minima[loss_improvements], loss_improvements]
        new_alpha, new_beta, new_theta = map(get_best_param, (alpha, beta, theta))
        # 2. Far enough from previous poses
        old_theta = self.__cached_params.theta[loss_improvements]
        R_old = T.euler_angles_to_matrix(old_theta, 'ZXZ')
        R_new = T.euler_angles_to_matrix(new_theta, 'ZXZ')
        so3_distances = so3.so3_relative_angle(R_old, R_new)
        so3_improvements = (so3_distances >= self.R_eps).argwhere().squeeze(dim=1)
        new_alpha, new_beta, new_theta = map(lambda p: p[so3_improvements],
                                             (new_alpha, new_beta, new_theta))
        improvements = [loss_improvements[i] for i in so3_improvements]
        improvements = torch.stack(improvements) if len(improvements) > 0 else torch.empty(0)
        new_params = Params(self.params.z, new_alpha, new_beta, new_theta)
        return new_params, improvements

    def multistart(self) -> Tensor:
        # 1. Compute restart angles (SO(3) local minima)
        restart_angles = self.get_restart_angles()
        # 2. Cache current state
        self.__best_losses = self.memories[-1].best_losses
        self.__cached_params = self.params.clone().detach()
        self.__cached_Z = self.Z.clone()
        self.__cached_X = [x.clone() for x in self.X]
        # 3. Prepare params for parallel restarts
        N = len(self.Z)
        K = self.topk
        alpha = repeat(self.__cached_params.alpha, K)
        beta = repeat(self.__cached_params.beta, K)
        theta = restart_angles.transpose(0, 1)
        Z = repeat(self.__cached_Z.clone(), K)
        X = K * self.X
        # θ: (K * N, 3), α: (K * N), β: (K * N, 3), Z: (K * N, D)
        alpha, beta, theta, Z = map(flatten, (alpha, beta, theta, Z))
        # 4. Run parallel restart
        self.params = Params(self.params.z, alpha, beta, theta).clone().detach().requires_grad_()
        self.Z = Z
        self.X = X
        self.joint_fit(z=False)
        # 5. Find improved objects
        new_params, improvements = self.get_multistart_improvements(K, N)
        # 6. Update memory
        self.memories[-1].multistart = True
        self.memories[-1].improvements = improvements.cpu().clone()
        # 7. Update cached params with improvements
        if len(improvements) > 0:
            self.__cached_params.index_update_(new_params, improvements)
        # 8. Reset to normal state
        self.params = self.__cached_params.clone().detach().requires_grad_()
        self.Z = self.__cached_Z.clone()
        self.X = [x.clone() for x in self.__cached_X]
        if self.verbose and len(improvements) > 0:
            print(f'Multistart improved {len(improvements)} objects:')
            print(improvements)
        # 9. Optionally Update learning rate
        if len(self.memories) > 2 and torch.equal(improvements, self.memories[-3].improvements):
            self.lr /= 10
            if self.verbose:
                print('Multistart improved same objects twice in a row.')
                print(f'Reducing lr to {self.lr}.')
        return improvements

    # ____________________________________ 3. Main functions ____________________________________ #

    def fit(self, X: Views, max_iter: int = 30) -> None:
        """ Optimize template and N rigid motions from N views.

        !!! Note
            `Views` is a type alias defined as
            ```python
            Views: TypeAlias = "Tensor | Sequence[Tensor]"
            ```

        Args:
            X (Views): Point clouds to register. Tensor (N, K, 3) or list of N Tensors (Ki, 3).
            max_iter (int, optional): Maximal number of optimization loop. Defaults to 30.
        """
        X = center_max_norm(X)
        self.init(X)
        for i in range(max_iter):
            if self.verbose:
                print(f'LOOP {i+1}')
            self.joint_fit()
            improvements = self.multistart()
            if len(improvements) == 0:
                if self.verbose:
                    print('Multistart converged.')
                break
            if self.verbose:
                print(80 * '-')
        else:
            print('Max iterations reached. Multistart did not converged.')
        if self.last_joint_fit:
            self.joint_fit()

    def fit_transform(self, X: Views, max_iter: int = 30) -> Views:
        """ Run the fit method, apply the estimated rigid motions to the views and return the
        registered views.

        !!! Note
            `Views` is a type alias defined as
            ```python
            Views: TypeAlias = "Tensor | Sequence[Tensor]"
            ```

        Args:
            X (Views): Point clouds to register. Tensor (N, K, 3) or list of N Tensors (Ki, 3).
            max_iter (int, optional): Maximal number of optimization loop. Defaults to 30.

        Returns:
            Views: list of N Tensors (Ki, 3) representing registered views.
        """
        self.fit(X, max_iter)
        with torch.no_grad():
            return self.registrate()

    # ______________________________________ 4. Save & Load _____________________________________ #

    @property
    def state(self) -> dict:
        state = dict(config=asdict(self.config), ae_class_name=self.ae_class_name)
        if not hasattr(self, 'X'):  # has not been initialized
            return state
        state = cast(dict, state)
        state['X'] = self.X
        state['Z'] = self.Z
        state['memories'] = [asdict(m) for m in self.memories]
        state['params'] = asdict(self.params)
        return state

    @staticmethod
    def load_state(path: str | Path) -> dict:
        path = Path(path)
        parent = path.parent.resolve()
        name = Path(path.stem).with_suffix('.pt')  # works if name has extension or not
        state = torch.load(parent / name)
        return state

    def save_state(self, name: str, output_dir: str = '.') -> None:
        output_dir_path = Path(output_dir).resolve()
        output_dir_path.mkdir(exist_ok=True)
        name_path = Path(Path(name).stem).with_suffix('.pt')  # works if name has extension or not
        torch.save(self.state, output_dir_path / name_path)

    @classmethod
    def from_state(cls, *, path=None, state=None, **kwargs) -> POLAR:
        if sum(map(exists, (path, state))) != 1:
            raise ValueError('Exactly one of path and state must be specified.')
        if path is not None:
            state = POLAR.load_state(path)
        state = cast(dict, state)
        for k, v in kwargs.items():
            state['config'][k] = v
        model = cls(**state['config'])
        if 'X' not in state:  # has not been initialized
            return model
        if model.verbose:
            print(':: Initializing model ...')
        # has been initialized
        model.device = model.set_device(state['X'])
        model.X = state['X']
        model.Z = state['Z']
        model.params = Params(**state['params'])
        model.memories = [Memory(**m) for m in state['memories']]
        model.so3 = model.init_so3()
        model.noise = model.init_noise()
        model.ae = model.init_ae()  # assumes default AE
        return model
