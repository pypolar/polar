from __future__ import annotations
import torch
from torch import Tensor
from .core import Transform
from . import functionals as F


# +---------------------------------------------------------------------------------------------+ #
# |                                          I - BASE                                           | #
# +---------------------------------------------------------------------------------------------+ #

class Center(Transform):

    """ See [polar.train.data.transforms.functionals.center][]. """

    def __init__(self) -> None:
        super().__init__()

    def get_params(self, **data: Tensor) -> dict:
        return dict()

    def apply(self, pointclouds: Tensor) -> Tensor:
        return F.center(pointclouds)


class Normalize(Transform):

    """ See [polar.train.data.transforms.functionals.normalize][]. """

    def __init__(self) -> None:
        super().__init__()

    def get_params(self, **data: Tensor) -> dict:
        return dict()

    def apply(self, pointclouds: Tensor) -> Tensor:
        return F.normalize(pointclouds)


class RandomSample(Transform):

    """ Sample a unique set of points in each batch element. 
    
    See [polar.train.data.transforms.functionals.sample][]. """

    def __init__(self, num_points: int) -> None:
        """_summary_

        Args:
            num_points (int): Number of points to select per point clouds.
        """
        super().__init__()
        self.num_points = num_points

    def get_params(self, **data: Tensor) -> dict:
        batch_size = Transform.get_batch_size(**data)
        num_points = Transform.get_num_points(**data)
        indices = torch.randint(num_points, size=(batch_size, self.num_points))
        return dict(indices=indices)

    def apply(self, pointclouds: Tensor, indices: Tensor) -> Tensor:
        return F.sample(pointclouds, indices)


# +---------------------------------------------------------------------------------------------+ #
# |                                         II - SIM(3)                                         | #
# +---------------------------------------------------------------------------------------------+ #

class RandomTranslate(Transform):

    r""" Translate each point cloud in a batch by a random unique value in
        $[-\text{max_t}, \text{max_t}]$.        
        See [polar.train.data.transforms.functionals.translate][].
    """

    def __init__(self, max_t: float = 0., p: float = 1) -> None:
        """_summary_

        Args:
            max_t (float, optional):
                Maximal norm of the randomly generated translations. Defaults to `0`.
            p (float, optional): Probability to apply the random translations. Defaults to `1`.
        """
        super().__init__(p)
        self.max_t = max_t

    def get_params(self, **data: Tensor) -> dict:
        batch_size = Transform.get_batch_size(**data)
        t = torch.rand(size=(batch_size, 3))
        t = 2 * self.max_t * t - self.max_t
        return dict(t=t)

    def apply(self, pointclouds: Tensor, t: Tensor) -> Tensor:
        return F.translate(pointclouds, t)


class RandomRotate(Transform):

    """ Rotate each point cloud in a batch by a random unique rotation.         
        See [polar.train.data.transforms.functionals.rotate][].
    """

    def __init__(self, min_angle: float = 0., max_angle: float = 180., p: float = 1) -> None:
        """_summary_

        !!! Warning
            Angles are in degrees.

        Args:
            min_angle (float, optional):
                Minimal relative angle between the identity and the generated rotations. Defaults to `0`.
            max_angle (float, optional):
                Maximal relative angle between the identity and the generated rotations. Defaults to `180`.
            p (float, optional): Probability to apply the rotations. Defaults to `1`.
        """
        super().__init__(p)
        self.min_angle = min_angle * torch.pi / 180
        self.max_angle = max_angle * torch.pi / 180

    @staticmethod
    def random_rotation(n: int, max_angle: float, min_angle: float = 0) -> Tensor:
        r""" See [Rodrigues Rotation Formula](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula),
            section Matrix notation.
        The rotation through an angle $\theta$ counterclockwise about the axis $k = (k_x, k_y, k_z)$ is given by
        $K = 
        \begin{pmatrix}
        0 &  -k_z &  k_y \\
        k_z & 0 & -k_x \\
        - k_y & k_x & 0\end{pmatrix}$
        then
        $R = I + \sin(\theta)K + (1 - \cos(\theta))K²$.
        $R$ is an element of the Lie groupe $SO(3)$ and $K$ is an element of the Lie algebra $\mathfrak{so}(3)$
        generating that group. Note that $R = \exp(\theta K)$.
        """
        axis = torch.randn(n, 3)
        axis /= torch.linalg.vector_norm(axis, dim=1, keepdim=True)
        angle = min_angle + torch.rand(n).squeeze() * (max_angle - min_angle)
        zero = torch.zeros(n)
        K = [[zero, -axis[:, 2], axis[:, 1]], [axis[:, 2], zero, -axis[:, 0]], [-axis[:, 1], axis[:, 0], zero]]
        K = torch.stack([torch.stack(k) for k in K]).permute(2, 0, 1)
        sin = torch.sin(angle)[:, None, None]
        cos = torch.cos(angle)[:, None, None]
        I = torch.eye(3).expand_as(K)
        R = I + sin * K + (1 - cos) * K.bmm(K)
        return R

    def get_params(self, **data: Tensor) -> dict:
        batch_size = Transform.get_batch_size(**data)
        R = RandomRotate.random_rotation(batch_size, self.max_angle, self.min_angle)
        return dict(R=R)

    def apply(self, pointclouds: Tensor, R: Tensor) -> Tensor:
        return F.rotate(pointclouds, R)


class RandomRigidMotion(Transform):

    """ Rotate & Translate each point cloud in a batch by a random unique rotation.
        Min/Max angles in degrees.

        See [polar.train.data.transforms.functionals.apply_rigid_motion][].
    """

    def __init__(
        self, max_t: float = 0., min_angle: float = 0., max_angle: float = 180., p: float = 1
    ) -> None:
        super().__init__(p)
        self.random_rotate = RandomRotate(min_angle, max_angle)
        self.random_translate = RandomTranslate(max_t)

    def get_params(self, **data: Tensor) -> dict:
        R = self.random_rotate.get_params(**data)['R']
        t = self.random_translate.get_params(**data)['t']
        return dict(R=R, t=t)

    def apply(self, pointclouds: Tensor, R: Tensor, t) -> Tensor:
        return F.apply_rigid_motion(pointclouds, R, t)


class RandomScale(Transform):

    """ Randomly scale a batch of point clouds, with a unique scale factor for each point cloud. 
    
        See [polar.train.data.transforms.functionals.scale][].
    """

    def __init__(self, min_scale: float = 0., max_scale: float = 1, p: float = 1) -> None:
        super().__init__(p)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def get_params(self, **data: Tensor) -> dict:
        factors = torch.rand(Transform.get_batch_size(**data))
        factors = (self.max_scale - self.min_scale) * factors + self.min_scale
        return dict(factors=factors)

    def apply(self, pointclouds: Tensor, factors: Tensor) -> Tensor:
        return F.scale(pointclouds, factors)


# +---------------------------------------------------------------------------------------------+ #
# |                                     III - DEGRADATIONS                                      | #
# +---------------------------------------------------------------------------------------------+ #


class RandomJit(Transform):

    r""" Add gaussian noise with unique random standard deviation per point cloud in a batch.
        For each batch element, the standard deviation $\sigma$ is such that
        $\sigma \sim \mathcal{U}(0, \text{sigma_max})$.

    See [polar.train.data.transforms.functionals.jit][].
    """

    def __init__(self, sigma_max: float, p: float = 1) -> None:
        super().__init__(p)
        self.sigma_max = sigma_max

    def get_params(self, **data: Tensor) -> dict:
        batch_size = Transform.get_batch_size(**data)
        sigmas = self.sigma_max * torch.rand(batch_size)
        return dict(sigmas=sigmas)

    def apply(self, pointclouds: Tensor, sigmas: Tensor) -> Tensor:
        """ Call [polar.train.data.transforms.functionals.jit][] on a batch of point clouds, with
            params obtained by `self.get_params(...)`.
        """
        return F.jit(pointclouds, sigmas)


class RandomPlaneCut(Transform):

    """ Randomly generate a unique plane for each batch element and cut through it, such that a proportion
        `keep_ratio` is kept for each point cloud.

        See [polar.train.data.transforms.functionals.plane_cut][].
    """

    def __init__(self, keep_ratio: float = 0.7, p: float = 1) -> None:
        """_summary_

        Args:
            keep_ratio (float, optional): Proportion of each point cloud to keep. Defaults to 0.7.
            p (float, optional): Probablity to crop a given batch. Defaults to 1.
        """
        super().__init__(p)
        self.keep_ratio = keep_ratio

    @staticmethod
    def uniform_sphere(num: int) -> Tensor:
        """ Uniform sampling on a 3-sphere ([Source](https://gist.github.com/andrewbolster/10274979)).

        Args:
            num (int, optional): Number of vectors to sample (or `None` if single). Defaults to `None`.

        Returns:
            Tensor: Random vector of size `(num, 3)` with unit norm. If `num` is `None`, returned value will have
                size `(3,)`.
        """
        phi = torch.distributions.Uniform(0, 2 * torch.pi).rsample(torch.Size((num,)))
        cos_theta = torch.distributions.Uniform(-1.0, 1.0).rsample(torch.Size((num,)))
        theta = torch.arccos(cos_theta)
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        return torch.stack((x, y, z), dim=-1)

    def get_params(self, **data: Tensor) -> dict:
        """ Returns a batch of directions sampled in S3, of shape `(batch_size, 3)` from a
            dictionary whose values are batches of point clouds of shape
            `(batch_size, num_points, spatial_dim)`.

        Returns:
            Dictionary with generated planes and keep ratio value.
        """
        batch_size = Transform.get_batch_size(**data)
        planes = RandomPlaneCut.uniform_sphere(batch_size)
        return dict(planes=planes, keep_ratio=self.keep_ratio)

    def apply(self, pointclouds: Tensor, planes: Tensor, keep_ratio: float) -> Tensor:
        """ Call [polar.train.data.transforms.functionals.plane_cut][] on a batch of point clouds, with params
        obtained by `self.get_params(...)`.
        """
        augmented_pointclouds, mask = F.plane_cut(pointclouds, planes, keep_ratio, return_mask=True)
        self.last_params['mask'] = mask
        return augmented_pointclouds
