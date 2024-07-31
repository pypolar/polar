from __future__ import annotations
import torch
from torch import Tensor


# +---------------------------------------------------------------------------------------------+ #
# |                                          I - BASE                                           | #
# +---------------------------------------------------------------------------------------------+ #

def center(pointclouds: Tensor) -> Tensor:
    """ Center each element in a batch of point clouds (substract the mean over the second dim).

    Args:
        pointclouds (Tensor): Batch of point clouds `(batch_size, num_points, *)`.

    Returns:
        Batch of centered point clouds.
    """
    return pointclouds - pointclouds.mean(dim=1, keepdim=True)


def normalize(pointclouds: Tensor) -> Tensor:
    """ Scale each element in a batch of point clouds so that it lies exactly within the unit sphere.

    Args:
        pointclouds (Tensor): Batch of point clouds `(batch_size, num_points, *)`.

    Returns:
        Batch of normalized point clouds.
    """
    pointclouds_ = center(pointclouds)
    max_norm = pointclouds_.norm(dim=2).amax(dim=1)
    return scale(pointclouds_, 1 / max_norm)


def max_norm():
    pass


def sample(pointclouds: Tensor, indices: Tensor) -> Tensor:
    """ Sample the elements of the batch using the provided index tensor.

    Args:
        pointclouds (Tensor): Batch of point clouds `(batch_size, num_points, *)`.
        indices (Tensor): Tensor of indices to keep `(batch_size, num_sampled_points)`.

    Returns:
        Batch of sampled point clouds `(batch_size, num_sampled_points, *)`.
    """
    return pointclouds.gather(1, indices.repeat(3, 1, 1).permute(1, 2, 0).to(pointclouds.device))


def pairwise_max_norm(pointclouds1: Tensor, pointclouds2: Tensor) -> tuple[Tensor, Tensor]:
    """ Scale each pair of elements of the two batches by their maximal norm.

    !!! Warning
        `pointclouds1` and `pointclouds2` MUST have the same length.

    Args:
        pointclouds1 (Tensor): Batch of point clouds `(batch_size, n, *)`.
        pointclouds2 (Tensor): Batch of point clouds `(batch_size, m, *)`.

    Returns:
        tuple[Tensor, Tensor]: The two batches of normalized point clouds.
    """
    assert len(pointclouds1) == len(pointclouds2), 'Inputs must be of same length.'
    norms1 = center(pointclouds1).norm(dim=2).amax(dim=1)  # B
    norms2 = center(pointclouds2).norm(dim=2).amax(dim=1)  # B
    max_norms = torch.stack((norms1, norms2)).amax(dim=0)  # (2, B) -> B
    return scale(pointclouds1, 1 / max_norms), scale(pointclouds2, 1 / max_norms)


# +---------------------------------------------------------------------------------------------+ #
# |                                         II - SIM(3)                                         | #
# +---------------------------------------------------------------------------------------------+ #

def translate(pointclouds: Tensor, t: Tensor) -> Tensor:
    """ Apply a unique translation to each element of the batch of point clouds.

    Args:
        pointclouds (Tensor): Batch of point clouds `(batch_size, num_points, *)`.
        t (Tensor): Batch of translation vectors `(batch_size, *)`.

    Returns:
        Batch of translated point clouds `(batch_size, num_points, *)`.
    """
    return pointclouds + t[:, None, :].to(pointclouds.device)


def rotate(pointclouds: Tensor, R: Tensor) -> Tensor:
    """ Apply a unique rotation to each element of the batch of point clouds.

    Args:
        pointclouds (Tensor): Batch of point clouds `(batch_size, num_points, 3)`.
        R (Tensor): Batch of rotation matrices `(batch_size, 3, 3)`.

    Returns:
        Batch of rotated point clouds `(batch_size, num_points, 3)`.
    """
    return pointclouds.bmm(R.mT.to(pointclouds.device))


def apply_rigid_motion(pointclouds: Tensor, R_or_T: Tensor, t: Tensor | None = None) -> Tensor:
    """ Apply a unique rigid motion, *i.e.* the composition of a rotation and a translation to
        each point cloud in the batch.

    Args:
        pointclouds (Tensor): Batch of point clouds `(batch_size, num_points, *)`.
        R_or_T (Tensor): If `t` is None, this must be a rigid motion matrix `(batch_size, 4, 4)`.
        t (Tensor | None, optional): Batch of translation vectors `(batch_size, 3)`. Defaults to `None`.

    Raises:
        ValueError: If `t` is `None` and `R_or_T` is not of shape `(batch_size, 4, 4)`.

    Returns:
        Tensor: Batch of rigidly moved point clouds `(batch_size, num_points, 3)`.
    """
    if t is not None:
        return translate(rotate(pointclouds, R_or_T), t)
    if not R_or_T.shape[1:] == (4, 4):
        raise ValueError('when translation is not given, a batch of (4, 4) motions but be given.')
    R, t = R_or_T[:, :3, :3], R_or_T[:, :3, 3]
    return translate(rotate(pointclouds, R), t)


def scale(pointclouds: Tensor, values: Tensor) -> Tensor:
    """ Apply a unique scaling factor to each point cloud in the provided batch.

    Args:
        pointclouds (Tensor): Batch of point clouds `(batch_size, num_points, *)`.
        values (Tensor): Batch of scalar scaling values `(batch_size,)`.

    Returns:
        Batch of scaled point clouds `(batch_size, num_points, *)`.
    """
    return pointclouds * values[:, None, None].to(pointclouds.device)


# +---------------------------------------------------------------------------------------------+ #
# |                                     III - DEGRADATIONS                                      | #
# +---------------------------------------------------------------------------------------------+ #

def jit(pointclouds: Tensor, sigmas: Tensor) -> Tensor:
    """ Add white gaussian noise with variance specified per batch element.

    Args:
        pointclouds (Tensor): Batch of point clouds `(batch_size, num_points, *)`.
        sigmas (Tensor): Noise variance per batch element.

    Returns:
        Tensor: Noisy batch of point clouds X = X + eps, eps ~ N(0, sigma)
    """
    gaussian_noise = sigmas[:, None, None].to(pointclouds.device) * torch.randn_like(pointclouds)
    return pointclouds + gaussian_noise


def plane_cut(
    pointclouds: Tensor, planes: Tensor, keep_ratio: float, return_mask: bool
) -> Tensor | tuple[Tensor, Tensor]:
    r""" Being given a direction in $\mathcal{S}_3$, retain points which lie within the half-space oriented in
    this direction, such that `keep_ratio * num_points` are retained.

    Args:
        pointclouds (Tensor): Batch of point clouds of shape `(batch_size, num_points, 3)`.
        planes (Tensor): Batch of direction in S3, of shape `(batch_size, 3)`.
        keep_ratio (float): Ratio of points to retain. Outputs will be shaped `(batch_size, n, 3)`,
            where `n = keep_ratio * num_points`.
        return_mask (bool): Whether to return the cropping mask alongside the cutted batch.

    Returns:
        Tensor: Batch of cutted pointclouds, of shape `(batch_size, keep_ratio * num_points, 3)`.
    """
    pointclouds = center(pointclouds)
    distances_from_planes = (pointclouds @ planes[:, :, None].to(pointclouds.device)).squeeze()
    d_threshold = torch.quantile(distances_from_planes, 1.0 - keep_ratio, dim=1)
    mask = distances_from_planes > d_threshold[:, None]
    # Unfortunately, mask does not contain exactly the same number of points per batch element.
    # I'm forced to do a dirty trick to add some 1 to each line of the mask.
    # Maybe there's a clever way to do this but I couldn't find it.
    num_points_to_keep = mask.sum(dim=1).max()
    zeros = torch.argwhere(~mask)
    for i in range(len(mask)):
        current_num_points = mask[i].sum()
        num_points_to_add = num_points_to_keep - current_num_points
        mask[zeros[zeros[:, 0] == i][:num_points_to_add].T.unbind()] = True
    cutted = torch.masked_select(pointclouds, mask[:, :, None]).reshape(len(pointclouds), -1, 3)
    return (cutted, mask) if return_mask else cutted
