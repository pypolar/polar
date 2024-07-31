from __future__ import annotations
from typing import Sequence
import torch
from torch import Tensor


class Transform:
    """ Base Transform class. Every Transforms inherits this class and implements `get_params()`
        and `apply()`; `apply()` is always based on the functional counterpart of the Transform
        class. Any Transform accepts multiple batches of point clouds (typically sources and
        targets) as it is often desired to apply the same random transform to many batches of
        points clouds. If multiple point clouds are passed, they MUST all have the same length.
        Any Transform is applied with a provided probability `self.p`.
    """

    def __init__(self, p: float = 1.0) -> None:
        self.p = p

    @staticmethod
    def get_batch_size(**data: Tensor) -> int:
        """ Static method. Assume every values in data dict is of same shape. """
        return list(data.values())[0].shape[0]

    @staticmethod
    def get_num_points(**data: Tensor) -> int:
        """ Static method. Assume every values in data dict is of same shape. """
        return list(data.values())[0].shape[1]

    def register_params(self, params: dict) -> None:
        self.last_params = params

    def get_params(self, **data: Tensor) -> dict:
        """ Shared parameters for one apply (usually random values).

        Args:
            **data (Tensor): Dictionary with str as keys and batch of point clouds of shape
                             `(batch_size, num_points, *)` where `*` denotes spatial coordinates as
                             values. Typically, `data = {'source': ..., 'target': ...}`.

        Returns:
            params (dict): Params used by the transform (e.g. Euler angles for rotation).
        """
        raise NotImplementedError

    def apply(self, pointclouds: Tensor, **params) -> Tensor:
        """ Apply the functional transform with the params obtained by `self.get_params()` to
            one batch of point clouds.

        Args:
            pointclouds (Tensor): Batch of point clouds of shape `(batch_size, num_points, *)`
                                  where `*` denotes spatial coordinates.
        Returns:
            Transformed tensor: Transformed batch of point clouds of shape `(batch_size,
                                num_points, *)` where `*` denotes spatial coordinates.
        """
        raise NotImplementedError

    def __call__(self, **data: Tensor) -> dict[str, Tensor]:
        """ Call `self.apply` with a probability `self.p` on every values in the provided
            dictionary.

        Returns:
            Transformed data: Same dictionary structure as input. The values have been
                              transformed (with a certain probability).
        """
        if torch.rand(size=(1, )) < self.p:
            params = self.get_params(**data)
            self.register_params(params)
            for k, v in data.items():
                data[k] = self.apply(v, **params)
        return data


class Compose(Transform):

    """ Very simple mechanism to chain Transforms.
        Nothing more than a wrapper able to store a sequence of Transforms, to be applied
        iteratively on every values in a provided dictionary.
        It also has a probability, typically used so that only a portion of a dataset is augmented
        during training.

        !!! Example
            ```python
            from polar.train.data import transforms as T
            center_normalize = T.Compose((T.Center(), T.Normalize()))
            ```
    """

    def __init__(self, transforms: Sequence[Transform], p: float = 1.0) -> None:
        """_summary_

        Args:
            transforms (Sequence[Transform]): Transformations to be randomly composed during a
                                              call.
            p (float, optional): Probability to apply the provided sequence. Defaults to 1.0.
        """
        super(Compose, self).__init__()
        self.transforms = {t.__class__.__name__: t for t in transforms}
        self.p = p

    def __call__(self, **data: Tensor) -> dict[str, Tensor]:
        if torch.rand(size=(1, )) < self.p:
            for transform in self.transforms.values():
                data = transform(**data)
        return data
