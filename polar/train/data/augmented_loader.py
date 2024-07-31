from __future__ import annotations
from typing import Generator
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from .modelnet import ModelNet
from . import transforms as T


class AugmentedDataLoader:

    """ Apply same motion to sources and targets. Degrade sources only. Intented to be used
        to train an autoencoder to reconstruct and restore point clouds.
    """

    def __init__(
        self, dataloader: DataLoader[ModelNet], num_points: int = 1024, shuffle: bool = False,
        sigma: float = 0., min_scale: float = 1., max_angle: float = 180., max_trans: float = 0.,
        keep_ratio: float = 1., p: float = 1, handle_device: bool = True
    ) -> None:
        r"""_summary_

        Args:
            dataloader (DataLoader[ModelNet]):
                A ModelNet dataloader instance, typically from [polar.train.data.factory.get_modelnet_dataloader][].
            num_points (int, optional): Number of points in each cloud. Defaults to 1024.
            shuffle (bool, optional):
                Shuffle the dense point clouds (5000 points before sampling `num_points`) points. If `True`, sources
                and targets will be two unique sampling of the same underlying surface. Defaults to False.
            sigma (float, optional): Isotropic noise standard deviation. Defaults to `0`.
            min_scale (float, optional):
                If $< 1$, will randomly scale each batch with a factor $s \sim \mathcal{U}(\text{min_scale}, 1)$.
                Defaults to `1`.
            max_angle (float, optional):
                For each point cloud, randomly sample a rotation whose relative angle with the identity is in
                $[0, \text{max_angle}]$. Defaults to `180`.
            max_trans (float, optional):
                For each point cloud, randomly sample a translation whose norm is int $[0, \text{max_trans}]$.
                Defaults to `0`.
            keep_ratio (float, optional):
                If $< 1$, will randomly crop each batch with a factor $k \sim \mathcal{U}(\text{keep_ratio}, 1)$.
                Defaults to `1`.
            p (float, optional): Probability to apply the augmentation. Defaults to `1`.
            handle_device (bool, optional):
                If `True`, will guess the device and move point clouds to it. Defaults to `True`.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.handle_device = handle_device
        self.dataloader = dataloader
        self.num_points = num_points
        self.shuffle = shuffle
        self.sigma = sigma
        self.min_scale = min_scale
        self.max_angle = max_angle
        self.max_trans = max_trans
        self.keep_ratio = keep_ratio
        self.init_transforms(p)

    def init_transforms(self, p: float) -> None:
        self.sample = T.RandomSample(self.num_points)
        self.center_normalize = T.Compose((T.Center(), T.Normalize()))
        if self.max_trans > 0:
            self.rigid_motion = T.RandomRigidMotion(max_t=self.max_trans, max_angle=self.max_angle)
        else:
            self.rigid_motion = T.RandomRotate(max_angle=self.max_angle)
        self.degradations = None
        degradations = list()
        if self.sigma > 0:
            degradations.append(T.RandomJit(self.sigma, p=p))
        if self.keep_ratio < 1:
            degradations.append(T.RandomPlaneCut(self.keep_ratio, p=p))
        if len(degradations) > 0:
            self.degradations = T.Compose(degradations)
        self.scale = None
        if self.min_scale < 1:
            self.scale = T.RandomScale(min_scale=self.min_scale, max_scale=1, p=p)

    def get_groundtruth_transform(self) -> Tensor:
        R_s2t_gt = self.rigid_motion.last_params['R']
        if self.max_trans > 0:
            t_s2t_gt = self.rigid_motion.last_params['t']
        else:
            t_s2t_gt = torch.zeros(len(R_s2t_gt), 3, device=R_s2t_gt.device)
        T_s2t_gt = torch.eye(4).repeat(len(R_s2t_gt), 1, 1)
        T_s2t_gt[:, :3, :3] = R_s2t_gt
        T_s2t_gt[:, :3, 3] = t_s2t_gt
        return T_s2t_gt

    def process_batch(self, sources: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        device = self.device if self.handle_device else sources.device
        sources = sources.to(device)
        targets = sources.clone()
        if self.shuffle:
            sources = self.sample(data=sources)['data']
            targets = self.sample(data=targets)['data']
            data = dict(sources=sources, targets=targets)
        else:
            data = self.sample(sources=sources, targets=targets)
        data = self.center_normalize(**data)
        data = self.rigid_motion(**data)
        T_s2t_gt = self.get_groundtruth_transform().to(device)
        if self.degradations is not None:
            data['sources'] = self.degradations(sources=data['sources'])['sources']
        data = self.center_normalize(**data)
        if self.scale is not None:
            data = self.scale(**data)
        return data['sources'], data['targets'], T_s2t_gt

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self) -> Generator[tuple[Tensor, Tensor, Tensor]]:
        for batch in self.dataloader:
            yield self.process_batch(batch)
