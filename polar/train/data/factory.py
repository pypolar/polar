from __future__ import annotations
from typing import Sequence, Any
from argparse import Namespace
from torch.utils.data import DataLoader
from .modelnet import ModelNet
from .augmented_loader import AugmentedDataLoader


def args_to_param_subset(args: Namespace, keys: tuple[str, ...]) -> dict[str, Any]:
    args_dict = vars(args)
    params = {k: args_dict[k] for k in keys if k in args}
    return params


# _________________________________________________ 1. Base Loaders _________________________________________________ #

def get_modelnet_dataloader(
    batch_size: int, num_workers: int,
    rootdir: str = 'modelnet', split: str = 'train', classes: Sequence[str] | None = None,
    exclude_classes: Sequence[str] | None = None, samples_per_class: int | None = None, return_labels: bool = False
) -> DataLoader[ModelNet]:
    """ Instanciate a basic ModelNet Pytorch dataloader. Each batch is composed of fixed length point clouds
        `(batch_size, num_points, 3)`.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Parallel loading with `num_workers` processes.
        rootdir (str, optional): Path to the directory containing the `.h5` files. Defaults to 'modelnet'.
        split (str, optional): 'train' or 'test. Defaults to 'train'.
        classes (Sequence[str] | None, optional):
            Shape categories to use. If `None`, load all categories. See `ModelNet.all_classes`. Defaults to `None`.
        exclude_classes (Sequence[str] | None, optional):
            Shape categories to exclude from the dataset. Defaults to `None`.
        samples_per_class (int | None, optional): Number of point clouds per category to load. Defaults to `None`.
        return_labels (bool, optional):
            If `True`, return the class index alongside the point cloud. Defaults to `False`.

    Returns:
        Standard Pytorch DataLoader.
    """
    dataset = ModelNet(rootdir, split, classes, exclude_classes, samples_per_class, return_labels)
    loader_params = dict(num_workers=num_workers, pin_memory=True, shuffle=True, drop_last=False)
    return DataLoader(dataset, **loader_params, batch_size=batch_size)  # type: ignore


def get_modelnet_dataloaders(
    batch_size: int, num_workers: int,
    rootdir: str = 'modelnet', classes: Sequence[str] | None = None,
    exclude_classes: Sequence[str] | None = None, samples_per_class: int | None = None, return_labels: bool = False
) -> tuple[DataLoader[ModelNet], DataLoader[ModelNet]]:
    """ Same as [polar.train.data.factory.get_modelnet_dataloader][], but returns a tuple of train and test dataloaders.

    Returns:
        Train loader, Test loader.
    """
    split = 'train'
    train_loader = get_modelnet_dataloader(**locals())
    split = 'test'
    locals_ = {k: v for k, v in locals().items() if k != 'train_loader'}
    test_loader = get_modelnet_dataloader(**locals_)
    return train_loader, test_loader


def get_modelnet_dataloader_from_args(args: Namespace) -> DataLoader[ModelNet]:
    """ Same as [polar.train.data.factory.get_modelnet_dataloader][], but accepts an `argparse.Namespace` object
        instead of keyword arguments.

    Returns:
        Standard Pytorch DataLoader.
    """
    dataset = ModelNet.from_args(args)
    loader_params = dict(num_workers=args.num_workers, pin_memory=True, shuffle=True, drop_last=False)
    return DataLoader(dataset, **loader_params, batch_size=args.batch_size)


def get_modelnet_dataloaders_from_args(args: Namespace) -> tuple[DataLoader[ModelNet], DataLoader[ModelNet]]:
    """ Same as [polar.train.data.factory.get_modelnet_dataloader_from_args][], but returns a tuple of train and test
        dataloaders.

    Returns:
        Train loader, Test loader.
    """
    args.split = 'train'
    train_loader = get_modelnet_dataloader_from_args(args)
    args.split = 'test'
    test_loader = get_modelnet_dataloader_from_args(args)
    return train_loader, test_loader


# _______________________________________________ 2. Augmented Loaders ______________________________________________ #

def get_augmented_dataloader(
    batch_size: int, num_workers: int,
    rootdir: str = 'modelnet', split: str = 'train', classes: Sequence[str] | None = None,
    exclude_classes: Sequence[str] | None = None, samples_per_class: int | None = None, return_labels: bool = False,
    num_points: int = 1024, shuffle: bool = False, sigma: float = 0., min_scale: float = 1.,
    max_angle: float = 180., max_trans: float = 0., keep_ratio: float = 1., p: float = 1, handle_device: bool = True
) -> AugmentedDataLoader:
    """ See polar.train.data.factory.AugmentedDataLoader for the arguments description.

    Returns:
        A ModelNet40 dataloader with random motions and degradations. 
    """
    dataloader = get_modelnet_dataloader(batch_size, num_workers, rootdir, split, classes, exclude_classes,
                                         samples_per_class, return_labels)
    augmented_dataloader = AugmentedDataLoader(dataloader, num_points, shuffle, sigma, min_scale, max_angle, max_trans,
                                               keep_ratio, p, handle_device)
    return augmented_dataloader


def get_augmented_dataloaders(
    batch_size: int, num_workers: int,
    rootdir: str = 'modelnet', classes: Sequence[str] | None = None,
    exclude_classes: Sequence[str] | None = None, samples_per_class: int | None = None, return_labels: bool = False,
    num_points: int = 1024, shuffle: bool = False, sigma: float = 0., min_scale: float = 1.,
    max_angle: float = 180., max_trans: float = 0., keep_ratio: float = 1., p: float = 1, handle_device: bool = True
) -> tuple[AugmentedDataLoader, AugmentedDataLoader]:
    """ See polar.train.data.factory.AugmentedDataLoader for the arguments description. Same as
        [polar.train.data.factory.get_augmented_dataloader][], but but returns a tuple of train and test dataloaders.

    Returns:
        A ModelNet40 dataloader with random motions and degradations. 
    """
    split = 'train'
    train_loader = get_augmented_dataloader(**locals())
    split = 'test'
    locals_ = {k: v for k, v in locals().items() if k != 'train_loader'}
    test_loader = get_augmented_dataloader(**locals_)
    return train_loader, test_loader


def get_augmented_dataloader_from_args(args: Namespace) -> AugmentedDataLoader:
    """ See polar.train.data.factory.AugmentedDataLoader for the arguments description. Same as
        [polar.train.data.factory.get_augmented_dataloader][], but accepts an `argparse.Namespace` object
        instead of keyword arguments.

    Returns:
        A ModelNet40 dataloader with random motions and degradations. 
    """
    dataloader = get_modelnet_dataloader_from_args(args)
    keys: tuple[str, ...] = ('num_points', 'shuffle', 'sigma', 'min_scale', 'max_angle', 'max_trans', 'keep_ratio',
                             'source_only', 'p', 'handle_device')
    params = args_to_param_subset(args, keys)
    return AugmentedDataLoader(dataloader, **params)


def get_augmented_dataloaders_from_args(args: Namespace) -> tuple[AugmentedDataLoader, AugmentedDataLoader]:
    """ Same as [polar.train.data.factory.get_augmented_dataloader_from_args][], but returns a tuple of train and test
        dataloaders.

    Returns:
        Train loader, Test loader.
    """
    args.split = 'train'
    train_loader = get_augmented_dataloader_from_args(args)
    args.split = 'test'
    test_loader = get_augmented_dataloader_from_args(args)
    return train_loader, test_loader
