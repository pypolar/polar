from __future__ import annotations
from typing import cast, Sequence
import os
import h5py
from pathlib import Path
import numpy as np
from argparse import Namespace
from numpy.typing import NDArray
from torch.utils.data import Dataset


# ___________________________________________________________________________________________________________________ #

ALL_CLASSES = ('airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup',
               'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop',
               'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa',
               'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox')


NO_SYMMETRIES = ('airplane', 'car', 'chair', 'guitar', 'keyboard', 'laptop', 'mantel',  # half1
                 'monitor', 'person', 'piano', 'plant', 'sofa', 'stairs', 'toilet')


# ___________________________________________________________________________________________________________________ #

def download_modelnet40(dir: Path | str | None = None) -> None:
    """ Download the full ModelNet40 dataset archive as two `.h5` files (train & test) (~ 1 Go).

    Args:
        dir (Path | str | None, optional): Where to store the downloaded filed. Defaults to None.
    """
    if dir is None:
        dir = Path(__file__).resolve().parent / 'modelnet40'
    dir = Path(dir).resolve()
    dir.mkdir(exist_ok=True)
    if not (dir / 'modelnet40_ply_hdf5_2048').exists():
        www = Path('https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip')
        os.system(f'wget --no-check-certificate {www}; unzip {www.name}')
        os.system(f'mv {www.stem} {str(dir)}')
        os.system(f'rm {www.name}')


def load_modelnet(rootdir: Path | str, split: str) -> tuple[NDArray, NDArray]:
    filepath = (Path(rootdir) / split).with_suffix('.h5')
    with h5py.File(filepath, 'r') as f:
        points = f['points' if split == 'train' else 'source'][...]  # type: ignore
        labels = f['labels' if split == 'train' else 'label'][...]  # type: ignore
        points = cast(NDArray, points)
        labels = cast(NDArray, labels)
    return points, labels


def load_and_select_samples(
    rootdir: str, split: str,
    classes: Sequence[str] | None = None, exclude_classes: Sequence[str] | None = None,
    samples_per_class: int | None = None,
) -> tuple[NDArray, NDArray]:
    """ Load ModelNet, select classes to include/exclude, and number of samples per class. """
    if isinstance(classes, (tuple, list)) and len(classes) == 0:
        classes = None
    if isinstance(exclude_classes, (tuple, list)) and len(exclude_classes) == 0:
        exclude_classes = None
    if isinstance(classes, str):
        classes = [classes]
    if isinstance(exclude_classes, str):
        exclude_classes = [exclude_classes]
    # 1. load all samples
    points, labels = load_modelnet(rootdir, split)
    # 2. select classes
    class_name_to_class_idx = lambda name: ALL_CLASSES.index(name.lower().strip())  # noqa: E731
    # class name formatting
    if classes is None:
        classes = ALL_CLASSES
    if list(classes) == ['no_symmetries']:
        classes = NO_SYMMETRIES
    if list(classes) == ['no_symmetries_half1']:
        N = len(NO_SYMMETRIES)
        classes = NO_SYMMETRIES[:N // 2]
    if list(classes) == ['no_symmetries_half2']:
        N = len(NO_SYMMETRIES)
        classes = NO_SYMMETRIES[N // 2:]
    if exclude_classes is not None and list(exclude_classes) == ['no_symmetries']:
        exclude_classes = NO_SYMMETRIES
    if exclude_classes is not None and list(exclude_classes) == ['no_symmetries_half1']:
        N = len(NO_SYMMETRIES)
        exclude_classes = NO_SYMMETRIES[:N // 2]
    if exclude_classes is not None and list(exclude_classes) == ['no_symmetries_half2']:
        N = len(NO_SYMMETRIES)
        exclude_classes = NO_SYMMETRIES[N // 2:]
    # 2.1 to include
    class_indices = np.array(list(map(class_name_to_class_idx, classes)))
    # 2.2 to include
    if exclude_classes is not None:
        excluded_class_indices = np.array(list(map(class_name_to_class_idx, exclude_classes)))
        excluded_sample_indices = np.isin(class_indices, excluded_class_indices, invert=True)
        class_indices = class_indices[excluded_sample_indices]
    sample_indices = np.isin(labels, class_indices)
    points, labels = load_modelnet(rootdir, split)
    points = points[sample_indices]
    labels = labels[sample_indices]
    # 3. select samples per class
    select_samples = lambda label: np.argwhere(labels == label)[:samples_per_class].squeeze()
    indices_by_class = np.concatenate(list(map(select_samples, class_indices)))
    points = points[indices_by_class]
    labels = labels[indices_by_class]
    return points, labels


# ___________________________________________________________________________________________________________________ #

class ModelNet(Dataset):

    def __init__(
        self,
        rootdir: str = 'modelnet', split: str = 'train', classes: Sequence[str] | None = None,
        exclude_classes: Sequence[str] | None = None, samples_per_class: int | None = None, return_labels: bool = False
    ) -> None:
        """_summary_

        Args:
            rootdir (str, optional): Path to the directory containing the `.h5` files. Defaults to 'modelnet'.
            split (str, optional): 'train' or 'test. Defaults to 'train'.
            classes (Sequence[str] | None, optional):
                Shape categories to use. If `None`, load all categories. See `ModelNet.all_classes`. Defaults to `None`.
            exclude_classes (Sequence[str] | None, optional):
                Shape categories to exclude from the dataset. Defaults to `None`.
            samples_per_class (int | None, optional): Number of point clouds per category to load. Defaults to `None`.
            return_labels (bool, optional):
                If `True`, return the class index alongside the point cloud. Defaults to `False`.
        """
        super().__init__()
        load_params = (rootdir, split, classes, exclude_classes, samples_per_class)
        self.points, self.labels = load_and_select_samples(*load_params)
        self.split = split
        self.return_labels = return_labels

    @property
    def all_classes(self) -> tuple[str, ...]:
        """ Tuple of 40 strings, one for each class. """
        return ALL_CLASSES

    @classmethod
    def from_args(cls, args: Namespace) -> ModelNet:
        args_dict = vars(args)
        keys: tuple[str, ...] = ('rootdir', 'split', 'classes', 'exclude_classes', 'samples_per_class', 'return_labels')
        params = {k: args_dict[k] for k in keys if k in args}
        return cls(**params)

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, index) -> NDArray | tuple[NDArray, NDArray]:
        points, class_label = self.points[index], self.labels[index]
        return (points, class_label) if self.return_labels else points
