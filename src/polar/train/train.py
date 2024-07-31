from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import toml
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch3d.loss import chamfer_distance
from .data import get_augmented_dataloaders_from_args, AugmentedDataLoader
from ..network import PointNetAE


# _________________________________________ CONTAINERS __________________________________________ #

@dataclass
class DataLoaders:

    train_loader: AugmentedDataLoader
    test_loader:  AugmentedDataLoader

    @classmethod
    def from_argparse(cls, args: Namespace) -> DataLoaders:
        return cls(*get_augmented_dataloaders_from_args(args))


class RegularizedChamferDistance(torch.nn.Module):

    def __init__(self, norm: int = 2, density_weight: float = 0., density_radius: float = 0.1) -> None:
        super(RegularizedChamferDistance, self).__init__()
        self.chamfer = lambda x, y: chamfer_distance(x, y, batch_reduction='sum', norm=norm)[0]
        self.density_weight = density_weight
        self.density_radius = density_radius

    @classmethod
    def from_argparse(cls, args: Namespace) -> RegularizedChamferDistance:
        return cls(args.norm, args.density_weight, args.density_radius)

    def forward(self, X_hat: Tensor, X: Tensor) -> Tensor:
        chamfer_distance = self.chamfer(X_hat, X)
        if self.density_weight == 0:
            return chamfer_distance  # type: ignore
        pairwise_distances = torch.cdist(X_hat, X_hat)
        densities = (pairwise_distances <= self.density_radius).sum(dim=1)
        return chamfer_distance + self.density_weight * densities.float().std()


@dataclass
class Model:

    network:   PointNetAE
    optimizer: AdamW
    scheduler: ReduceLROnPlateau
    criterion: RegularizedChamferDistance
    device:    torch.device

    def get_last_lr(self) -> float:
        return float(self.optimizer.param_groups[0]['lr'])

    @classmethod
    def from_argparse(cls, args: Namespace) -> Model:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = PointNetAE.from_argparse(args).to(device)
        module = network.encoder if args.freeze_decoder else network
        criterion = RegularizedChamferDistance.from_argparse(args)
        optimizer = AdamW(module.parameters(), lr=args.lr)
        if args.checkpoint is not None and args.resume_optimizer:
            optimizer.load_state_dict(torch.load(args.checkpoint)['optimizer'])
        scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        return cls(network, optimizer, scheduler, criterion, device)


# +---------------------------------------------------------------------------------------------+ #
# |                                            TRAIN                                            | #
# +---------------------------------------------------------------------------------------------+ #

def save(model: Model, tracker: dict, dir: str, name: str) -> None:
    path = Path(dir) / f"{name}.pt"
    state = dict(network=model.network.state_dict(),
                 optimizer=model.optimizer.state_dict(),
                 tracker=tracker)
    torch.save(state, path)


def half_epoch(data: DataLoaders, model: Model, training: bool) -> dict[str, float]:
    torch.set_grad_enabled(training)
    model.network.train() if training else model.network.eval()
    model.optimizer.zero_grad(set_to_none=True)
    loader = data.train_loader if training else data.test_loader
    chamfer = length = 0
    for sources, targets, _ in loader:
        x_hat = model.network(sources)
        loss = model.criterion(x_hat, targets)
        chamfer += loss.item()
        length += len(sources)
        if not training:
            continue
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
    chamfer /= length
    if not training:
        model.scheduler.step(chamfer)
    return dict(chamfer=chamfer)


def train(args: Namespace) -> None:
    args.task = 'reconstruction'
    data = DataLoaders.from_argparse(args)
    model = Model.from_argparse(args)
    num_params = model.network.num_params
    print('Num params:')
    print(f"Encoder...: {num_params['encoder']}")
    print(f"Decoder...: {num_params['decoder']}")
    tracker = {'train': {'chamfer': []}, 'val': [], 'lr': []}
    with tqdm(range(args.epochs)) as pbar:
        for _ in pbar:
            train_values = half_epoch(data, model, training=True)
            val_values = half_epoch(data, model, training=False)
            tracker['train']['chamfer'].append(train_values['chamfer'])
            tracker['val'].append(val_values['chamfer'])
            lr = model.get_last_lr()
            tracker['lr'].append(lr)
            display = dict(train_chamfer=train_values['chamfer'],
                           val_chamfer=val_values['chamfer'], lr=lr)
            pbar.set_postfix(display)
    save(model, tracker, args.log_dir, args.name)


def save_args(args: Namespace) -> None:
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(exist_ok=True, parents=True)
    path = log_dir / f'{args.name}.toml'
    with open(path, 'w') as file:
        toml.dump(vars(args), file)


def parse_args() -> Namespace:
    parser = ArgumentParser('Run AE training')
    # REQUIRED
    parser.add_argument('--name', required=True)
    # dataset
    parser.add_argument('--rootdir', type=str, default='modelnet')
    parser.add_argument('--classes', type=str, default=None, nargs='+')
    parser.add_argument('--exclude_classes', type=str, default=None, nargs='+')
    parser.add_argument('--samples_per_class', type=int, default=None)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--num_points', type=int, default=1024)
    # rigid motion
    parser.add_argument('--max_angle', type=int, default=180)
    parser.add_argument('--max_trans', type=float, default=0.0)
    # augment
    parser.add_argument('--sigma', type=float, default=0.0)                 # isotropic noise
    parser.add_argument('--min_scale', type=float, default=1.0)             # scale
    parser.add_argument('--keep_ratio', type=float, default=1.0)            # partial visibility
    parser.add_argument('--p', type=float, default=0.5)                     # augment probability
    # dataloader
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    # model
    parser.add_argument('--first_stage_widths', type=int, default=(64, 64), nargs='+')
    parser.add_argument('--second_stage_widths', type=int, default=(64, 128, 1024), nargs='+')
    parser.add_argument('--decoder_widths', type=int, default=(1024, 1024), nargs='+')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--norm', type=int, default=2)
    parser.add_argument('--density_weight', type=float, default=0.)
    parser.add_argument('--density_radius', type=float, default=0.1)
    # train
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--resume_optimizer', action='store_true')
    parser.add_argument('--freeze_decoder', action='store_true')
    parser.add_argument('--epochs', type=int,  default=150)
    parser.add_argument('--log_dir', default='logs/ae')
    return parser.parse_args()


def main():
    args = parse_args()
    save_args(args)
    train(args)


if __name__ == '__main__':
    main()
