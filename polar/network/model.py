from __future__ import annotations
from argparse import Namespace
import torch
from torch import nn, Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MLP, global_max_pool


# +---------------------------------------------------------------------------------------------+ #
# |                                      POINTNET ENCODER                                       | #
# +---------------------------------------------------------------------------------------------+ #

class PointNet(nn.Module):

    def __init__(self, in_channels: int = 3, first_stage_widths: tuple[int, ...] = (64, 64),
                 second_stage_widths: tuple[int, ...] = (64, 128, 1024)) -> None:
        super().__init__()
        self.local_mlp = MLP([in_channels, *first_stage_widths])
        self.global_mlp = MLP(list(second_stage_widths))

    @staticmethod
    def to_batch(data: Batch | Tensor | list[Tensor]) -> Batch:
        if isinstance(data, Batch):
            return data
        data = Batch.from_data_list([Data(x=pcd, pos=pcd) for pcd in data])
        return data

    def forward(self, data: Batch) -> Tensor:
        data = PointNet.to_batch(data)
        x, batch = data.x, data.batch
        x = self.local_mlp(x, batch)
        x = self.global_mlp(x, batch)
        x = global_max_pool(x, batch)
        return x


# +---------------------------------------------------------------------------------------------+ #
# |                                     RECONSTRUCTION HEAD                                     | #
# +---------------------------------------------------------------------------------------------+ #

class ReconstructionHead(nn.Module):

    def __init__(self, latent_dim: int = 1024, widths: tuple[int, ...] = (1024, 1024),
                 num_points: int = 2048, dropout: float = 0.1, out_channels: int = 3) -> None:
        super().__init__()
        self.mlp = MLP([latent_dim, *widths, num_points * out_channels], dropout=dropout)
        self.num_points = num_points
        self.out_channels = out_channels

    def forward(self, z: Tensor) -> Tensor:
        return self.mlp(z).view(-1, self.num_points, self.out_channels)


# +---------------------------------------------------------------------------------------------+ #
# |                                        AUTO-ENCODER                                         | #
# +---------------------------------------------------------------------------------------------+ #

class PointNetAE(nn.Module):

    def __init__(self, encoder: PointNet, decoder: ReconstructionHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def build(
        cls,
        in_channels: int = 3,
        first_stage_widths: tuple[int, ...] = (64, 64),
        second_stage_widths: tuple[int, ...] = (64, 128, 1024),
        decoder_widths: tuple[int, ...] = (1024, 1024),
        num_points: int = 1024,
        dropout: float = 0.1,
        out_channels: int = 3,
    ) -> PointNetAE:
        """ Main constructor for the PointNetAE class.

        Args:
            in_channels (int, optional): Input channels. Defaults to 3.
            first_stage_widths (tuple[int, ...], optional):
                First PointNet MLP layer widths. Defaults to (64, 64).
            second_stage_widths (tuple[int, ...], optional):
                Second PointNet MLP layer widths. Defaults to (64, 128, 1024).
            decoder_widths (tuple[int, ...], optional):
                Decoder MLP layer widths. Defaults to (1024, 1024).
            num_points (int, optional):
                Decoder number of outputted points. Defaults to 1024.
            dropout (float, optional): Decoder MLP dropout rate. Defaults to 0.1.
            out_channels (int, optional): Output channels. Defaults to 3.

        Returns:
            PointNetAE: PointNet autoencoder.
        """
        latent_dim = second_stage_widths[-1]
        encoder = PointNet(in_channels, first_stage_widths, second_stage_widths)
        decoder = ReconstructionHead(latent_dim, decoder_widths, num_points, dropout, out_channels)
        return cls(encoder, decoder)

    @classmethod
    def from_argparse(cls, args: Namespace) -> PointNetAE:
        """ Same as [polar.network.model.PointNetAE.build][] but taking parameters as a `Namespace` object.

        Args:
            args (Namespace): `argparse.Namespace` containing the arguments of the build method.

        Returns:
            PointNetAE: PointNet autoencoder.
        """
        ae_params: tuple[str, ...] = ('first_stage_widths', 'second_stage_widths',
                                      'decoder_widths', 'num_points', 'dropout')
        ae_params_dict = {k: getattr(args, k) for k in ae_params}
        ae = cls.build(**ae_params_dict)
        if 'checkpoint' in args and args.checkpoint is not None:
            ae.load_state_dict(torch.load(args.checkpoint)['network'])
        return ae

    def encode(self, x: Tensor | Batch) -> Tensor:
        """ Encode a batch of point clouds.

        Args:
            x (Tensor | Batch): `(batch_size, num_points_i, spatial_dim)`.

        Returns:
            Tensor: `(batch_size, latent_dim)`.
        """
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """ Decode a batch of latent vectors.

        Args:
            z (Tensor): `(batch_size, latent_dim)`.

        Returns:
            Tensor: `(batch_size, num_points, spatial_dim)`.
        """
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

    @property
    def num_params(self) -> dict[str, str]:
        """ Formatted number of trainable parameters of the encoder and of the decoder.

        Returns:
            **Num Params**:
                Keys: (encoder, decoder). Values: formatted strings of the form '10,000,000'.
        """
        params_as_str = lambda m: f'{sum(p.numel() for p in m.parameters() if p.requires_grad):,}'
        encoder = params_as_str(self.encoder)
        decoder = params_as_str(self.decoder)
        return dict(encoder=encoder, decoder=decoder)
