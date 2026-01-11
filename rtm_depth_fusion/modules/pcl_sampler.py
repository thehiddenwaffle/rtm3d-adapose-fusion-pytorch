import typing as ty
from collections.abc import Iterable

import torch as tch
import torch.nn as nn


class PCLSampler(nn.Module):
    """PCL Sampler directly inspired by AdaPose(not the Wi-Fi based one)"""

    def __init__(
        self,
        enc_feats: ty.Tuple[int] = (3, 64),
        dec_feats: ty.Tuple[int] = (64, 256, 512),
        pc_dim_out: int = 56,
    ):
        super().__init__()

        assert isinstance(enc_feats, Iterable) and isinstance(dec_feats, Iterable)
        assert len(enc_feats) > 1 and len(dec_feats) > 1

        self.pc_dim_out = pc_dim_out

        # Encoder: 1D convs over points
        encoder_feat_pairs = (
            (nn.Conv1d(in_l, out_l, 1), nn.ReLU(inplace=True))
            for in_l, out_l in zip(enc_feats, enc_feats[1:])
        )
        self.encoder = nn.Sequential(*sum(encoder_feat_pairs, ()))  # flatten and spread

        # Decoder: FC layers
        decoder_feat_pairs = (
            (nn.Linear(in_l, out_l), nn.ReLU(inplace=True))
            for in_l, out_l in zip(dec_feats, dec_feats[1:])
        )
        self.decoder = nn.Sequential(
            *sum(decoder_feat_pairs, ()),
            nn.Linear(dec_feats[-1], pc_dim_out * 3),  # output
        )

    def load_tf_weights(self, tf_weights):
        # ---- Encoder ----
        conv_layers = [l for l in self.encoder if isinstance(l, nn.Conv1d)]
        for i, layer in enumerate(conv_layers):
            w = tch.from_numpy(tf_weights.get_tensor(f"encoder_conv_layer_{i}/W"))
            b = tch.from_numpy(tf_weights.get_tensor(f"encoder_conv_layer_{i}/b"))

            layer.weight.data.copy_(w.squeeze(0).permute(2, 1, 0).contiguous())
            layer.bias.data.copy_(b)

        # ---- Decoder ----
        linear_layers = [l for l in self.decoder if isinstance(l, nn.Linear)]
        for i, layer in enumerate(linear_layers):
            w = tch.from_numpy(tf_weights.get_tensor(f"decoder_fc_{i}/W"))
            b = tch.from_numpy(tf_weights.get_tensor(f"decoder_fc_{i}/b"))

            layer.weight.data.copy_(w.permute(1, 0).contiguous())
            layer.bias.data.copy_(b)

    def forward(self, input_pc):
        B, T, J, N, _ = input_pc.shape

        x = input_pc.reshape(B * T * J, N, 3)
        x = x.permute(0, 2, 1)

        x = self.encoder(x)
        x = tch.max(x, dim=2).values

        x = self.decoder(x)

        x = x.reshape(B * T, J, self.pc_dim_out, 3)
        return x
