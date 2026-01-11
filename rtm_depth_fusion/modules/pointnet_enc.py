import typing as ty

import torch as tch
import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    """PointNet Set Abstraction (SA) Module from AdaPose"""

    def __init__(self, mlp: ty.Tuple[int] = (64, 128, 1024)):
        """
        mlp: tuple of feature sizes
        """
        super().__init__()

        self.mlp_iter = mlp

        # conv1: 1x3 over (J, N) with input channel = 1
        conv1_out_feat = 64
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=conv1_out_feat,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=0,
        )

        mlp_input_feat = 64
        # conv2: 1x1
        self.conv2 = nn.Conv2d(conv1_out_feat, mlp_input_feat, kernel_size=1)

        in_feat = mlp_input_feat
        layers = []
        convs_only = [self.conv1, self.conv2]
        for out_feat in mlp:
            conv = nn.Conv2d(in_feat, out_feat, kernel_size=1)
            layers.append(conv)
            convs_only.append(conv)
            layers.append(nn.ReLU(inplace=True))
            in_feat = out_feat

        self.mlp = nn.Sequential(*layers)
        self.all_conv_layers = convs_only

    def load_tf_weights(self, tf_weights):
        conv_tf_names = [
            (f"gen/gen/conv{i + 1}/weights", f"gen/gen/conv{i + 1}/biases", l)
            for i, l in enumerate(self.all_conv_layers)
        ]

        for w_name, b_name, layer in conv_tf_names:
            w = tch.from_numpy(
                tf_weights.get_tensor(w_name)
            )  # TF: [1, 1, in_ch, out_ch]
            b = tch.from_numpy(tf_weights.get_tensor(b_name))

            # Convert to PyTorch Conv1d: [out_ch, in_ch, kernel_size]
            w_pt = w.permute(3, 2, 0, 1).contiguous()
            layer.weight.data.copy_(w_pt)
            layer.bias.data.copy_(b)

    def forward(self, pcloud):
        """
        pcloud: (B*T, J, N, 3)
        """

        B_T, J, N, _ = pcloud.shape

        x = pcloud.view(B_T, 1, J * N, 3)

        # conv1: kernel=(1,3) over last dim
        x = self.conv1(x)  # (B*T, 64, J, N*3-2) # TODO
        x = F.relu(x)

        # conv2: 1x1
        x = self.conv2(x)
        x = F.relu(x)

        # MLP stack
        x = self.mlp(x)  # (B*T, mlp[-1], J, *)

        # Max pool over J dimension (num_point in TF code)
        x = tch.max(x, dim=2).values  # (B*T, C, 1, *)

        return x.view(B_T, self.mlp_iter[-1])
