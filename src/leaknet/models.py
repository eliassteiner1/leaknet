"""
The model(s). There is really only one network: a "double attention" encoder that first looks across all sensors of
one timestep and then across all timesteps, giving one 100-d vector per scene. What changed between the experiments is
only the small head on top of it (and the loss, see losses.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from leaknet.attention import AttentionPooling, SelfAttention

N_TIMESTEPS = 100
N_SENSORS_MAX = 10
N_SENSOR_FEATS = 5  # [x, y, concentration, wind direction, wind speed]


def build_input(sample):
    """
    Puts the sample dict into the form [BS, Time, Sensor_nr, features_per_sensor]. Every sensor at every timestep gets
    its own position and concentration plus the (global) wind direction and speed of that timestep.
    """

    pos_sensors = sample["pos_sensors"]  # [BS, S, 2]
    con_sensors = sample["con_sensors"]  # [BS, T, S]
    wind_direc = sample["wind_direc"]  # [BS, T]
    wind_speed = sample["wind_speed"]  # [BS, T]
    BS, T, S = con_sensors.shape

    x = torch.cat(
        (
            pos_sensors[:, None, :, :].expand(BS, T, S, 2),
            con_sensors[:, :, :, None],
            wind_direc[:, :, None, None].expand(BS, T, S, 1),
            wind_speed[:, :, None, None].expand(BS, T, S, 1),
        ),
        dim=3,
    )

    return x.float()


class DoubleAttentionEncoder(nn.Module):
    """scene (= over sensors) attention -> pooling -> time attention -> pooling. Output: [BS, 100]"""

    def __init__(self) -> None:
        super().__init__()

        self.linear_1x1 = nn.Linear(5, 20)
        self.linear_1x2 = nn.Linear(20, 20)
        self.scene_attention_1 = SelfAttention(20, 20, 20)
        self.linear_1x3 = nn.Linear(20, 20)
        self.scene_attention_2 = SelfAttention(20, 20, 20)
        self.scene_attention_pool = AttentionPooling(20, [20])

        self.linear_2x1 = nn.Linear(20, 100)
        self.linear_2x2 = nn.Linear(100, 100)
        self.time_attention_1 = SelfAttention(100, 100, 100)
        self.linear_2x3 = nn.Linear(100, 100)
        self.time_attention_2 = SelfAttention(100, 100, 100)
        self.time_attention_pool = AttentionPooling(100, [100])

        self.activation = nn.ReLU()

    def forward(self, sample):
        # data preparation ---------------------------------------------------------------------------------------------
        x = build_input(sample)
        # the mask is the same for all timesteps, repeat it to match the shape for the attention layer
        M = sample["msk_sensors"].unsqueeze(1).repeat(1, N_TIMESTEPS, 1)

        # aggregate all the broadcasting dimensions in one [BS*time, n_sensors, n_feats]
        x = x.reshape(-1, N_SENSORS_MAX, N_SENSOR_FEATS)
        M = M.reshape(-1, N_SENSORS_MAX)

        # forward function ---------------------------------------------------------------------------------------------
        # scene-elementwise linear layers
        x = self.activation(self.linear_1x1(x))
        x = self.activation(self.linear_1x2(x))
        # apply scene-wide attention layer (note: the padded sensors are NOT masked here, only in the pooling!)
        x = self.scene_attention_1(x) + x
        # scene-elementwise linear layers
        x = self.activation(self.linear_1x3(x))
        # apply scene-wide attention layer
        x = self.scene_attention_2(x) + x
        # apply scene-wide attention pooling (squeeze really important because output is [BS*time, 1, n_feats=20])
        x = self.scene_attention_pool(x, M).squeeze(1)

        # redistribute time dimension again (now only need to broadcast along BS dimension)
        x = x.reshape(-1, N_TIMESTEPS, 20)

        # time-elementwise linear layers
        x = self.activation(self.linear_2x1(x))
        x = self.activation(self.linear_2x2(x))
        # apply time-wide attention layer
        x = self.time_attention_1(x) + x
        # time-elementwise linear layers
        x = self.activation(self.linear_2x3(x))
        # apply time-wide attention layer
        x = self.time_attention_2(x) + x
        # apply time-wide attention pooling (squeeze really important because output is [BS, 1, n_feats=100])
        x = self.time_attention_pool(x).squeeze(1)

        return x


### heads --------------------------------------------------------------------------------------------------------------
class PositionHead(nn.Module):
    """phase 1: just regress the (x, y) of the one leak. -> [BS, 2]"""

    def __init__(self) -> None:
        super().__init__()
        self.linear_3x1 = nn.Linear(100, 100)
        self.linear_3x2 = nn.Linear(100, 50)
        self.linear_3x3 = nn.Linear(50, 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear_3x1(x))
        x = self.activation(self.linear_3x2(x))
        return self.linear_3x3(x)


class SlotHead(nn.Module):
    """phase 2 / sinkhorn: n_slots candidate leaks, each [confidence (logit), x, y]. -> [BS, n_slots, 3]"""

    def __init__(self, n_slots=3) -> None:
        super().__init__()
        self.DIM = n_slots
        self.linear_3x1 = nn.Linear(100, 100)
        self.linear_3x2 = nn.Linear(100, 50)
        self.linear_3x3 = nn.Linear(50, self.DIM * 3)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear_3x1(x))
        x = self.activation(self.linear_3x2(x))
        x = self.linear_3x3(x)
        return x.view(-1, self.DIM, 3)


class BlobHead(nn.Module):
    """phase 2 / heatmap: parameters of 5 gaussian blobs, each [strength, mean x, mean y, s1, s2, alpha]. -> [BS, 5, 6]"""

    def __init__(self) -> None:
        super().__init__()
        self.linear_3x1 = nn.Linear(100, 100)
        self.linear_3x2 = nn.Linear(100, 50)
        self.linear_3x3 = nn.Linear(50, 5 * 6)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear_3x1(x))
        x = self.activation(self.linear_3x2(x))
        x = self.linear_3x3(x)
        return x.view(-1, 5, 6)


class ConvHeatmapHead(nn.Module):
    """phase 2 / heatmap: 128-d embedding -> [2, 8, 8] -> upsampling + convolutions -> [BS, 64, 64] (logits)"""

    def __init__(self) -> None:
        super().__init__()
        self.linear_3x1 = nn.Linear(100, 100)
        self.linear_3x2 = nn.Linear(100, 100)
        self.linear_3x3 = nn.Linear(100, 2 * 8 * 8)

        self.conv_1x1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=0)
        self.conv_2x1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=0)
        self.conv_3x1 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=0)

        self.activation1 = nn.ReLU()
        self.activation2 = nn.LeakyReLU()

    def forward(self, x):
        x = self.activation1(self.linear_3x1(x))
        x = self.activation1(self.linear_3x2(x))
        x = self.activation1(self.linear_3x3(x))

        # upscale embedding to generate "prediction field"
        x = x.view(-1, 2, 8, 8)  # 8x8 "images" with 2 channels

        x = F.interpolate(x, size=16 + 2, mode="bilinear")
        x = self.activation2(self.conv_1x1(x))
        x = F.interpolate(x, size=32 + 2, mode="bilinear")
        x = self.activation2(self.conv_2x1(x))
        x = F.interpolate(x, size=64 + 2, mode="bilinear")
        x = self.conv_3x1(x)

        return x.squeeze(1)


class LinearHeatmapHead(nn.Module):
    """phase 2 / sanity check: flattened heatmap straight out of a linear layer. -> [BS, 32, 32] (logits)"""

    def __init__(self) -> None:
        super().__init__()
        self.linear_3x1 = nn.Linear(100, 100)
        self.linear_3x2 = nn.Linear(100, 100)
        self.linear_3x3 = nn.Linear(100, 32 * 32)
        self.activation1 = nn.ReLU()

    def forward(self, x):
        x = self.activation1(self.linear_3x1(x))
        x = self.activation1(self.linear_3x2(x))
        x = self.linear_3x3(x)
        return x.view(-1, 32, 32)


class CountHead(nn.Module):
    """phase 2 / sanity check: one number, the number of leaks. -> [BS]"""

    def __init__(self) -> None:
        super().__init__()
        self.linear_3x1 = nn.Linear(100, 50)
        self.linear_3x2 = nn.Linear(50, 10)
        self.linear_3x3 = nn.Linear(10, 1)
        self.activation1 = nn.ReLU()

    def forward(self, x):
        x = self.activation1(self.linear_3x1(x))
        x = self.activation1(self.linear_3x2(x))
        x = self.linear_3x3(x)
        return x.squeeze(1)


HEADS = {
    "position": PositionHead,
    "slots": SlotHead,
    "blobs": BlobHead,
    "conv_heatmap": ConvHeatmapHead,
    "linear_heatmap": LinearHeatmapHead,
    "count": CountHead,
}


# working attention network!
class WithDoubleAttention(nn.Module):
    def __init__(self, head="position") -> None:
        super().__init__()
        self.encoder = DoubleAttentionEncoder()
        self.head = HEADS[head]()

    def forward(self, sample):
        return self.head(self.encoder(sample))


# just flatten the whole input and pass it to an mlp (just learns noise)
class FirstDumbMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.lin1 = nn.Linear(3200, 500)
        self.lin2 = nn.Linear(500, 50)
        self.lin3 = nn.Linear(50, 2)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0)

    def forward(self, sample):
        pos_sensors = sample["pos_sensors"]
        wind_direc = sample["wind_direc"]
        wind_speed = sample["wind_speed"]
        con_sensors = sample["con_sensors"]
        BS, T, S = con_sensors.shape

        # add some noise to the wind patterns
        noise_stren = 0.1
        wind_direc = wind_direc + noise_stren * torch.rand_like(wind_direc)
        wind_speed = wind_speed + noise_stren * torch.rand_like(wind_speed)

        # per timestep: [wind direction, wind speed, all sensor positions, all concentrations] = 32 values
        x = torch.cat(
            (
                wind_direc[:, :, None],
                wind_speed[:, :, None],
                pos_sensors.reshape(BS, 1, S * 2).expand(BS, T, S * 2),
                con_sensors,
            ),
            dim=2,
        )
        x = x.reshape(BS, -1).float()

        x = self.dropout(self.activation(self.lin1(x)))
        x = self.dropout(self.activation(self.lin2(x)))
        x = self.lin3(x)

        return x


def build_model(name):
    """name is either "mlp" (flat baseline) or one of the heads for the double attention network"""
    model = FirstDumbMLP() if name == "mlp" else WithDoubleAttention(head=name)

    # custom model weight initialization (only the linear layers, the convolutions keep the pytorch default)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    return model
