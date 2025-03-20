import math

import torch
import torch.nn as nn

DIMS_EYE = 6  # (x, y, z) normalized direction of left and right eyes.
DIMS_HEAD_DIRECTION = 3  # (x, y, z) normalized direction of head.
DIMS_OUT_PITCH_YAW = 2  # Output: predict the pitch and yaw of the next direction.


class GazeMLP(nn.Module):

    def __init__(self, hidden_sizes=None):
        super(GazeMLP, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = list()

        self.input_size = DIMS_EYE + DIMS_HEAD_DIRECTION

        self.net = nn.Sequential()
        in_dim = self.input_size
        for hidden_size in hidden_sizes:
            self.net.append(nn.Linear(in_dim, hidden_size))
            self.net.append(nn.ReLU())
            in_dim = hidden_size
        self.net.append(nn.Linear(in_dim, DIMS_OUT_PITCH_YAW))

    def forward(self, x):  # (N, 9)
        return self.net(x)  # (N, 2)


class GazeLSTM(nn.Module):

    def __init__(self, hidden_sizes=None):
        super(GazeLSTM, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = list()

        self.input_size = DIMS_EYE + DIMS_HEAD_DIRECTION

        if len(hidden_sizes) == 0:
            self.hidden_size = DIMS_OUT_PITCH_YAW
            self.lstm = nn.LSTM(self.input_size, DIMS_OUT_PITCH_YAW)
            self.net = None
        else:
            self.hidden_size = hidden_sizes[0]
            self.lstm = nn.LSTM(self.input_size, hidden_sizes[0])
            self.net = nn.Sequential()
            in_dim = hidden_sizes[0]
            for hidden_size in hidden_sizes[1:]:
                self.net.append(nn.Linear(in_dim, hidden_size))
                self.net.append(nn.ReLU())
                in_dim = hidden_size
            self.net.append(nn.Linear(in_dim, DIMS_OUT_PITCH_YAW))

    def forward(self, x, h0=None, c0=None):  # (N, 9), (1, N, H), (1, N, H)
        """
        Predict next head position from previous gaze and head position.
        :param x: Batch of gaze vector(s) as `[*gaze_l, *gaze_r, *head]`.
        :param h0: Initial hidden state.
        :param c0: Initial context vector.
        :return: `(y, hn, cn)` as a tuple.
        """
        # N = batch_size
        # H = hidden_size if self.net is not None else 3
        if h0 is not None and c0 is not None:
            hx0 = h0, c0
        else:
            hx0 = None
        x = torch.unsqueeze(x, 0)  # (1, N, 9)
        x, (hn, cn) = self.lstm(x, hx0)  # (1, N, H), ((1, N, H), (1, N, H))
        x = x[0]  # (N, H)

        if self.net is None:
            y = x
        else:
            y = self.net(x)

        return y, hn, cn  # (N, 2), (1, N, H), (1, N, H)


class GazeVectorBaseline(nn.Module):

    def __init__(self, dead_zone_deg=5.0):
        super(GazeVectorBaseline, self).__init__()
        self.dead_zone_deg = dead_zone_deg
        # 'Steepness' - the speed at which the dead zone tapers off.
        self.a_pitch = torch.tensor(2.0)
        # Soft inflection point of the dead zone.
        self.b_pitch = torch.tensor(dead_zone_deg * math.pi / 180.0)  # Convert to radians.
        # Vertical shift.
        self.c_pitch = torch.asin(torch.tensor(-0.08))  # Start aiming a little below the center.
        # Slope of asymptote - how much the head should rotate as a function of distance from the forward gaze.
        self.v_pitch = torch.tensor(1.0)

        self.a_yaw = torch.tensor(2.0)
        self.b_yaw = torch.tensor(dead_zone_deg * math.pi / 180.0)
        self.c_yaw = torch.tensor(0.0)  # Horizontal shift.
        self.v_yaw = torch.tensor(1.0)

    def forward(self, x):  # (N, 9)
        # Head angle (x[:, 6:9]) is ignored.

        # Calculate combined, normalized gaze.
        g = x[:, :3] + x[:, 3:6]  # (N, 3)
        g /= torch.unsqueeze(torch.linalg.norm(g, dim=1), 1)

        z_sq = g[:, 2] ** 2
        norm_pitch = torch.sqrt(g[:, 1] ** 2 + z_sq)  # (N)
        pitch = torch.asin(g[:, 1] / norm_pitch)
        norm_yaw = torch.sqrt(g[:, 0] ** 2 + z_sq)  # (N)
        yaw = torch.acos(g[:, 0] / norm_yaw) - torch.pi / 2

        pitch = dead_zone(pitch - self.c_pitch, self.a_pitch, self.b_pitch, self.v_pitch)
        yaw = dead_zone(yaw - self.c_yaw, self.a_yaw, self.b_yaw, self.v_yaw)
        return torch.stack([pitch, yaw], dim=1)  # (N, 2)


class GazeVectorParameterized(GazeVectorBaseline):
    """
    A parameterized version of the vector baseline.
    """

    def __init__(self, dead_zone_deg=5.0):
        super(GazeVectorParameterized, self).__init__(dead_zone_deg=dead_zone_deg)
        self.a_pitch = torch.nn.Parameter(self.a_pitch)
        self.b_pitch = torch.nn.Parameter(self.b_pitch)
        self.c_pitch = torch.nn.Parameter(self.c_pitch)
        self.v_pitch = torch.nn.Parameter(self.v_pitch)
        self.a_yaw = torch.nn.Parameter(self.a_yaw)
        self.b_yaw = torch.nn.Parameter(self.b_yaw)
        self.c_yaw = torch.nn.Parameter(self.c_yaw)
        self.v_yaw = torch.nn.Parameter(self.v_yaw)


def dead_zone(x, a, b, v):
    """
    Apply a soft dead zone (i.e., halt movement) with radius `b` to the predicted angular change.
    """
    # Asymptotic in x=[0+, ~b] to y=0.
    # Asymptotic in x=[~b, inf] to y=vx.
    # Inflects at x=b.
    # As `a` increases, the faster the inflection occurs.
    return v * x / (1 + torch.e ** (a * (-x + b)))
