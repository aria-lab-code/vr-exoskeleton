import torch
import torch.nn as nn

DIMS_EYE = 6  # (x, y, z) of left and right eyes.
DIMS_HEAD_DIRECTION = 3  # (x, y, z) direction of head.
DIMS_OUT_PITCH_YAW = 2  # Output: predict the pitch and yaw of the next direction.


class GazeMLP(nn.Module):

    def __init__(self, hidden_sizes=None, predict_relative_head=False):
        super(GazeMLP, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = list()

        self.input_size = DIMS_EYE + DIMS_HEAD_DIRECTION
        dims_out = DIMS_OUT_PITCH_YAW if predict_relative_head else DIMS_HEAD_DIRECTION

        self.net = nn.Sequential()
        in_dim = self.input_size
        for hidden_size in hidden_sizes:
            self.net.append(nn.Linear(in_dim, hidden_size))
            self.net.append(nn.ReLU())
            in_dim = hidden_size
        self.net.append(nn.Linear(in_dim, dims_out))

    def forward(self, x):
        return self.net(x)


class GazeLSTM(nn.Module):

    def __init__(self, hidden_sizes=None, predict_relative_head=False):
        super(GazeLSTM, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = list()

        self.input_size = DIMS_EYE + DIMS_HEAD_DIRECTION
        dims_out = DIMS_OUT_PITCH_YAW if predict_relative_head else DIMS_HEAD_DIRECTION

        if len(hidden_sizes) == 0:
            self.hidden_size = dims_out
            self.lstm = nn.LSTM(self.input_size, dims_out)
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
            self.net.append(nn.Linear(in_dim, dims_out))

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

        return y, hn, cn  # (N, 3), (1, N, H), (1, N, H)
