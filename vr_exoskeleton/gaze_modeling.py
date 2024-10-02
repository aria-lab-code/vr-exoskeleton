import torch
import torch.nn as nn

OUTPUT_SIZE = 3  # Predict (x,y,z)_{t+1} of head.


class GazeMLP(nn.Module):

    def __init__(self, instance_size, window_size=3, hidden_sizes=None):
        super(GazeMLP, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = list()

        self.input_size = instance_size * window_size
        self.net = nn.Sequential()
        # Input is formatted as left eye, right eye, and head vectors for each time step in the context window:
        #   (*(*(x,y[,z])_{eye_l}, *(x,y[,z])_{eye_r}, *(x,y,z)_{head})_{t-w+1},
        #    *(*(x,y[,z])_{eye_l}, *(x,y[,z])_{eye_r}, *(x,y,z)_{head})_{t-w+2},
        #     ...
        #    *(*(x,y[,z])_{eye_l}, *(x,y[,z])_{eye_r}, *(x,y,z)_{head})_{t}     )
        in_dim = self.input_size
        for hidden_size in hidden_sizes:
            self.net.append(nn.Linear(in_dim, hidden_size))
            self.net.append(nn.ReLU())
            in_dim = hidden_size
        self.net.append(nn.Linear(in_dim, OUTPUT_SIZE))

    def forward(self, x):
        return self.net(x)


class GazeLSTM(nn.Module):

    def __init__(self, instance_size, hidden_sizes=None):
        super(GazeLSTM, self).__init__()

        self.input_size = instance_size
        if hidden_sizes is None or len(hidden_sizes) == 0:
            self.hidden_size = OUTPUT_SIZE
            self.lstm = nn.LSTM(instance_size, OUTPUT_SIZE)
            self.net = None
        else:
            self.hidden_size = hidden_sizes[0]
            self.lstm = nn.LSTM(instance_size, hidden_sizes[0])
            self.net = nn.Sequential()
            in_dim = hidden_sizes[0]
            for hidden_size in hidden_sizes[1:]:
                self.net.append(nn.Linear(in_dim, hidden_size))
                self.net.append(nn.ReLU())
                in_dim = hidden_size
            self.net.append(nn.Linear(in_dim, OUTPUT_SIZE))

    def forward(self, x, h0=None, c0=None):  # (L, N, I), (1, N, H), (1, N, H)
        """
        Predict next head position from previous gaze and head position.
        :param x: Sequence and/or batch of gaze vector(s) as `[*gaze_l, *gaze_r, *head]`.
        :param h0: Initial hidden state.
        :param c0: Initial context vector.
        :return: `(y, hn, cn)` as a tuple.
        """
        # L = sequence length
        # N = batch_size
        # I = instance_size
        # H = hidden_size
        # O = OUTPUT_SIZE
        if h0 is not None and c0 is not None:
            hx0 = h0, c0
        else:
            hx0 = None
        x, (hn, cn) = self.lstm(x, hx0)  # (L, N, H), ((1, N, H), (1, N, H))

        if self.net is None:
            y = x
        else:
            y = torch.stack([self.net(x[i]) for i in range(x.shape[0])], dim=0)

        return y, hn, cn
