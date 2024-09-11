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

    def forward(self, x, h_0=None, c_0=None):  # (L, N, I), (1, N, H), (1, N, H)
        """
        Predict next head position from previous gaze and head position.
        :param x: Sequence and/or batch of gaze vector(s) as `[*gaze_l, *gaze_r, *head]`.
        :param h_0: Initial hidden state.
        :param c_0: Initial context vector.
        :return: `(y, h_n, c_n)` as a tuple.
        """
        # L = sequence length
        # N = batch_size
        # I = instance_size
        # H = hidden_size
        # O = OUTPUT_SIZE
        if h_0 is not None and c_0 is not None:
            hx_0 = h_0, c_0
        else:
            hx_0 = None
        x, (h_n, c_n) = self.lstm(x, hx_0)  # (L, N, H), ((1, N, H), (1, N, H))

        if self.net is None:
            y = x
        else:
            y = torch.zeros((x.shape[0], x.shape[1], OUTPUT_SIZE)).to(x.device)  # (L, N, O)
            for i in range(x.shape[0]):
                y[i] = self.net(x[i])  # Like `TimeDistributed` in Keras.

        return y, h_n, c_n
