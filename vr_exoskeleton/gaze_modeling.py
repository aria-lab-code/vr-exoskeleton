import torch.nn as nn

OUTPUT_SIZE = 3  # Predict (x,y,z)_{t+1} of head.


class GazeMLP(nn.Module):

    def __init__(self, window_size=3, hidden_sizes=None, drop_gaze_z=False):
        super(GazeMLP, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = list()

        self.net = nn.Sequential()
        # Input is formatted as left eye, right eye, and head vectors for each time step in the context window:
        #   (*(*(x,y[,z])_{eye_l}, *(x,y[,z])_{eye_r}, *(x,y,z)_{head})_{t-w+1},
        #    *(*(x,y[,z])_{eye_l}, *(x,y[,z])_{eye_r}, *(x,y,z)_{head})_{t-w+2},
        #     ...
        #    *(*(x,y[,z])_{eye_l}, *(x,y[,z])_{eye_r}, *(x,y,z)_{head})_{t}     )
        instance_size = 7 if drop_gaze_z else 9
        in_dim = instance_size * window_size
        for hidden_size in hidden_sizes:
            self.net.append(nn.Linear(in_dim, hidden_size))
            self.net.append(nn.ReLU())
            in_dim = hidden_size
        self.net.append(nn.Linear(in_dim, OUTPUT_SIZE))

    def forward(self, x):
        return self.net(x)


class GazeLSTM(nn.Module):

    def __init__(self, hidden_size=32, num_layers=1, drop_gaze_z=False):
        super(GazeLSTM, self).__init__()

        instance_size = 7 if drop_gaze_z else 9
        self.lstm = nn.LSTM(
            instance_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            proj_size=OUTPUT_SIZE
        )

    def forward(self, x, hx_0=None):
        """
        Predict next head position from previous gaze and head position.
        :param x: Batch or single vector of `[*gaze_l, *gaze_r, *head]`.
        :param hx_0: As a tuple `(h_0, c_0)`.
        :return: `(y, (h_n, c_n))` as a tuple where the second element is also itself a tuple.
        """
        # N = batch_size
        # L = sequence length
        # H_{in} = instance_size
        # Y = num_layers
        # H_{out} = proj_size
        # H_{cell} = hidden_size
        # input: (N, L, H_{in}), ((Y, N, H_{out}), (Y, N, H_{cell}))
        return self.lstm(x, hx_0)  # output: (N, L, H_{out}), ((Y, N, H_{out}), (Y, N, H_{cell}))
