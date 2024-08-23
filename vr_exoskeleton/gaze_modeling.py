import torch.nn as nn


class GazeMLP(nn.Module):

    def __init__(self, window_size=3, hidden_sizes=None, drop_gaze_z=False):
        super(GazeMLP, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = list()

        self.net = nn.Sequential()
        # Input is formatted as left eye, right eye, and head vectors for each time step in the context window:
        #   (*(*(x,y,z)_{eye_l}, *(x,y,z)_{eye_r}, *(x,y,z)_{head})_{t-w+1},
        #    *(*(x,y,z)_{eye_l}, *(x,y,z)_{eye_r}, *(x,y,z)_{head})_{t-w+2},
        #     ...
        #    *(*(x,y,z)_{eye_l}, *(x,y,z)_{eye_r}, *(x,y,z)_{head})_{t}     )
        instance_size = 9
        if drop_gaze_z:
            instance_size = 7
        in_dim = instance_size * window_size
        for hidden_size in hidden_sizes:
            self.net.append(nn.Linear(in_dim, hidden_size))
            self.net.append(nn.ReLU())
            in_dim = hidden_size
        self.net.append(nn.Linear(in_dim, 3))  # Predict (x,y,z)_{t+1} of head.

    def forward(self, x):
        return self.net(x)
