import torch.nn as nn


class GazeMLP(nn.Module):

    def __init__(self, window_size, hidden_sizes=None):
        super(GazeMLP, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        self.net = nn.Sequential()
        in_dim = 9 * window_size
        for hidden_size in hidden_sizes:
            self.net.append(nn.Linear(in_dim, hidden_size))
            self.net.append(nn.ReLU())
            in_dim = hidden_size
        self.net.append(nn.Linear(in_dim, 3))  # Predict (x,y,z)_{t+1} of head.

    def forward(self, x):
        return self.net(x)
