import torch
import torch.nn as nn

INSTANCE_SIZE = 9  # (x,y,z) for each of gaze_left_eye, gaze_right_eye, head.
OUTPUT_SIZE = 3  # Predict (x,y,z)_{t+1} of head.


class GazeMLP(nn.Module):

    def __init__(self, hidden_sizes=None):
        super(GazeMLP, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = list()

        self.input_size = INSTANCE_SIZE  # Input is formatted as left eye, right eye, and head vectors.
        self.net = nn.Sequential()
        in_dim = self.input_size
        for hidden_size in hidden_sizes:
            self.net.append(nn.Linear(in_dim, hidden_size))
            self.net.append(nn.ReLU())
            in_dim = hidden_size
        self.net.append(nn.Linear(in_dim, OUTPUT_SIZE))

    def forward(self, x):
        return self.net(x)


class GazeLSTM(nn.Module):

    def __init__(self, hidden_sizes=None):
        super(GazeLSTM, self).__init__()

        self.input_size = INSTANCE_SIZE
        if hidden_sizes is None or len(hidden_sizes) == 0:
            self.hidden_size = OUTPUT_SIZE
            self.lstm = nn.LSTM(INSTANCE_SIZE, OUTPUT_SIZE)
            self.net = None
        else:
            self.hidden_size = hidden_sizes[0]
            self.lstm = nn.LSTM(INSTANCE_SIZE, hidden_sizes[0])
            self.net = nn.Sequential()
            in_dim = hidden_sizes[0]
            for hidden_size in hidden_sizes[1:]:
                self.net.append(nn.Linear(in_dim, hidden_size))
                self.net.append(nn.ReLU())
                in_dim = hidden_size
            self.net.append(nn.Linear(in_dim, OUTPUT_SIZE))

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


class AngleLoss(nn.Module):

    def forward(self, input_: torch.Tensor, target: torch.Tensor):
        return torch.mean(angle_error(input_, target, dim=-1))


def angle_error(input_: torch.Tensor, target: torch.Tensor, dim=None):
    input_norm = torch.linalg.vector_norm(input_, dim=dim)
    target_norm = torch.linalg.vector_norm(target, dim=dim)
    eps = torch.zeros(input_norm.size(), dtype=input_norm.dtype, device=input_.device) + 1e-10  # Avoid division by zero.
    input_norm = torch.maximum(eps, input_norm)
    target_norm = torch.maximum(eps, target_norm)
    return torch.acos((input_ * target).sum(dim=dim) / input_norm / target_norm)

def main():
    a = torch.tensor([[0.0, 0.0, 1.0]])
    b = torch.tensor([[0.0, 0.0, 1.0]])
    print(angle_error(a, b))

    a = torch.tensor([[1.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 0.0, 1.0]])
    print(angle_error(a, b))

    a = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    print(angle_error(a, b, dim=1))


if __name__ == '__main__':
    main()
