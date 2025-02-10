import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt

from vr_exoskeleton import gaze_modeling

# Below are calculated in the 'explore' notebook.
EYE_DIFF_MEAN = [0.00727914, 0.00137031]
EYE_DIFF_COV = [[7.67945083e-04, 4.95395192e-05],
                [4.95395192e-05, 7.07410755e-04]]


def predict_thetas(model, points, v_head, rng, sample=10, device=None):
    if isinstance(v_head, list):
        v_head = np.array(v_head)
    V_head = np.zeros((sample, 3)) + v_head  # (n, 3)

    kwargs = dict()
    if isinstance(model, gaze_modeling.GazeLSTM):
        h0, c0 = None, None
        X = torch.tensor([[0., 0., 1., 0., 0., 1., 0., 0., 1.]] * sample, dtype=torch.float32)
        if device is not None:
            X = X.to(device)
        for _ in range(40):
            _, h0, c0 = model(X, h0=h0, c0=c0)
        kwargs['h0'] = h0
        kwargs['c0'] = c0

    point_to_theta = dict()
    for x, y in points:
        diff_sample = rng.multivariate_normal(EYE_DIFF_MEAN, EYE_DIFF_COV, sample)
        x_diff_sample = diff_sample[:, 0]
        y_diff_sample = diff_sample[:, 1]
        x_sample_left = x - x_diff_sample / 2
        y_sample_left = y - y_diff_sample / 2
        z_sample_left = np.sqrt(1 - x_sample_left ** 2 - y_sample_left ** 2)
        x_sample_right = x + x_diff_sample / 2
        y_sample_right = y + y_diff_sample / 2
        z_sample_right = np.sqrt(1 - x_sample_right ** 2 - y_sample_right ** 2)
        X = np.transpose(np.stack([x_sample_left, y_sample_left, z_sample_left,
                                   x_sample_right, y_sample_right, z_sample_right]))  # (n, 6)
        X = np.concatenate([X, V_head], axis=1)  # (n, 9)
        X = torch.tensor(X, dtype=torch.float32)
        if device is not None:
            X = X.to(device)
        Y_hat = model(X, **kwargs)
        if isinstance(Y_hat, tuple):  # LSTM returns hidden states.
            Y_hat, _, _ = Y_hat
        Y_hat = Y_hat.detach().cpu().numpy()
        point_to_theta[(x, y)] = np.mean(Y_hat, axis=0)  # As pitch, yaw.

    return point_to_theta


def hist_thetas(point_to_theta, path=None):
    thetas = [np.sqrt(theta_y ** 2 + theta_x ** 2) for theta_y, theta_x in point_to_theta.values()]

    plt.hist(thetas, bins=128)
    plt.title('Frequency of length of predicted head pitch/yaw')
    plt.ylabel('Length (as sqrt(th_y^2 + th_x^2)) of head pitch/yaw')

    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_vector_field(point_to_theta, title=None, path=None):
    # https://matplotlib.org/stable/gallery/color/individual_colors_from_cmap.html
    cmaps = [mpl.colormaps[name] for name in ('Blues_r', 'Oranges_r', 'Greens_r', 'Purples_r')]

    thetas = [np.sqrt(theta_y ** 2 + theta_x ** 2) for theta_y, theta_x in point_to_theta.values()]
    theta_min = min(thetas)
    theta_max = max(thetas)

    plt.figure(figsize=(9.0, 10.8))
    for (x, y), (theta_y, theta_x) in point_to_theta.items():
        theta = np.sqrt(theta_y ** 2 + theta_x ** 2)
        if theta_y > 0:
            if theta_x > 0:
                cmap = cmaps[0]
            else:
                cmap = cmaps[1]
        else:
            if theta_x > 0:
                cmap = cmaps[2]
            else:
                cmap = cmaps[3]
        color = cmap((theta - theta_min) / (theta_max - theta_min + 0.0001))
        plt.arrow(x, y, -theta_x, theta_y, color=color)
    if title is not None:
        plt.title(title)
    plt.ylabel('Y Left/Right Average')
    plt.xlabel('X Left/Right Average')

    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
