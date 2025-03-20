import argparse
import os
import time
from typing import List, Optional, Set, Union

import numpy as np
import torch
from matplotlib import pyplot as plt

from vr_exoskeleton import data_utils, gaze_modeling, spatial, vector_field


def main():
    """
    Only starts from the command line when this script is run using `python -m vr_exoskeleton.train ...`.
    """
    parser = argparse.ArgumentParser(
        description='Train a neural network on eye gaze and neck data.'
    )
    parser.add_argument('model_type', type=str,
                        help='Model architecture.')
    parser.add_argument('--run_name', type=str,
                        help='Optional name to append to output directory.')
    # Training/test set randomization.
    parser.add_argument('--test_ratio', default=0.25, type=float,
                        help='Ratio of users for whose data will be set aside for evaluation.')
    parser.add_argument('--val_ratio', default=0.1667, type=float,
                        help='Ratio of users within training set for whose data will be set aside for validation.')
    parser.add_argument('--seed', type=int,
                        help='Seeds the random number generator that shuffles and splits the data set.')
    # Data set.
    parser.add_argument('--use_eye_tracker_frames', action='store_true',
                        help='Flag to filter by the eye tracker\'s (120Hz) frame rate instead of the game\'s (~90Hz).')
    parser.add_argument('--ignore_users', nargs='+',
                        help='List of user IDs to ignore. (User21 has a weird frame-rate for the second task.)')
    parser.add_argument('--task_names_train', nargs='+',
                        help='Name of the specific task(s) chosen for training.')
    parser.add_argument('--downsampling_rate', default=1, type=int,
                        help='Rate by which data points will be down-sampled from the actual rate of 90Hz or 120Hz.')
    parser.add_argument('--allow_blinks_train', action='store_true',
                        help='Flag to leave alone portions of training data in which the user blinked.')
    parser.add_argument('--handle_blinks_test', default='repeat_last', choices=('noop', 'repeat_last'),
                        help='Method to handle blinks during evaluation.')
    # Model configuration.
    parser.add_argument('--hidden_sizes', nargs='*', type=int,
                        help='Sizes of the hidden layers.')
    # Optimization configuration.
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Size of data batches during training for which the network will be optimized.')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Optimizer learning rate.')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of passes through the training set.')
    parser.add_argument('--early_stopping_patience', '--patience', default=5, type=int,
                        help='Number of epochs to wait for improvement on the val set before stopping early.')
    kwargs = vars(parser.parse_args())
    train(**kwargs)


def train(
        model_type: str,
        run_name: str = None,
        test_ratio: float = 0.25,
        val_ratio: float = 0.1667,
        seed: Optional[int] = None,
        use_eye_tracker_frames: bool = False,
        ignore_users: Optional[Union[List, Set]] = None,
        task_names_train: Optional[List] = None,
        downsampling_rate: int = 1,
        allow_blinks_train: bool = False,
        handle_blinks_test: Optional[str] = 'repeat_last',
        hidden_sizes: Optional[List[int]] = None,
        batch_size: int = 32,
        lr: float = 0.001,
        epochs: int = 100,
        early_stopping_patience: int = 5,
):
    """
    Can be run from a notebook or another script, if desired.
    """
    if ignore_users is None:
        ignore_users = set()
    if hidden_sizes is None:
        hidden_sizes = list()

    # Create model.
    if model_type == 'mlp':
        model = gaze_modeling.GazeMLP(hidden_sizes=hidden_sizes)
    elif model_type == 'lstm':
        model = gaze_modeling.GazeLSTM(hidden_sizes=hidden_sizes)
    elif model_type == 'vector':
        model = gaze_modeling.GazeVectorBaseline()
    elif model_type == 'vector-p':
        model = gaze_modeling.GazeVectorParameterized()
    else:
        raise ValueError(f'Unknown `model_type`: {model_type}')

    stamp = str(int(time.time())) + '_' + model_type
    if run_name is not None:
        stamp += '_' + run_name
    if seed is not None:
        stamp += '_s' + str(seed)
    path_stamp = os.path.join('output', 'runs', stamp)
    os.makedirs(path_stamp, exist_ok=True)
    print(f'Saving to: {path_stamp}')

    # Seed random number generator. Used for splitting/shuffling data set.
    rng = np.random.default_rng(seed=seed)
    print(f'Using seed: {seed}')

    # Load meta data.
    users, tasks, user_task_paths = data_utils.get_user_task_paths(use_eye_tracker_frames=use_eye_tracker_frames,
                                                                   ignore_users=ignore_users)
    n_users = len(users)
    print(f'Total users: {n_users}')
    if task_names_train is None:
        task_names_train = tasks
    print(f'Training tasks: {task_names_train}')

    # Shuffle and split data sets by user, then collect file paths.
    perm = rng.permutation(n_users)  # Shuffle indices.
    n_users_test = int(n_users * test_ratio)
    n_users_val = int((n_users - n_users_test) * val_ratio)
    users_val = [users[i] for i in perm[:n_users_val]]
    users_train = [users[i] for i in perm[n_users_val:n_users - n_users_test]]
    users_test = [users[i] for i in perm[n_users - n_users_test:]]
    print(f'Training with users ({len(users_train)}): {users_train}')
    print(f'Validating with users ({len(users_val)}): {users_val}')
    print(f'Evaluating with users ({len(users_test)}): {users_test}')

    paths_train, paths_val = list(), list()
    for task in task_names_train:
        for user in users_train:
            paths_train.extend(user_task_paths[user][task])
        for user in users_val:
            paths_val.extend(user_task_paths[user][task])

    # Read training files, create data set.
    kwargs_train = {
        'downsampling_rate': downsampling_rate,
        'allow_blinks': allow_blinks_train,
    }
    X_train, Y_train = data_utils.load_X_Y(paths_train, **kwargs_train)
    X_val, Y_val = data_utils.load_X_Y(paths_val, **kwargs_train)
    Y_train = to_pitch_yaw(X_train, Y_train)
    Y_val = to_pitch_yaw(X_val, Y_val)
    print(f'X_train.shape: {X_train.shape}; Y_train.shape: {Y_train.shape}')
    print(f'X_val.shape: {X_val.shape}; Y_val.shape: {Y_val.shape}')

    if torch.cuda.is_available():
        device = torch.device('cuda')  # Use the default GPU device
    else:
        device = torch.device('cpu')
    model = model.to(device)
    print(f'Using device: {device}')

    # Train.
    if type(model) is not gaze_modeling.GazeVectorBaseline:
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses_train = list()
        losses_val = list()
        loss_val_min = None
        path_val_best = os.path.join(path_stamp, 'val_best.pth')
        early_stopping_counter = 0
        for epoch in range(epochs):
            order = rng.permutation(X_train.shape[0])  # Shuffle.
            X_train_, Y_train_ = X_train[order], Y_train[order]
            _, loss_train = _inference(model, X_train_, Y_train_, criterion, batch_size, device, optimizer=optimizer)
            losses_train.append(loss_train)
            _, loss_val = _inference(model, X_val, Y_val, criterion, batch_size, device)
            losses_val.append(loss_val)
            print(f'Epoch: {epoch: >3d}; train loss: {loss_train:.8f}; val loss: {loss_val:.8f}')

            if loss_val_min is None or loss_val < loss_val_min:
                loss_val_min = loss_val
                early_stopping_counter = 0
                torch.save(model.state_dict(), path_val_best)
            else:
                early_stopping_counter += 1
                if early_stopping_counter == early_stopping_patience:
                    break

        # Plot training loss.
        plt.plot(list(range(len(losses_train))), losses_train, label='Train')
        plt.plot(list(range(len(losses_val))), losses_val, label='Val')
        plt.title('Training Loss - {} ({})'.format(
            model_type.upper(),
            'All' if len(task_names_train) == 4 else ', '.join(task_names_train)
        ))
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(os.path.join(path_stamp, 'loss.png'), bbox_inches='tight')
        plt.close()

    del X_train
    del Y_train
    del X_val
    del Y_val

    # Create visualizations.
    print('Creating visualizations...')
    points = list()
    x_step, y_step = 0.01, 0.01
    for y in range(-40, 21):
        for x in range(-25, 26):
            points.append((x * x_step, y * y_step))
    point_to_theta = vector_field.predict_thetas(model, points, [0., 0., 1.], rng, sample=32, device=device)
    vector_field.hist_thetas(point_to_theta, path=os.path.join(path_stamp, 'theta_hist.png'))
    vector_field.plot_vector_field(
        point_to_theta,
        title='Function Estimation - {} ({})'.format(
            model_type.upper(),
            'All' if len(task_names_train) == 4 else ', '.join(task_names_train)
        ),
        path=os.path.join(path_stamp, 'theta_vector_field.png'))

    # Log parameters.
    path_log = os.path.join(path_stamp, 'log.txt')
    with open(path_log, 'w') as fd:
        fd.write(f'stamp: {stamp}\n')
        fd.write('\n')
        fd.write('============\nPARAMETERS\n============\n')
        fd.write('\n')
        fd.write(f'model_type: {model_type}\n')
        fd.write(f'test_ratio: {test_ratio}\n')
        fd.write(f'val_ratio: {val_ratio}\n')
        fd.write(f'seed: {seed}\n')
        fd.write('\n')
        fd.write(f'use_eye_tracker_frames: {use_eye_tracker_frames}\n')
        fd.write(f'ignore_users: {ignore_users}\n')
        fd.write(f'task_names_train: {task_names_train}\n')
        fd.write(f'downsampling_rate: {downsampling_rate}\n')
        fd.write(f'allow_blinks_train: {allow_blinks_train}\n')
        fd.write(f'handle_blinks_test: {handle_blinks_test}\n')
        fd.write('\n')
        fd.write(f'hidden_sizes: {hidden_sizes}\n')
        fd.write('\n')
        fd.write(f'batch_size: {batch_size}\n')
        fd.write(f'learning_rate: {lr}\n')
        fd.write(f'epochs: {epochs}\n')
        fd.write(f'early_stopping_patience: {early_stopping_patience}\n')
        fd.write('\n')
        fd.write('============\nRESULTS\n============\n')
        fd.write('\n')
        fd.write('train users ({:d}): {}\n'.format(len(users_train), ','.join(users_train)))
        fd.write('validation users ({:d}): {}\n'.format(len(users_val), ','.join(users_val)))
        fd.write('test users ({:d}): {}\n'.format(len(users_test), ','.join(users_test)))

    # Quit here if training only.
    if len(users_test) == 0:
        print('No test users! Skipping evaluation.')
        return list(), list()

    # Evaluate.
    losses_test = list()
    print(f'Evaluating...')
    for task in tasks:
        paths_test = list()
        for user in users_test:
            paths_test.extend(user_task_paths[user][task])
        X_test, Y_test = data_utils.load_X_Y(paths_test,
                                             downsampling_rate=downsampling_rate,
                                             allow_blinks=True)  # Assume presence of blinks during testing.
        Y_test = to_pitch_yaw(X_test, Y_test)
        criterion_test = torch.nn.MSELoss()
        Y_test_hat, loss_test = _inference(model, X_test, Y_test, criterion_test, batch_size, device,
                                           handle_blinks=handle_blinks_test)
        losses_test.append(loss_test)
        print(f'Loss on `{task}`: {loss_test:.8f}')
    loss_test_mean = np.mean(losses_test)
    print(f'Mean loss: {loss_test_mean:.8f}')

    # Log results.
    with open(path_log, 'a') as fd:
        if len(users_test) > 0:
            fd.write('\n')
            width_task = max(len(task) for task in tasks)
            for task, loss_test in zip(tasks, losses_test):
                fd.write('test loss for task {:>{w}}: {:.8f}\n'.format(task, loss_test, w=width_task))
            fd.write(f'test loss mean: {loss_test_mean:.8f}\n')

    return losses_test + [loss_test_mean], tasks + ['All']


def to_pitch_yaw(X, Y):
    pitch_yaw = np.zeros((Y.shape[0], Y.shape[1], 2), dtype=Y.dtype)  # (n, s, 2)
    for i, (v, v_next) in enumerate(zip(X[:, :, -3:], Y)):
        pitch_yaw[i][:, 0] = spatial.to_pitch(v, v_next)
        pitch_yaw[i][:, 1] = spatial.to_yaw(v, v_next)
    return pitch_yaw


def _inference(model, X, Y, criterion, batch_size, device, handle_blinks=None, optimizer=None):
    if handle_blinks is not None:
        batch_size = 1  # Shouldn't let the model infer a batch containing blinks.

    n, seq_len, dims_out = Y.shape

    Y_hat = list()  # [(b_i, s, o)]
    batch_losses = list()
    batch_sizes = list()
    i = 0
    while i < n:
        X_batch = X[i:i + batch_size]  # (b, s, p)
        Y_batch = Y[i:i + batch_size]  # (b, s, o)
        b = X_batch.shape[0]

        Y_batch_hat = list()  # [(b, o)]
        losses = list()
        h0, c0 = None, None
        for t in range(seq_len):
            if optimizer is not None:
                optimizer.zero_grad()

            x_t = X_batch[:, t]  # (b, p)
            x_t = torch.tensor(x_t).to(device)

            if handle_blinks is not None and (x_t[0, 2] == 0.0 or x_t[0, 5] == 0.0):
                if handle_blinks == 'noop' or (handle_blinks == 'repeat_last' and len(Y_batch_hat) == 0):
                    y_hat_t = torch.zeros((batch_size, dims_out), dtype=x_t.dtype).to(device)
                elif handle_blinks == 'repeat_last':
                    y_hat_t = torch.tensor(Y_batch_hat[-1]).to(device)
                else:
                    raise ValueError(f'Unknown value for `handle_blinks`: {handle_blinks}')
            else:
                kwargs = dict()
                if h0 is not None and c0 is not None:
                    kwargs = {'h0': h0, 'c0': c0}
                y_hat_t = model(x_t, **kwargs)  # (b, o)
                if isinstance(y_hat_t, tuple):
                    y_hat_t, hn, cn = y_hat_t  # (b, o), (1, b, z), (1, b, z)
                    h0, c0 = hn.detach(), cn.detach()  # Use final hidden states as inputs to the next sequence.

            y_t = torch.tensor(Y_batch[:, t]).to(device)  # (b, o)
            loss = criterion(y_hat_t, y_t)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            Y_batch_hat.append(y_hat_t.detach().cpu().numpy())
            losses.append(loss.detach().cpu().numpy())

        Y_batch_hat = [np.expand_dims(y_hat_t, axis=1) for y_hat_t in Y_batch_hat]  # [(b, 1, o)]
        Y_hat.append(np.stack(Y_batch_hat, axis=1))  # (b, s, o)
        batch_losses.append(np.mean(losses))
        batch_sizes.append(b)

        i += batch_size

    Y_hat = np.concatenate(Y_hat, axis=0)  # (n, s, o)
    return Y_hat, np.average(batch_losses, weights=batch_sizes)


if __name__ == '__main__':
    main()
