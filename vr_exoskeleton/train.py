import argparse
import os
import time

import numpy as np
import torch

from vr_exoskeleton import data_utils, gaze_modeling, spatial, vector_field


def main():
    """
    Only starts from the command line when this script is run using `python vr_exoskeleton/train.py ...`.
    """
    parser = argparse.ArgumentParser(
        description='Train a neural network on eye gaze and neck data.'
    )
    parser.add_argument('model_type', choices=('mlp', 'lstm'),
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
    parser.add_argument('--allow_mixed_splits', action='store_true',
                        help='Flag to allow data from the same users across train/validation/test sets.')
    # Data set.
    parser.add_argument('--base_users_folder', default='Users', type=str,
                        help='Folder name within `data/` from which to pull eye and head data.')
    parser.add_argument('--use_eye_tracker_frames', action='store_true',
                        help='Flag to filter by the eye tracker\'s (120Hz) frame rate instead of the game\'s (~90Hz).')
    parser.add_argument('--ignore_users', nargs='+',
                        help='List of user IDs to ignore. (User21 has a weird frame-rate for the second task.)')
    parser.add_argument('--task_names_train', nargs='+',
                        help='Name of the specific task(s) chosen for training.')
    parser.add_argument('--task_names_test', nargs='+',
                        help='Name of the specific task(s) chosen for evaluation.')
    parser.add_argument('--downsampling_rate', default=1, type=int,
                        help='Rate by which data points will be down-sampled from the actual rate of 90Hz or 120Hz.')
    parser.add_argument('--interpolate_blinks_train', action='store_true',
                        help='Flag to interpolate portions of training data in which the user blinked.')
    parser.add_argument('--handle_blinks_test', default='noop', choices=('noop', 'repeat_last'),
                        help='Method to handle blinks during evaluation.')
    # Model configuration.
    parser.add_argument('--predict_relative_head', action='store_true',
                        help='Flag to predict relative head directions/rotations instead of absolute ones.')
    parser.add_argument('--hidden_sizes', nargs='*', type=int,
                        help='Sizes of the hidden layers.')
    # Optimization configuration.
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Size of data batches during training for which the network will be optimized.')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Optimizer learning rate.')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of passes through the training set.')
    parser.add_argument('--early_stopping_patience', '--patience', default=5, type=int,
                        help='Number of epochs to wait for improvement on the val set before stopping early.')
    kwargs = vars(parser.parse_args())
    train(**kwargs)


def train(
        model_type,
        run_name=None,
        test_ratio=0.25,
        val_ratio=0.1667,
        seed=None,
        allow_mixed_splits=False,
        base_users_folder='Users',
        use_eye_tracker_frames=False,
        ignore_users=None,
        task_names_train=None,
        task_names_test=None,
        downsampling_rate=1,
        interpolate_blinks_train=False,
        handle_blinks_test='noop',
        predict_relative_head=False,
        hidden_sizes=None,
        batch_size=32,
        learning_rate=0.001,
        epochs=100,
        early_stopping_patience=5,
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
        model = gaze_modeling.GazeMLP(hidden_sizes=hidden_sizes,
                                      predict_relative_head=predict_relative_head)
    elif model_type == 'lstm':
        model = gaze_modeling.GazeLSTM(hidden_sizes=hidden_sizes,
                                       predict_relative_head=predict_relative_head)
    else:
        raise ValueError(f'Unknown `model_type`: {model_type}')

    stamp = str(int(time.time())) + '_' + model_type
    if run_name is not None:
        stamp += '_' + run_name
    if seed is not None:
        stamp += '_s' + str(seed)
    path_stamp = os.path.join('output', stamp)
    os.makedirs(path_stamp, exist_ok=True)
    print(f'Saving to: {path_stamp}')

    # Seed random number generator. Used for splitting/shuffling data set.
    rng = np.random.default_rng(seed=seed)
    print(f'Using seed: {seed}')

    # Load meta data.
    users, tasks, user_task_paths = data_utils.get_user_task_paths(base_users_folder=base_users_folder,
                                                                   use_eye_tracker_frames=use_eye_tracker_frames,
                                                                   ignore_users=ignore_users)
    print(f'Total users: {len(users)}')
    if task_names_train is None:
        task_names_train = tasks
    if task_names_test is None:
        task_names_test = tasks
    print(f'Training tasks: {task_names_train}')
    print(f'Evaluation tasks: {task_names_test}')

    # Split data set.
    if allow_mixed_splits:
        # Aggregate all user data paths, then shuffle and split.
        paths_union = list()
        paths_exclusive_train = list()
        paths_exclusive_test = list()
        for user in users:
            for task in tasks:
                paths = user_task_paths[user][task]
                if task in task_names_train:
                    if task in task_names_test:
                        paths_union.extend(paths)
                    else:
                        paths_exclusive_train.extend(paths)
                elif task in task_names_test:
                    paths_exclusive_test.extend(paths)

        perm_union = rng.permutation(len(paths_union))  # Shuffle.
        n_paths_union_test = int(len(paths_union) * test_ratio)
        n_paths_union_val = int((len(paths_union) - n_paths_union_test) * val_ratio)
        paths_val = [paths_union[i] for i in perm_union[:n_paths_union_val]]
        paths_train = [paths_union[i] for i in perm_union[n_paths_union_val:-n_paths_union_test]]
        paths_test = [paths_union[i] for i in perm_union[-n_paths_union_test:]]

        perm_train = rng.permutation(len(paths_exclusive_train))
        n_paths_train_val = int(len(paths_exclusive_train) * val_ratio)
        paths_val += [paths_exclusive_train[i] for i in perm_train[:n_paths_train_val]]
        paths_train += [paths_exclusive_train[i] for i in perm_train[n_paths_train_val:]]
        paths_test += paths_exclusive_test
        print('Number of train/validation/test trials: {:d}/{:d}/{:d}'
              .format(len(paths_train), len(paths_val), len(paths_test)))
    else:
        # Shuffle and split data sets by user, then collect file paths.
        perm = rng.permutation(len(users))  # Shuffle indices.
        n_users_test = int(len(users) * test_ratio)
        n_users_val = int((len(users) - n_users_test) * val_ratio)
        users_val = [users[i] for i in perm[:n_users_val]]
        users_train = [users[i] for i in perm[n_users_val:-n_users_test]]
        users_test = [users[i] for i in perm[-n_users_test:]]
        print(f'Training with users ({len(users_train)}): {users_train}')
        print(f'Validating with users ({len(users_val)}): {users_val}')
        print(f'Evaluating with users ({len(users_test)}): {users_test}')
        np.save(os.path.join(path_stamp, 'test_users.npy'), np.array(users_test))

        split_paths = [list(), list(), list()]
        for i, users_split in enumerate([users_train, users_val, users_test]):
            for user in users_split:
                task_names = task_names_test if i == 2 else task_names_train
                for task in task_names:
                    split_paths[i].extend(user_task_paths[user][task])
        paths_train, paths_val, paths_test = split_paths

    # Read training files, create data set.
    kwargs_train = {
        'downsampling_rate': downsampling_rate,
        'interpolate_blinks': interpolate_blinks_train,
    }
    X_train, Y_train = data_utils.load_X_Y(paths_train, **kwargs_train)
    X_val, Y_val = data_utils.load_X_Y(paths_val, **kwargs_train)
    if predict_relative_head:
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
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses_train = list()
    losses_val = list()
    loss_val_min = None
    path_val_best = os.path.join(path_stamp, 'val_best.pth')
    early_stopping_counter = 0
    for epoch in range(epochs):
        order = rng.permutation(X_train.shape[0])  # Shuffle.
        X_train_, Y_train_ = X_train[order], Y_train[order]
        _, loss_train = _inference(model, X_train_, Y_train_, criterion, batch_size, device,
                                   predict_relative_head=predict_relative_head, optimizer=optimizer)
        losses_train.append(loss_train)
        _, loss_val = _inference(model, X_val, Y_val, criterion, batch_size, device,
                                 predict_relative_head=predict_relative_head)
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

    del X_train
    del Y_train
    del X_val
    del Y_val

    # Evaluate.
    print('Evaluating...')
    X_test, Y_test = data_utils.load_X_Y(paths_test,
                                         downsampling_rate=downsampling_rate,
                                         interpolate_blinks=False)  # Assume presence of blinks during testing.
    if predict_relative_head:
        Y_test = to_pitch_yaw(X_test, Y_test)
    criterion_test = torch.nn.MSELoss()
    Y_test_hat, loss_test = _inference(model, X_test, Y_test, criterion_test, batch_size, device,
                                       predict_relative_head=predict_relative_head,
                                       handle_blinks=handle_blinks_test)
    print(f'Loss on held-out test set: {loss_test:.8f}')
    np.save(os.path.join(path_stamp, 'test_input.npy'), X_test)
    np.save(os.path.join(path_stamp, 'test_head_predicted.npy'), Y_test_hat)
    np.save(os.path.join(path_stamp, 'test_head_actual.npy'), Y_test)

    # Log parameters and results.
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
        fd.write(f'allow_mixed_splits: {allow_mixed_splits}\n')
        fd.write('\n')
        fd.write(f'base_users_folder: {base_users_folder}\n')
        fd.write(f'use_eye_tracker_frames: {use_eye_tracker_frames}\n')
        fd.write(f'ignore_users: {ignore_users}\n')
        fd.write(f'task_names_train: {task_names_train}\n')
        fd.write(f'task_names_test: {task_names_test}\n')
        fd.write(f'downsampling_rate: {downsampling_rate}\n')
        fd.write(f'interpolate_blinks_train: {interpolate_blinks_train}\n')
        fd.write(f'handle_blinks_test: {handle_blinks_test}\n')
        fd.write('\n')
        fd.write(f'predict_relative_head: {predict_relative_head}\n')
        fd.write(f'hidden_sizes: {hidden_sizes}\n')
        fd.write('\n')
        fd.write(f'batch_size: {batch_size}\n')
        fd.write(f'learning_rate: {learning_rate}\n')
        fd.write(f'epochs: {epochs}\n')
        fd.write(f'early_stopping_patience: {early_stopping_patience}\n')
        fd.write('\n')
        fd.write('============\nRESULTS\n============\n')
        fd.write('\n')
        fd.write(f'test loss: {loss_test:.8f}')

    # Create visualizations.
    print('Creating visualizations...')
    points = list()
    x_step, y_step = 0.01, 0.01
    for y in range(-40, 21):
        for x in range(-25, 26):
            points.append((x * x_step, y * y_step))
    point_to_theta = vector_field.predict_thetas(model, points, [0., 0., 1.], rng, device=device)
    vector_field.hist_thetas(point_to_theta, path_stamp=path_stamp)
    vector_field.plot_vector_field(point_to_theta, title='Function Estimation', path_stamp=path_stamp)


def to_pitch_yaw(X, Y):
    pitch_yaw = np.zeros((Y.shape[0], Y.shape[1], 2), dtype=Y.dtype)  # (n, s, 2)
    for i, (v, v_next) in enumerate(zip(X[:, :, -3:], Y)):
        pitch_yaw[i][:, 0] = spatial.to_pitch(v, v_next)
        pitch_yaw[i][:, 1] = spatial.to_yaw(v, v_next)
    return pitch_yaw


def _inference(model, X, Y, criterion, batch_size, device,
               predict_relative_head=False, handle_blinks=None, optimizer=None):
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
                if handle_blinks == 'noop':
                    if predict_relative_head:
                        y_hat_t = torch.zeros((batch_size, dims_out), dtype=x_t.dtype).to(device)
                    else:
                        y_hat_t = x_t.detach().clone()
                elif handle_blinks == 'repeat_last':
                    if len(Y_batch_hat) > 0:
                        y_hat_t = torch.tensor(Y_batch_hat[-1]).to(device)
                    elif predict_relative_head:  # Fallback to `noop` rules when no previous prediction exists.
                        y_hat_t = torch.zeros((batch_size, dims_out), dtype=x_t.dtype).to(device)
                    else:
                        y_hat_t = x_t.detach().clone()
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
            losses.append(loss.detach().cpu().numpy())

            Y_batch_hat.append(y_hat_t.detach().cpu().numpy())  # (b, h)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

        Y_batch_hat = [np.expand_dims(y_hat_t, axis=1) for y_hat_t in Y_batch_hat]  # [(b, 1, o)]
        Y_hat.append(np.stack(Y_batch_hat, axis=1))  # (b, s, o)
        batch_losses.append(np.mean(losses))
        batch_sizes.append(b)

        i += batch_size

    Y_hat = np.concatenate(Y_hat, axis=0)  # (n, s, o)
    return Y_hat, np.average(batch_losses, weights=batch_sizes)


if __name__ == '__main__':
    main()
