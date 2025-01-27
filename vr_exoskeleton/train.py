import argparse
import os
import time

import numpy as np
import torch

from vr_exoskeleton import data_utils, gaze_modeling


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
    # Data set.
    parser.add_argument('--use_eye_tracker_frames', action='store_true',
                        help='Flag to filter by the eye tracker\'s (120Hz) frame rate instead of the game\'s (~90Hz).')
    parser.add_argument('--ignore_users', nargs='+',
                        help='List of user IDs to ignore. (User21 has a weird frame-rate for the second task.)')
    parser.add_argument('--task_names', nargs='+',
                        help='Name of the specific task(s) chosen.')
    parser.add_argument('--downsampling_rate', default=1, type=int,
                        help='Rate by which data points will be down-sampled from the actual rate of 90Hz or 120Hz.')
    parser.add_argument('--interpolate_blinks_train', action='store_true',
                        help='Flag to interpolate portions of training data in which the user blinked.')
    parser.add_argument('--handle_blinks_test', default='noop', choices=('noop', 'repeat_last'),
                        help='Method to handle blinks during evaluation.')
    parser.add_argument('--predict_relative_head', action='store_true',
                        help='Flag to predict relative head directions instead of absolute ones.')
    # Model configuration.
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
        use_eye_tracker_frames=False,
        ignore_users=None,
        task_names=None,
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
        model = gaze_modeling.GazeMLP(
            hidden_sizes=hidden_sizes,
        )
    elif model_type == 'lstm':
        model = gaze_modeling.GazeLSTM(
            hidden_sizes=hidden_sizes,
        )
    else:
        raise ValueError(f'Unknown `model_type`: {model_type}')

    # TODO: Add logging.
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
    users, tasks, user_task_paths = data_utils.get_user_task_paths(use_eye_tracker_frames=use_eye_tracker_frames,
                                                                   ignore_users=ignore_users)
    print(f'Total users: {len(users)}')
    if task_names is None:
        task_names = tasks
    print(f'Tasks: {task_names}')

    # Shuffle and split users.
    perm = rng.permutation(len(users))  # Shuffle indices.
    n_users_test = int(len(users) * test_ratio)
    if n_users_test <= 0:
        raise ValueError(f'Parameter `test_ratio` is too small: {test_ratio}')
    if n_users_test >= len(users):
        raise ValueError(f'Parameter `test_ratio` is too large: {test_ratio}')
    n_users_val = int((len(users) - n_users_test) * val_ratio)
    users_val = [users[i] for i in perm[:n_users_val]]
    users_train = [users[i] for i in perm[n_users_val:-n_users_test]]
    users_test = [users[i] for i in perm[-n_users_test:]]
    print(f'Training with users ({len(users_train)}): {users_train}')
    print(f'Validating with users ({len(users_val)}): {users_val}')
    print(f'Evaluating with users ({len(users_test)}): {users_test}')

    # Read training files, create data set.
    kwargs_load = {
        'downsampling_rate': downsampling_rate,
        'interpolate_blinks': interpolate_blinks_train,
    }
    X_train, Y_train = data_utils.load_X_Y(users_train, task_names, user_task_paths, **kwargs_load)
    X_val, Y_val = data_utils.load_X_Y(users_val, task_names, user_task_paths, **kwargs_load)
    if predict_relative_head:
        Y_train -= X_train[:, :, 6:9]
        Y_val -= X_val[:, :, 6:9]
    print(f'X_train.shape: {X_train.shape}; Y_train.shape: {Y_train.shape}')

    if torch.cuda.is_available():
        device = torch.device('cuda')  # Use the default GPU device
    else:
        device = torch.device('cpu')
    model = model.to(device)
    print(f'Using device: {device}')

    # Train.
    criterion = torch.nn.MSELoss()
    # criterion = gaze_modeling.AngleLoss()
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
    X_test, Y_test = data_utils.load_X_Y(users_test, task_names, user_task_paths,
                                         interpolate_blinks=False)  # Assume presence of blinks during testing.
    if predict_relative_head:
        Y_test -= X_test[:, :, 6:9]
    criterion_test = torch.nn.MSELoss()
    # criterion_test = gaze_modeling.AngleLoss()
    Y_test_hat, loss_test = _inference(model, X_test, Y_test, criterion_test, batch_size, device,
                                       predict_relative_head=predict_relative_head, handle_blinks=handle_blinks_test)
    print(f'Loss on held-out test set: {loss_test:.8f}')
    np.save(os.path.join(path_stamp, 'test_users.npy'), np.array(users_test))
    np.save(os.path.join(path_stamp, 'test_input.npy'), X_test)
    np.save(os.path.join(path_stamp, 'test_head_predicted.npy'), Y_test_hat)
    np.save(os.path.join(path_stamp, 'test_head_actual.npy'), Y_test)


def _inference(model, X, Y, criterion, batch_size, device, predict_relative_head=False, handle_blinks=None, optimizer=None):
    if handle_blinks is not None:
        batch_size = 1  # Shouldn't let the model infer a batch containing blinks.

    n, seq_len = X.shape[0], X.shape[1]

    Y_hat = list()  # [(b_i, s, 3)]
    batch_losses = list()
    batch_sizes = list()
    i = 0
    while i < n:
        X_batch = X[i:i + batch_size]  # (b, s, 9)
        Y_batch = Y[i:i + batch_size]  # (b, s, 3)
        b = X_batch.shape[0]

        Y_batch_hat = list()  # [(b, 3)]
        losses = list()
        h0, c0 = None, None
        for t in range(seq_len):
            if optimizer is not None:
                optimizer.zero_grad()

            x_t = X_batch[:, t]  # (b, 9)
            x_t = torch.tensor(x_t).to(device)

            if handle_blinks is not None and (x_t[0, 2] == 0.0 or x_t[0, 5] == 0.0):
                if handle_blinks == 'noop':
                    if predict_relative_head:
                        y_hat_t = torch.zeros((batch_size, 3), dtype=x_t.dtype).to(device)
                    else:
                        y_hat_t = x_t.detach().clone()
                elif handle_blinks == 'repeat_last':
                    if len(Y_batch_hat) > 0:
                        y_hat_t = torch.tensor(Y_batch_hat[-1])
                    elif predict_relative_head:  # Fallback to `noop` rules when no previous prediction exists.
                        y_hat_t = torch.zeros((batch_size, 3), dtype=x_t.dtype).to(device)
                    else:
                        y_hat_t = x_t.detach().clone()
                else:
                    raise ValueError(f'Unknown value for `handle_blinks`: {handle_blinks}')
            else:
                kwargs = dict()
                if h0 is not None and c0 is not None:
                    kwargs = {'h0': h0, 'c0': c0}
                y_hat_t = model(x_t, **kwargs)  # (b, 3)
                if isinstance(y_hat_t, tuple):
                    y_hat_t, hn, cn = y_hat_t  # (b, 3), (1, b, z), (1, b, z)
                    h0, c0 = hn.detach(), cn.detach()  # Use final hidden states as inputs to the next sequence.

            y_t = torch.tensor(Y_batch[:, t]).to(device)  # (b, 3)
            loss = criterion(y_hat_t, y_t)
            losses.append(loss.detach().cpu().numpy())

            Y_batch_hat.append(y_hat_t.detach().cpu().numpy())  # (b, 3)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

        Y_batch_hat = [np.expand_dims(y_hat_t, axis=1) for y_hat_t in Y_batch_hat]  # [(b, 1, 3)]
        Y_hat.append(np.stack(Y_batch_hat, axis=1))  # (b, s, 3)
        batch_losses.append(np.mean(losses))
        batch_sizes.append(b)

        i += batch_size

    Y_hat = np.concatenate(Y_hat, axis=0)  # (n, s, 3)
    return Y_hat, np.average(batch_losses, weights=batch_sizes)


if __name__ == '__main__':
    main()
