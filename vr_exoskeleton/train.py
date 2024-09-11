import argparse
import os
import time
from collections import defaultdict

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
    parser.add_argument('--test_ratio', default=0.4, type=float,
                        help='Ratio of users for whose data will be set aside for evaluation.')
    parser.add_argument('--val_ratio', default=0.25, type=float,
                        help='Ratio of users within training set for whose data will be set aside for validation.')
    parser.add_argument('--seed', type=int,
                        help='Seeds the random number generator that shuffles and splits the data set.')
    # Data set.
    parser.add_argument('--downsampling_rate', default=1, type=int,
                        help='Rate by which data points will be down-sampled from the actual rate of 120Hz.')
    parser.add_argument('--drop_blinks', action='store_true',
                        help='Flag to exclude portions of training and validation data in which the user blinked.')
    parser.add_argument('--drop_gaze_z', action='store_true',
                        help='Flag to drop the z-dimension of the left and right gaze vectors.')
    parser.add_argument('--task_names', nargs='+',
                        help='Name of the specific task(s) chosen.')
    # Model configuration.
    parser.add_argument('--mlp_window_size', default=3, type=int,
                        help='MLP only. Number of time steps in the past used to predict the next neck movement.')
    parser.add_argument('--hidden_sizes', nargs='*', type=int,
                        help='Sizes of the hidden layers.')
    parser.add_argument('--seq_len_max', default=1024, type=int,
                        help='LSTM only. Maximum sequence length.')
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
        test_ratio=0.4,
        val_ratio=0.25,
        seed=None,
        downsampling_rate=1,
        drop_blinks=False,
        drop_gaze_z=False,
        task_names=None,
        mlp_window_size=3,
        hidden_sizes=None,
        seq_len_max=1024,
        batch_size=64,
        learning_rate=0.001,
        epochs=100,
        early_stopping_patience=5,
):
    """
    Can be run from a notebook or another script, if desired.
    """
    if hidden_sizes is None:
        hidden_sizes = list()

    # Create model.
    instance_size = 7 if drop_gaze_z else 9
    if model_type == 'mlp':
        model = gaze_modeling.GazeMLP(
            instance_size,
            window_size=mlp_window_size,
            hidden_sizes=hidden_sizes,
        )
        window_size = mlp_window_size
    elif model_type == 'lstm':
        model = gaze_modeling.GazeLSTM(
            instance_size,
            hidden_sizes=hidden_sizes,
        )
        window_size = 1
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
    users, tasks, user_task_paths = data_utils.get_user_task_paths()
    print(f'Total users: {len(users)}')
    if task_names is None:
        task_names = tasks
    print(f'Tasks: {task_names}')

    # Shuffle and split users.
    perm = rng.permutation(len(users))  # Shuffle indices.
    n_users_test = max(0, min(len(users), int(len(users) * test_ratio)))  # Bound in [0, n].
    if n_users_test == 0:
        raise ValueError(f'Parameter `test_ratio` is too small: {test_ratio}')
    if n_users_test == len(users):
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
        'window_size': window_size,
        'downsampling_rate': downsampling_rate,
        'drop_blinks': drop_blinks,
        'drop_gaze_z': drop_gaze_z,
    }
    sequences_X_train, sequences_Y_train =\
        data_utils.load_sequences_X_Y(users_train, task_names, user_task_paths, **kwargs_load)
    sequences_X_val, sequences_Y_val =\
        data_utils.load_sequences_X_Y(users_val, task_names, user_task_paths, **kwargs_load)
    if model_type in {'lstm'}:
        X_train, Y_train = to_time_series_dict(sequences_X_train, sequences_Y_train, seq_len_max)
        X_val, Y_val = to_time_series_dict(sequences_X_val, sequences_Y_val, seq_len_max)
        print('X_train sequence lengths: {}'.format({length: sequences.shape[1] for length, sequences in X_train.items()}))
    else:
        X_train, Y_train = np.concatenate(sequences_X_train, axis=0), np.concatenate(sequences_Y_train, axis=0)
        X_val, Y_val = np.concatenate(sequences_X_val, axis=0), np.concatenate(sequences_Y_val, axis=0)
        print(f'X_train.shape: {X_train.shape}; Y_train.shape: {Y_train.shape}')

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
        if model_type in {'lstm'}:
            loss_train = train_one_epoch_recurrent(model, X_train, Y_train, optimizer, criterion, batch_size, device, rng)
            losses_train.append(loss_train)
            _, loss_val = evaluate_recurrent(model, X_val, Y_val, criterion, batch_size, device)
        else:
            loss_train = train_one_epoch(model, X_train, Y_train, optimizer, criterion, batch_size, device, rng)
            losses_train.append(loss_train)
            _, loss_val = evaluate(model, X_val, Y_val, criterion, device)
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

    # Evaluate.
    del X_train
    del Y_train
    del X_val
    del Y_val
    sequences_X_test, sequences_Y_test = data_utils.load_sequences_X_Y(
        users_test,
        task_names,
        user_task_paths,
        window_size=window_size,
        downsampling_rate=downsampling_rate,
        drop_blinks=False,  # Assume that is not possible to ignore blinks during testing.
        drop_gaze_z=drop_gaze_z,
    )
    if model_type in {'lstm'}:
        X_test, Y_test = to_time_series_dict(sequences_X_test, sequences_Y_test, seq_len_max)
        Y_test_hat, loss_test = evaluate_recurrent(model, X_test, Y_test, criterion, batch_size, device)
    else:
        X_test, Y_test = np.concatenate(sequences_X_test, axis=0), np.concatenate(sequences_Y_test, axis=0)
        Y_test_hat, loss_test = evaluate(model, X_test, Y_test, criterion, device)
    print(f'Loss on held-out test set: {loss_test:.8f}')
    np.save(os.path.join(path_stamp, 'test_users.npy'), np.array(users_test))
    np.save(os.path.join(path_stamp, 'test_input.npy'), X_test)
    np.save(os.path.join(path_stamp, 'test_head_predicted.npy'), Y_test_hat)
    np.save(os.path.join(path_stamp, 'test_head_actual.npy'), Y_test)


def to_time_series_dict(sequences_X, sequences_Y, seq_len_max):
    X, Y = defaultdict(list), defaultdict(list)
    for sequence_X, sequence_Y in zip(sequences_X, sequences_Y):
        if len(sequence_X) <= seq_len_max:
            X[len(sequence_X)].append(sequence_X)
            Y[len(sequence_Y)].append(sequence_Y)
        else:
            # Break up long sequences.
            i = 0
            while i < len(sequence_X):
                chunk_X, chunk_Y = sequence_X[i:i + seq_len_max], sequence_Y[i:i + seq_len_max]
                X[len(chunk_X)].append(chunk_X)
                Y[len(chunk_Y)].append(chunk_Y)
                i += seq_len_max

    # `transpose()` is to provide axes as:
    #  (sequence_length, batch_size, instance_size)
    # instead of:
    #  (batch_size, sequence_length, instance_size)
    # in order to comply with `LSTM(batch_first=False)`.
    X = {length: np.array(chunks).transpose(1, 0, 2)
         for length, chunks in X.items()}  # (length, N_{length}, instance_size) for each `length`.
    Y = {length: np.array(chunks).transpose(1, 0, 2)
         for length, chunks in Y.items()}  # (length, N_{length}, output_size=3) for each `length`.
    return X, Y


def train_one_epoch(model, X, Y, optimizer, criterion, batch_size, device, rng):
    n = X.shape[0]
    order = rng.permutation(n)  # Shuffle.
    i = 0
    batch_losses = list()
    batch_sizes = list()
    while i < n:
        optimizer.zero_grad()
        X_batch = torch.tensor(X[order[i:i + batch_size]]).to(device)
        Y_hat_batch = model(X_batch)
        Y_batch = torch.tensor(Y[order[i:i + batch_size]]).to(device)
        loss = criterion(Y_hat_batch, Y_batch)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.detach().cpu().numpy())
        batch_sizes.append(X_batch.shape[0])
        i += batch_size

    return np.average(batch_losses, weights=batch_sizes)


def train_one_epoch_recurrent(model, X, Y, optimizer, criterion, batch_size, device, rng):
    losses = list()
    sizes = list()
    for length in X.keys():
        batch_losses = list()
        batch_sizes = list()
        n = X[length].shape[1]
        order = rng.permutation(n)  # Shuffle.
        i = 0
        while i < n:
            optimizer.zero_grad()
            X_batch = torch.tensor(X[length][:, order[i:i + batch_size]]).to(device)
            Y_hat_batch, _, _ = model(X_batch)
            Y_batch = torch.tensor(Y[length][:, order[i:i + batch_size]]).to(device)
            loss = criterion(Y_hat_batch, Y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.detach().cpu().numpy())
            batch_sizes.append(X_batch.shape[1])
            i += batch_size
        losses.append(np.average(batch_losses, weights=batch_sizes))
        sizes.append(n)

    return np.average(losses, weights=sizes)


def evaluate(model, X, Y, criterion, device):
    with torch.no_grad():
        Y_hat = model(torch.tensor(X).to(device))
    loss = criterion(Y_hat, torch.tensor(Y).to(device))
    return Y_hat.detach().cpu().numpy(), loss.detach().cpu().numpy()


def evaluate_recurrent(model, X, Y, criterion, batch_size, device):
    Y_hat = dict()
    losses = list()
    sizes = list()
    for length in X.keys():
        batch_Y_hat = list()
        batch_losses = list()
        batch_sizes = list()
        n = X[length].shape[1]
        i = 0
        while i < n:
            X_batch = torch.tensor(X[length][:, i:i + batch_size]).to(device)
            with torch.no_grad():
                Y_hat_batch, _, _ = model(X_batch)
            Y_batch = torch.tensor(Y[length][:, i:i + batch_size]).to(device)
            loss = criterion(Y_hat_batch, Y_batch)
            batch_Y_hat.append(Y_hat_batch.detach().cpu().numpy())
            batch_losses.append(loss.detach().cpu().numpy())
            batch_sizes.append(X_batch.shape[1])
            i += batch_size
        Y_hat[length] = np.concatenate(batch_Y_hat, axis=1)
        losses.append(np.average(batch_losses, weights=batch_sizes))
        sizes.append(n)

    return Y_hat, np.average(losses, weights=sizes)


if __name__ == '__main__':
    main()
