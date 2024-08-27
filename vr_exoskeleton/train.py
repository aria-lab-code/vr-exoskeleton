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
    parser.add_argument('--window_size', default=3, type=int,
                        help='Number of time steps in the past used to predict the next neck movement.')
    parser.add_argument('--downsampling_rate', default=1, type=int,
                        help='Rate by which data points will be down-sampled from the actual rate of 120Hz.')
    parser.add_argument('--hidden_sizes', nargs='*', type=int,
                        help='Sizes of the hidden layers of the MLP. May be empty.')
    parser.add_argument('--keep_blinks', action='store_true',
                        help='Flag to leave in portions of training and validation data in which the user blinked.')
    parser.add_argument('--drop_gaze_z', action='store_true',
                        help='Flag to drop the z-dimension of the left and right gaze vectors.')
    parser.add_argument('--task_names', nargs='+',
                        help='Name of the specific task(s) chosen.')
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
        run_name=None,
        test_ratio=0.4,
        val_ratio=0.25,
        seed=None,
        window_size=3,
        downsampling_rate=1,
        hidden_sizes=None,
        keep_blinks=False,
        drop_gaze_z=False,
        task_names=None,
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

    # TODO: Add logging.
    stamp = str(int(time.time()))
    if run_name is not None:
        stamp += '_' + run_name
    path_stamp = os.path.join('output', stamp)
    os.makedirs(path_stamp, exist_ok=True)
    print(f'Saving to: {path_stamp}')

    # Seed random number generator. Used for splitting/shuffling data set.
    rng = np.random.default_rng(seed=seed)
    print(f'Using seed: {seed}')

    if torch.cuda.is_available():
        device = torch.device('cuda')  # Use the default GPU device
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

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
        'keep_blinks': keep_blinks,
        'drop_gaze_z': drop_gaze_z
    }
    X_train, Y_train = data_utils.load_X_Y(users_train, task_names, user_task_paths, **kwargs_load)
    X_val, Y_val = data_utils.load_X_Y(users_val, task_names, user_task_paths, **kwargs_load)
    print(f'X_train.shape: {X_train.shape}; Y_train.shape: {Y_train.shape}')

    # Create model.
    model = gaze_modeling.GazeMLP(
        window_size=window_size,
        hidden_sizes=hidden_sizes,
        drop_gaze_z=drop_gaze_z
    ).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train.
    losses_train = list()
    losses_val = list()
    loss_val_min = None
    path_val_best = os.path.join(path_stamp, 'val_best.pth')
    early_stopping_counter = 0
    for epoch in range(epochs):
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
    X_test, Y_test = data_utils.load_X_Y(
        users_test,
        task_names,
        user_task_paths,
        window_size=window_size,
        downsampling_rate=downsampling_rate,
        keep_blinks=True,  # Assume that is not possible to ignore blinks during testing.
        drop_gaze_z=drop_gaze_z
    )
    Y_test_hat, loss_test = evaluate(model, X_test, Y_test, criterion, device)
    print(f'Loss on held-out test set: {loss_test:.8f}')
    np.save(os.path.join(path_stamp, 'test_users.npy'), np.array(users_test))
    np.save(os.path.join(path_stamp, 'test_input.npy'), X_test)
    np.save(os.path.join(path_stamp, 'test_head_predicted.npy'), Y_test_hat)
    np.save(os.path.join(path_stamp, 'test_head_actual.npy'), Y_test)


def train_one_epoch(model, X, Y, optimizer, criterion, batch_size, device, rng):
    order = rng.permutation(X.shape[0])  # Shuffle.
    i = 0
    batch_losses = list()
    while i < X.shape[0]:
        optimizer.zero_grad()
        X_batch = torch.tensor(X[order[i:i + batch_size]]).to(device)
        Y_batch_hat = model(X_batch)
        Y_batch = torch.tensor(Y[order[i:i + batch_size]]).to(device)
        loss = criterion(Y_batch_hat, Y_batch)
        loss.backward()
        optimizer.step()
        batch_losses.append(np.mean(loss.detach().cpu().numpy()))
        i += batch_size
    return np.mean(batch_losses)


def evaluate(model, X, Y, criterion, device):
    with torch.no_grad():
        Y_hat = model(torch.tensor(X).to(device))
        loss = criterion(Y_hat, torch.tensor(Y).to(device))
    return Y_hat.detach().cpu().numpy(), np.mean(loss.detach().cpu().numpy())


if __name__ == '__main__':
    main()
