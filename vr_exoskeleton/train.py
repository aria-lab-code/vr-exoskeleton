import argparse
import time

import numpy as np
import torch

from vr_exoskeleton import data_utils, gaze_modeling


def main():
    # TODO: argparse
    test_ratio = 0.4
    val_ratio = 0.25
    seed = 4
    window_size = 3
    drop_blinks = True
    batch_size = 64
    learning_rate = 0.001
    epochs = 100
    early_stopping_patience = 10
    train(
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        seed=seed,
        window_size=window_size,
        drop_blinks=drop_blinks,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
    )


def train(
        test_ratio=0.4,
        val_ratio=0.25,
        seed=None,
        window_size=3,
        drop_blinks=True,
        batch_size=64,
        learning_rate=0.001,
        epochs=100,
        early_stopping_patience=10,
):
    stamp = int(time.time())

    # Seed random number generator. Used for splitting/shuffling data set.
    rng = np.random.default_rng(seed=seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')  # Use the default GPU device
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Load meta data.
    users, tasks, user_task_paths = data_utils.get_user_task_paths()
    print(f'Total users: {len(users)}')

    # Shuffle and split users.
    perm = rng.permutation(len(users))  # Shuffle indices.
    n_users_test = int(len(users) * test_ratio)
    if n_users_test == 0:
        raise ValueError(f'Parameter `test_ratio` is too small: {test_ratio}')
    if n_users_test == len(users):
        raise ValueError(f'Parameter `test_ratio` is too large: {test_ratio}')
    n_users_val = int(len(users) * (1.0 - test_ratio) * val_ratio)
    users_val = [users[i] for i in perm[:n_users_val]]
    users_train = [users[i] for i in perm[n_users_val:-n_users_test]]
    users_test = [users[i] for i in perm[-n_users_test:]]
    print(f'Training with users ({len(users_train)}): {users_train}')
    print(f'Validating with users ({len(users_val)}): {users_val}')
    print(f'Evaluating with users ({len(users_test)}): {users_test}')

    # Read training files, create data set.
    kwargs_load = {'window_size': window_size, 'drop_blinks': drop_blinks}
    X_train, Y_train = data_utils.load_X_Y(users_train, tasks, user_task_paths, **kwargs_load)
    X_val, Y_val = data_utils.load_X_Y(users_val, tasks, user_task_paths, **kwargs_load)
    print(f'X_train.shape: {X_train.shape}; Y_train.shape: {Y_train.shape}')

    # Create model.
    model = gaze_modeling.GazeMLP(window_size).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train.
    losses_train = list()
    losses_val = list()
    loss_val_min = None
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
            torch.save(model.state_dict(), f'best_val_{stamp:d}.pth')
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
        tasks,
        user_task_paths,
        window_size=window_size,
        drop_blinks=False  # Assume that this is not possible during testing.
    )
    Y_test_hat, loss_test = evaluate(model, X_test, Y_test, criterion, device)
    print(f'Loss on held-out test set: {loss_test:.8f}')
    np.save(f'eval_{stamp:d}_users.npy', np.array(users_test))
    np.save(f'eval_{stamp:d}_input.npy', X_test)
    np.save(f'eval_{stamp:d}_head_predicted.npy', Y_test_hat)
    np.save(f'eval_{stamp:d}_head_actual.npy', Y_test)


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
