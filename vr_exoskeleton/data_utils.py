import os
from collections import defaultdict

import numpy as np
import pandas as pd

PATH_DATA = 'data'
PATH_SCORES = os.path.join(PATH_DATA, 'ScoreRecord.csv')

TASK_NAMES = (
    'LinearSmoothPursuit',
    'ArcSmoothPursuit',
    'RapidVisualSearch',
    'RapidVisualSearchAvoidance',
)
N_TRIALS = 3

SECONDS_PER_TRIAL = 90


def get_user_task_paths(use_eye_tracker_frames=False, ignore_users=None):
    if ignore_users is None:
        ignore_users = set()

    base_users_folder = 'Users'
    if not use_eye_tracker_frames:
        _write_90hz_files_if_needed()
        base_users_folder += '_90hz'
    path_users = os.path.join(PATH_DATA, base_users_folder)

    user_task_paths = defaultdict(lambda: defaultdict(list))
    users = sorted([name for name in os.listdir(path_users)
                    if name.startswith('User') and name not in ignore_users],
                   key=lambda user_: int(user_[4:]))  # Numerical sort by ID number.
    for user in users:
        files = sorted(os.listdir(os.path.join(path_users, user)))
        for file in files:
            parts = file.split('_')
            if parts[0] == user:
                task = parts[1]
                path = os.path.join(path_users, user, file)
                user_task_paths[user][task].append(path)

    keys0 = user_task_paths[users[0]].keys()
    for user in users[1:]:
        keys = user_task_paths[user].keys()
        if keys != keys0:
            raise ValueError(f'Not all tasks are the same among users: {user}:{keys} != {users[0]}:{keys0}')
    task_to_index = {task: i for i, task in enumerate(TASK_NAMES)}
    tasks = sorted(list(keys0), key=lambda k: task_to_index[k])
    for user in users:
        for task in tasks:
            if len(user_task_paths[user][task]) != N_TRIALS:
                raise ValueError(f'User `{user}` didn\'t complete exactly {N_TRIALS:d} trials for task `{task}`.')
    return users, tasks, user_task_paths


def _write_90hz_files_if_needed():
    path_users_90hz = os.path.join(PATH_DATA, 'Users_90hz')
    os.makedirs(path_users_90hz, exist_ok=True)

    users, tasks, user_task_paths = get_user_task_paths(use_eye_tracker_frames=True)
    for user in users:
        path_user_90hz = os.path.join(path_users_90hz, user)
        os.makedirs(path_user_90hz, exist_ok=True)
        for task in tasks:
            for path in user_task_paths[user][task]:
                _, tail = os.path.split(path)
                path_90hz = os.path.join(path_user_90hz, tail)
                if not os.path.exists(path_90hz):
                    print(f'Writing 90hz file: {path_90hz}')
                    df = pd.read_csv(path)
                    update_indices = list()
                    # Drop the second instance in pairs of repeated head directions.
                    for i in range(1, len(df)):
                        if any(v != v_prev for v, v_prev in zip(df.iloc[i, -3:], df.iloc[i - 1, -3:])):
                            update_indices.append(i)
                    df = df.iloc[update_indices]
                    df.to_csv(path_90hz, index=False)


def load_X_Y(
        paths,
        downsampling_rate=1,
        allow_blinks=False,
):
    if downsampling_rate < 1:
        raise ValueError(f'Down-sample rate cannot be less than 1: {downsampling_rate:d}')

    sequences_X = list()
    sequences_Y = list()
    for path in paths:
        df = pd.read_csv(path)
        df.drop(columns=['time_stamp(ms)'], inplace=True)
        data = df.to_numpy().astype(np.float32)
        n = data.shape[0]

        # Check whether this user blinked through the WHOLE trial (extremely unlikely!).
        if all(row[2] == 0.0 and row[5] == 0.0 for row in data):
            continue

        if not allow_blinks:
            left, right = slice(0, 3), slice(3, 6)
            for side in (left, right):
                data_side = data[:, side]
                indices_blink = [i for i, v in enumerate(data_side[:, -1]) if v == 0.0]
                intervals_blink = _to_intervals(indices_blink)
                for interval in intervals_blink:
                    if interval[0] == 0:
                        # Start of trial; Repeat first valid eye direction.
                        for i in range(0, interval[1] + 1):
                            data[i, side] = data[interval[1] + 1, side]
                    elif interval[1] == n - 1:
                        # End of trial; Repeat last valid eye direction.
                        for i in range(interval[0], n):
                            data[i, side] = data[interval[0] - 1, side]
                    else:
                        # Anywhere in the middle; Normalized linearly interpolate (nlerp).
                        length = interval[1] - interval[0] + 1
                        start, end = data[interval[0] - 1, side], data[interval[1] + 1, side]
                        delta = (end - start) / (length + 1)
                        for i in range(length):
                            delta_scale = start + (i + 1) * delta
                            data[interval[0] + i, side] = delta_scale / np.linalg.norm(delta_scale)

        # Create numpy array from open-eye segments.
        for shift in range(downsampling_rate):
            # Consider only every `downsampling_rate` rows, if applicable.
            data_shift = data[shift::downsampling_rate]
            X = data_shift[:-1]
            Y = data_shift[1:, 6:]
            sequences_X.append(X)
            sequences_Y.append(Y)

    # Truncate to smallest sequence length.
    length_min = min([sequence_X.shape[0] for sequence_X in sequences_X])
    X, Y = (np.stack([seq[:length_min] for seq in sequences], axis=0)
            for sequences in (sequences_X, sequences_Y))
    return X, Y  # (n, seq_len, 6 + h), (n, seq_len, h)


def _to_intervals(indices):
    if len(indices) == 0:
        return list()

    intervals = list()
    start = indices[0]
    index_prev = indices[0]
    for index in indices[1:]:
        if index - 1 != index_prev:
            intervals.append((start, index_prev))
            start = index
        index_prev = index
    intervals.append((start, indices[-1]))
    return intervals
