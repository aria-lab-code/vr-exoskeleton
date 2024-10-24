import os
from collections import defaultdict

import numpy as np
import pandas as pd

PATH_DATA = 'data'
PATH_USERS = os.path.join(PATH_DATA, 'Users')
PATH_USERS_90HZ = os.path.join(PATH_DATA, 'Users_90hz')
PATH_SCORES = os.path.join(PATH_DATA, 'ScoreRecord.csv')

TASK_NAMES = (
    'ArcSmoothPursuit',
    'LinearSmoothPursuit',
    'RapidVisualSearch',
    'RapidVisualSearchAvoidance',
)
N_TRIALS = 3

SECONDS_PER_TRIAL = 90


def get_user_task_paths(use_update_frames=True):
    if use_update_frames:
        _write_90hz_files_if_needed()
        path_users = PATH_USERS_90HZ
    else:
        path_users = PATH_USERS

    user_task_paths = defaultdict(lambda: defaultdict(list))
    users = sorted([name for name in os.listdir(path_users)
                    if name.startswith('User')],
                   key=lambda user_: int(user_[4:]))  # Numerical sort by ID number.
    for user in users:
        files = sorted(os.listdir(os.path.join(path_users, user)))
        for file in files:
            parts = file.split('_')
            assert parts[0] == user, f'File name in folder doesn\'t match user `{user}`: {parts[0]}'
            task = parts[1]
            path = os.path.join(path_users, user, file)
            user_task_paths[user][task].append(path)

    keys0 = user_task_paths[users[0]].keys()
    for user in users[1:]:
        keys = user_task_paths[user].keys()
        if keys != keys0:
            raise ValueError(f'Not all tasks are the same among users: {user}:{keys} != {users[0]}:{keys0}')
    tasks = sorted(list(keys0))
    for user in users:
        for task in tasks:
            if len(user_task_paths[user][task]) != N_TRIALS:
                raise ValueError(f'User `{user}` didn\'t complete exactly {N_TRIALS:d} trials for task `{task}`.')
    return users, tasks, user_task_paths


def _write_90hz_files_if_needed():
    os.makedirs(PATH_USERS_90HZ, exist_ok=True)
    users, tasks, user_task_paths = get_user_task_paths(use_update_frames=False)
    for user in users:
        path_user_90hz = os.path.join(PATH_USERS_90HZ, user)
        os.makedirs(path_user_90hz, exist_ok=True)
        for task in tasks:
            for path in user_task_paths[user][task]:
                _, tail = os.path.split(path)
                path_90hz = os.path.join(path_user_90hz, tail)
                if not os.path.exists(path_90hz):
                    print(f'Writing 90hz file: {path_90hz}')
                    df = pd.read_csv(path)
                    update_indices = list()
                    for i in range(1, len(df)):
                        if any(v != v_prev for v, v_prev in zip(df.iloc[i, -2:], df.iloc[i - 1, -2:])):
                            update_indices.append(i)
                    df = df.iloc[update_indices]
                    df.to_csv(path_90hz, index=False)
                    # pd.DataFrame().to_csv()


def load_sequences_X_Y(
        users,
        tasks,
        user_task_paths,
        window_size=1,
        downsampling_rate=1,
        drop_blinks=False,
        drop_gaze_z=False,
        predict_relative_head=False,
        drop_head_input=False,
):
    if window_size < 1:
        raise ValueError(f'Window size cannot be less than 1: {window_size:d}')
    if downsampling_rate < 1:
        raise ValueError(f'Down-sample rate cannot be less than 1: {downsampling_rate:d}')

    sequences_X = list()
    sequences_Y = list()
    for user in users:
        for task in tasks:
            for trial in range(N_TRIALS):
                df = pd.read_csv(user_task_paths[user][task][trial])
                df = df.drop(columns=['time_stamp(ms)'])

                intervals_valid = list()
                if not drop_blinks or drop_gaze_z:
                    # Use all rows, including during blinks.
                    intervals_valid.append((0, len(df) - 1))
                else:
                    # Filter on both eyes open.
                    df_open = df[(df['eye_in_head_left_z'] != 0.0) & (df['eye_in_head_right_z'] != 0.0)]

                    if len(df_open) > 0:  # It is unlikely that any user closed their eyes through the entire trial.
                        # Collect only contiguous ranges of data where the eyes are open.
                        start_open = df_open.index[0]
                        for i in range(1, len(df_open.index)):
                            if df_open.index[i - 1] != df_open.index[i] - 1:
                                intervals_valid.append((start_open, df_open.index[i - 1]))
                                start_open = df_open.index[i]
                        intervals_valid.append((start_open, df_open.index[-1]))

                if drop_gaze_z:
                    df = df.drop(columns=['eye_in_head_left_z', 'eye_in_head_right_z'])
                instance_size = len(df.columns)
                if drop_head_input:
                    instance_size -= 3  # Don't include head_[x|y|z] in `X`.

                # Create numpy array from open-eye segments.
                for start, end in intervals_valid:
                    for shift in range(downsampling_rate):
                        # Skip segments with a duration that doesn't fit an entire window.
                        if end - start - shift < window_size * downsampling_rate:
                            continue

                        # Consider only every `downsampling_rate` rows, if applicable.
                        data = df.iloc[start + shift:end + 1:downsampling_rate].to_numpy()

                        X = np.zeros((len(data) - window_size, instance_size * window_size), np.float32)
                        for w in range(window_size):
                            # The entire range starting from `start` accounts for the first
                            # [9|7|6|4]-value-wide 'column' of X. Increment the starting point of the
                            # window via `w` and fill in the next [9|7|6|4]-value-wide 'column' of X.
                            #
                            # X[:                            Every row.
                            #    , w * i_s:(w + 1) * i_s]    The w-th [9|7|6|4]-value-wide 'column',
                            #                                  i.e., [6|4] eye gaze, [3|0] head values.
                            #
                            # data[w:-window_size + w           Exactly `n - window_size` data points,
                            #                                     starting from index `w`.
                            #                        , :i_s]    First `instance_size` columns
                            #                                     (may exclude head).
                            X[:, w * instance_size:(w + 1) * instance_size] = data[w:-window_size + w, :instance_size]

                        # All head values (`-3:`) from `window_size` until the end.
                        Y = data[window_size:, -3:].astype(np.float32)
                        if predict_relative_head:
                            Y = Y[1:] - Y[:-1]
                            X = X[:-1]  # Chop off the last data point to match length of Y.

                        sequences_X.append(X)
                        sequences_Y.append(Y)

    return sequences_X, sequences_Y
