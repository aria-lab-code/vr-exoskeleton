import os
from collections import defaultdict

import numpy as np
import pandas as pd

PATH_DATA = 'data'
PATH_SCORES = os.path.join(PATH_DATA, 'ScoreRecord.csv')

N_TRIALS = 3


def get_user_task_paths():
    user_task_paths = defaultdict(lambda: defaultdict(list))
    users = sorted([name for name in os.listdir(PATH_DATA)
                    if name.startswith('User')],
                   key=lambda user_: int(user_[4:]))  # Numerical sort by ID number.
    for user in users:
        files = sorted(os.listdir(os.path.join(PATH_DATA, user)))
        for file in files:
            parts = file.split('_')
            assert parts[0] == user, f'File name in folder doesn\'t match user `{user}`: {parts[0]}'
            task = parts[1]
            path = os.path.join(PATH_DATA, user, file)
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


def load_X_Y(users, tasks, user_task_paths, window_size=3, keep_blinks=False):
    if window_size <= 0:
        raise ValueError(f'Window size cannot be less than 1: {window_size:d}')

    segments_X = list()
    segments_Y = list()
    for user in users:
        for task in tasks:
            for trial in range(N_TRIALS):
                df = pd.read_csv(user_task_paths[user][task][trial])

                intervals_valid = list()
                if keep_blinks:
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

                # Create numpy array from open-eye segments.
                for start, end in intervals_valid:
                    # Skip segments with a very short duration.
                    if end + 1 - start <= window_size:
                        continue

                    X = np.zeros((end + 1 - start - window_size, 9 * window_size), np.float32)
                    for w in range(window_size):
                        # Basically, the entire range starting from `start` accounts for the first
                        # nine-value-wide 'column' of X. Increment the starting point of the
                        # window via `w` and fill in the next nine-value-wide 'column' of X.
                        #
                        # X[:                      Every row.
                        #    , 9 * w:9 * w + 9]    The w-th nine-value-wide 'column' - 6 eye gaze, 3 head values.
                        #
                        # df.iloc[start + w:end + 1 - window_size + w           The indices of the ORIGINAL dataframe.
                        #                                            , 1:10]    The eye gaze columns and head columns.
                        X[:, 9 * w:9 * w + 9] = \
                            df.iloc[start + w:end + 1 - window_size + w, 1:10].to_numpy()
                    segments_X.append(X)

                    Y = np.zeros((end + 1 - start - window_size, 3), np.float32)
                    # All head values from index `start + window_size` until the end.
                    Y[:, :] = \
                        df.iloc[start + window_size:end + 1, 7:10].to_numpy()
                    segments_Y.append(Y)
    return np.concatenate(segments_X, axis=0), np.concatenate(segments_Y, axis=0)
