import argparse
import os.path

import torch

from vr_exoskeleton import gaze_modeling


def main():
    parser = argparse.ArgumentParser(
        description='Export a saved (trained) model to the ONNX format.'
    )
    parser.add_argument('path',
                        help='Path to `.pth` file.')
    parser.add_argument('--window_size', default=3, type=int,
                        help='Window size of the MLP.')
    parser.add_argument('--hidden_sizes', nargs='*', type=int,
                        help='Sizes of the hidden layers of the MLP.')
    parser.add_argument('--drop_gaze_z', action='store_true',
                        help='Flag to drop the z-dimension of the left and right gaze vectors.')
    kwargs = vars(parser.parse_args())
    export(**kwargs)


def export(path, window_size=3, hidden_sizes=None, drop_gaze_z=False):
    model = gaze_modeling.GazeMLP(window_size=window_size, hidden_sizes=hidden_sizes, drop_gaze_z=drop_gaze_z)
    weights = torch.load(path)
    model.load_state_dict(weights)

    head, tail = os.path.split(path)  # Separate leading directories.
    fname_base, _ = os.path.splitext(tail)  # Without extension.
    path_out = os.path.join(head, f'{fname_base}.onnx')
    torch.onnx.export(
        model,
        torch.zeros(1, 9 * window_size),
        path_out,
        export_params=True,
        opset_version=10,  # This is what Jordan used previously.
    )
    print(f'Saved ONNX format model to `{path_out}`.')


if __name__ == '__main__':
    main()
