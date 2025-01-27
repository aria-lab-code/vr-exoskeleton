import argparse
import os.path

import torch

from vr_exoskeleton import gaze_modeling


def main():
    parser = argparse.ArgumentParser(
        description='Export a saved (trained) model to the ONNX format.'
    )
    parser.add_argument('model_type', choices=('mlp', 'lstm'),
                        help='Model architecture.')
    parser.add_argument('path',
                        help='Path to `.pth` file.')
    parser.add_argument('--hidden_sizes', nargs='*', type=int,
                        help='Sizes of the hidden layers of the MLP.')
    parser.add_argument('--drop_head_input', action='store_true',
                        help='Flag to remove head position data as input, hypothetically improving inference.')
    kwargs = vars(parser.parse_args())
    export(**kwargs)


def export(model_type, path, hidden_sizes=None, drop_head_input=False):
    if hidden_sizes is None:
        hidden_sizes = list()

    instance_size = 9
    if drop_head_input:
        instance_size -= 3
    if model_type == 'mlp':
        model = gaze_modeling.GazeMLP(hidden_sizes=hidden_sizes)
        inputs = torch.zeros(1, model.input_size)
        input_names = ('input',)
        output_names = ('output',)
    elif model_type == 'lstm':
        model = gaze_modeling.GazeLSTM(hidden_sizes=hidden_sizes)
        inputs = (torch.zeros(1, model.input_size),
                  torch.zeros(1, 1, model.hidden_size),
                  torch.zeros(1, 1, model.hidden_size))
        input_names = ('input', 'h0', 'c0')
        output_names = ('output', 'hn', 'cn')
    else:
        raise ValueError(f'Unknown `model_type`: {model_type}')

    weights = torch.load(path, weights_only=True)
    model.load_state_dict(weights)

    head, tail = os.path.split(path)  # Separate leading directories.
    fname_base, _ = os.path.splitext(tail)  # Without extension.
    path_out = os.path.join(head, f'{fname_base}.onnx')
    torch.onnx.export(
        model,
        inputs,
        path_out,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=10,  # This is what Jordan used previously.
        # opset_version=11,
        # do_constant_folding=True,
    )
    print(f'Saved ONNX format model to `{path_out}`.')


if __name__ == '__main__':
    main()
