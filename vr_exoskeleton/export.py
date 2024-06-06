import torch

from vr_exoskeleton import gaze_modeling


def main():
    window_size = 3
    pth_fname = 'best_val_1717450374.pth'

    if not pth_fname.endswith('.pth'):
        pth_fname = pth_fname + '.pth'

    model = gaze_modeling.GazeMLP(window_size)
    weights = torch.load(pth_fname)
    model.load_state_dict(weights)
    torch.onnx.export(
        model,
        torch.zeros(1, 9 * window_size),
        f'{pth_fname[:-4]}.onnx',
        export_params=True,
        opset_version=10,  # This is what Jordan used previously.
    )


if __name__ == '__main__':
    main()
