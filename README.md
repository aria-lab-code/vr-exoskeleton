# vr-exoskeleton

Analysis and modeling from data collected for the VR Exoskeleton project.

## Install

To create a virtual environment and install dependencies from pip:

```commandline
python -m venv venv
```

On Mac/Linux:
```commandline
source venv/bin/activate
pip install -r requirements.txt
```

On Windows:
```commandline
venv\Scripts\activate
pip install -r requirements.txt
```

To also run/edit the IPython notebooks (the second command installs an IPython kernel profile based on this virtual environment):

```commandline
pip install -r requirements-nb.txt
python -m ipykernel install --user --name=vr-exoskeleton
```

## Download Data

Navigate to the [dataset hosted on Figshare](https://figshare.com/articles/dataset/EyeTrackingVRDataset/25749378).

Click the **Download all** button.

Extract the content to the `data` directory:

```commandline
unzip ~/Downloads/25749378.zip data
```

## Train

Train a MLP model:

```commandline
python -m vr_exoskeleton.train mlp --run_name my_run --seed 1
```

Train a LSTM model:

```commandline
python -m vr_exoskeleton.train lstm --run_name my_other_run --seed 2
```

## Export

Export the trained model to the [ONNX](https://pytorch.org/docs/stable/onnx.html) format:

```commandline
python vr_exoskeleton/export.py
```

You can check that the model exported with proper dimensions and input/output layer names via [Netron](https://netron.app).

## Notebooks

From the repository root, open Jupyter Lab by:

```commandline
jupyter-lab
```
