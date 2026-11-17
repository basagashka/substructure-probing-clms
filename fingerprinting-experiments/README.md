# Fingerprinting Experiments

Basic models trained on fingerprints extracted using [RDKit](https://zenodo.org/records/13469390). 

## Installation

To create an environment, you can use conda via: 

    conda create --name=fingerprinting-experiments python=3.13.5
    conda activate fingerprinting-experiments
    pip install -r requirements.txt 

## Fingerprint extraction

Extract fingerprints via:

    ./extract_all_fps.sh

## Running the Code

We provide scripts for training and evaluating the models on lipophilicity prediction `python train_eval.py` and solubility prediction `train_eval_esol.py`