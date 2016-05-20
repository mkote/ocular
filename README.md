# dat6-d606-16
Bayesian Optimization of Parameters for Ocular Artifact Detection and Removal, and Classification of
4-class motor-imagery EEG data.

## Requirements

### Linux / Ubuntu / Windows7/8

- [Anaconda 2.7 python packages](https://www.continuum.io/downloads) (recommended)
- mne
- protobuf
- skll

or

- Python 2.7
- SciPy
- NumPy
- Scikit-Learn
- mne
- protobuf
- skll

Refer to the installation guides of each dependency (or just use the anaconda2
distribution).

## Setup

    sudo python d606/spearmint/setup.py install

## Usage

Optimize hyperparameters:

    python d606/optimizeparams.py

Evaluate:

    python d606/main.py [subject]