# dat6-d606-16
Bayesian Optimization of Parameters for Ocular Artifact Detection and Removal, and Classification of 4-class motor-imagery EEG data.

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

Refer to the installation guide of each dependency.

## Setup

1. Install spearmint: `sudo python d606/spearmint/setup.py install`

2. Get the data: [Download the dataset](http://bnci-horizon-2020.eu/database/data-sets) (001-2014), then create a folder called 'matfiles' and move the .mat files there.

## Usage

Optimize hyperparameters:

```python d606/optimizeparams.py```

Evaluate:

```python d606/main.py [subject]```