![test](https://github.com/tdard/land-cover-segmentation/actions/workflows/tests.yml/badge.svg)

# Land Cover Segmentation 

Using SG2LC Land Cover Dataset (modified by Preligens in ENS Challenge) <br>

To use and download the dataset, you must own an account on [https://challengedata.ens.fr/](https://challengedata.ens.fr/), and you must participate to the [Preligens Challenge 2021](https://challengedata.ens.fr/participants/challenges/48/).

# Usage

## Dataset (pytorch)
If you want to download the dataset, then you must set the following environment variables with your credentials: <br>
```
export CHALLENGE_USERNAME=<username>
export CHALLENGE_PWD=<userpassword>
```

```
from land_cover import LandCoverSegmentationDataset

dataset = LandCoverSegmentationDataset(root=..., transform=...)
```

## Datamodule (pytorch-lightning)
```
from land_cover import LandCoverSegmentationDataModule

dm = LandCoverSegmentationDataModule(root=..., transform=..., <extra_params>)
```

Note: the datamodule does not have test and predict dataloaders

## Augmentations
- select_rgb_channels_only (since the dataset is RGB-NIR)


# Installation
## Environment: conda
Create a new environment using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate land_cover
```

## Testing

Make sure that the following variables are set if you want to test the download:
```
export CHALLENGE_USERNAME=<username>
export CHALLENGE_PWD=<userpassword>
```

Then, you can run pytest in the root directory:
```
$ pytest
```

## Installation with pip
Make sure you are located in the root directory and that the conda environment is activated before installing
```
python3.8 -m pip install .
```
