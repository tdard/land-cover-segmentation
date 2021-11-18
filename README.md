# Land Cover Segmentation 

Using SG2LC Land Cover Dataset (modified by Preligens in ENS Challenge)

# Usage

## Dataset (pytorch)
```
from land_cover import LandCoverSegmentationDataset

dataset = LandCoverSegmentationDataset(images_dir=..., masks_dir=..., transform=...)
```

## Datamodule (pytorch-lightning)
```
from land_cover import LandCoverSegmentationDataModule

dm = LandCoverSegmentationDataModule(images_dir=..., masks_dir=..., transform=..., <extra_params>)
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
```
export IMAGES_FOLDER=<the/folder/to/the/images>
export MASKS_FOLDER=<the/folder/to/the/masks>
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