import torchvision.transforms.v2 as v2
from pathlib import Path
import os 

from .datasets import FloodDataset, LandcoverDataset
from .normalization import normalizer, normalizer_post, normalizer_pre



##############
##  GENERAL
##############
"""
This section loads train, test and val sets for the S1GFloods and LandCover.aI datasets, so these can be directly imported for later use.
Check the folder structure in the README or adapt the file if <ou use a different folder structure.  
"""

#region
dataset_path = Path(os.path.join(os.path.dirname(__file__),"..","Datasets"))

train_dir_floods = dataset_path.joinpath('data_floods').joinpath('train')
val_dir_floods = dataset_path.joinpath('data_floods').joinpath('val')
test_dir_floods = dataset_path.joinpath('data_floods').joinpath('test')


data_augmentation_floods = [
    v2.GaussianBlur((9,9), sigma=(0.1, 5.0)), # sigma is chosen uniformly at random
    v2.RandomHorizontalFlip(),  # p = 0.5
    v2.RandomVerticalFlip(),    # p = 0.5
    v2.RandomAdjustSharpness(2),  # p = 0.5
]


train_set_floods_local = FloodDataset(train_dir_floods, "train.txt",transform = data_augmentation_floods, normalizer_pre = normalizer_pre, normalizer_post = normalizer_post)
val_set_floods_local = FloodDataset(val_dir_floods, "val.txt",normalizer_pre = normalizer_pre, normalizer_post = normalizer_post)
test_set_floods_local = FloodDataset(test_dir_floods, "test.txt",normalizer_pre = normalizer_pre, normalizer_post = normalizer_post)


data_dir_landcover = dataset_path.joinpath('data_landcover')
image_dir_landcover = data_dir_landcover.joinpath('output')
test_txt_landcover = data_dir_landcover.joinpath('test.txt')
train_txt_landcover = data_dir_landcover.joinpath('train.txt')
val_txt_landcover = data_dir_landcover.joinpath('val.txt')


# list of data augmentation methods (one is always applied)
data_augmentation_landcover = [
      v2.GaussianBlur((9,9), sigma=(0.1, 5.0)), # sigma is chosen uniformly at random
      v2.RandomHorizontalFlip(),  # p = 0.5
      v2.RandomVerticalFlip(),    # p = 0.5
      v2.RandomAdjustSharpness(2),  # p = 0.5
      v2.ColorJitter(brightness=0.5,contrast=0,saturation=0,hue=0), #brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
      v2.ColorJitter(brightness=0,contrast=0.5,saturation=0,hue=0), # same as brightness
      v2.ColorJitter(brightness=0,contrast=0,saturation=0.5,hue=0), # same as brightness
      v2.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0.07), # hue_factor is chosen uniformly from [-hue, hue]
  ]

# load all datasets now normalized
train_set_landcover_local  = LandcoverDataset(image_dir_landcover, image_dir_landcover, train_txt_landcover, '_m', transform = data_augmentation_landcover, normalizer=normalizer)
val_set_landcover_local  = LandcoverDataset(image_dir_landcover, image_dir_landcover, val_txt_landcover, '_m',  normalizer=normalizer)
test_set_landcover_local  = LandcoverDataset(image_dir_landcover, image_dir_landcover, test_txt_landcover, '_m',  normalizer=normalizer)


#endregion


