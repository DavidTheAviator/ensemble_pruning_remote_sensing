
import json
import copy
import torch
import os
import sys
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import os
from pathlib import Path
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

from Ensemble.normalization import normalizer
from Ensemble.datasets import LandcoverDataset
from Ensemble.models import UNet
from Ensemble.utils import free_up_memory, timestamp, StreamToLogger
from Ensemble.evaluation import evaluate_landcover
from Ensemble.train import train_model_landcover



            
##############
##  Ensemble Creation Script
##############
#region

"""
Experiment Folder Structure 
experiment_{timestamp}/
├── events.log
├── models.log
├── datasets/
│   ├── dataset1.npy
│   ├── dataset1.txt
│   ├── dataset2.npy
│   ├── ....
├── saved_models/
│   ├── model1.pth
│   ├── model2.pth
│   ├── model3.pth
│   ├── ....
├── runs/
│   ├── model1_{params}_{timestamp}
│   ├── model2_{params}_{timestamp}
│   ├── model3_{params}_{timestamp}
│   ├── ....
└── checkpoints/
    ├── model1_{params}_{timestamp}
    │   ├──checkpoint_epoch{epoch}.pth
    │   └──checkpoint_epoch{epoch}.pth
    ├── model2_{params}_{timestamp}
    │   ├──checkpoint_epoch{epoch}.pth
    │   └──checkpoint_epoch{epoch}.pth
    ├── ....


"""

###set up all the folders for the experiment

#parent dir of Scripts directory
pruning_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
 
experiment_folder_path = pruning_dir.joinpath('experiments').joinpath('created_ensembles').joinpath(f"experiment_landcover_{timestamp()}")
checkpoints_path = experiment_folder_path.joinpath('checkpoints')
runs_path = experiment_folder_path.joinpath('runs')
datasets_path = experiment_folder_path.joinpath('datasets')
saved_models_path = experiment_folder_path.joinpath('saved_models')
os.system(f"mkdir {experiment_folder_path}")
os.system(f"mkdir {checkpoints_path}")
os.system(f"mkdir {saved_models_path}")
os.system(f"mkdir {runs_path}")
os.system(f"mkdir {datasets_path}")



#########################
#####ADAPT DATA HERE
##########################

train_dir = pruning_dir.joinpath('Datasets').joinpath('data_floods').joinpath('train')
val_dir = pruning_dir.joinpath('Datasets').joinpath('data_floods').joinpath('val')
test_dir = pruning_dir.joinpath('Datasets').joinpath('data_floods').joinpath('test')

data_dir = pruning_dir.joinpath('Datasets').joinpath('data_landcover')
image_dir = data_dir.joinpath('output')
test_txt = data_dir.joinpath('test.txt')
train_txt = data_dir.joinpath('train.txt')
val_txt = data_dir.joinpath('val.txt')

train_model_params = {
    'epochs': 25, 
    'batch_size': 8, 
    'learning_rate': 1e-2, 
    'min_learning_rate': 1e-05, 
    'momentum': 0.9, 
    't0': 2, 
    'checkpoint_list': [10,20], 
    'checkpoint_path': checkpoints_path, 
    'amp': True, 
    'weight_decay': 0, 
    'gradient_clipping': 1.0
  }

ensemble_size = 10

#########################
#########################




# create loggers
logger_events = logging.getLogger('events')
logger_models = logging.getLogger('models')
logger_streams = logging.getLogger('streams')
logger_events.setLevel(logging.INFO)
logger_models.setLevel(logging.INFO)
logger_streams.setLevel(logging.INFO)


# create file handlers
fh_events = logging.FileHandler(experiment_folder_path.joinpath('events.log'))
fh_models = logging.FileHandler(experiment_folder_path.joinpath('models.log'))
fh_streams = logging.FileHandler(experiment_folder_path.joinpath('streams.log'))
fh_events.setLevel(logging.INFO)
fh_models.setLevel(logging.INFO)
fh_streams.setLevel(logging.INFO)

#create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh_events.setFormatter(formatter)

logger_events.addHandler(fh_events)
logger_models.addHandler(fh_models)
logger_streams.addHandler(fh_streams)
#logger_events.addHandler(logging.StreamHandler(sys.stdout)) #to also show this on stdout

#log stdout and stderr
sys.stdout = StreamToLogger(logger_streams,logging.INFO)
sys.stderr = StreamToLogger(logger_streams,logging.ERROR)


logger_events.info("-------------SCRIPT LANDCOVER AI------------------")
logger_events.info("Created Loggers and Folder Structure")
logger_events.info("Starting Main Training Loop")



# list of data augmentation methods (one is always applied)
data_augmentation = [
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
train_set = LandcoverDataset(image_dir, image_dir, train_txt, '_m', transform = data_augmentation, normalizer=normalizer)
val_set = LandcoverDataset(image_dir, image_dir, val_txt, '_m',  normalizer=normalizer)
test_set = LandcoverDataset(image_dir, image_dir, test_txt, '_m',  normalizer=normalizer)

rng = np.random.default_rng()

#endregion

#region


#the main training loop
for i in range(1,ensemble_size+1):
  logger_events.info("------------------------------------")
  logger_events.info(f"Beginning of Training of Model: {i}")


  bagging_index = rng.choice(np.arange(len(train_set)), size=len(train_set),replace=True,shuffle=True)
  bagging_set = torch.utils.data.Subset(train_set, bagging_index)

  logger_events.info(f"Created Bagging Set of Size: {len(bagging_set)}")
  logger_events.info(f"First 10 elements: {bagging_index[:10]}")
  logger_events.info("Saving dataset index (as txt and np array)....")

  #save both as .txt and .npy
  np.savetxt(datasets_path.joinpath(f'dataset_{i}.txt'), bagging_index,fmt='%.i', delimiter=',')
  np.save(datasets_path.joinpath(f'dataset_{i}.npy'), bagging_index) 


  logger_events.info("Succesfully saved datasets....")



  #####Training
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = UNet(n_channels=3, n_classes=5)
  model = model.to(device= device, memory_format=torch.channels_last)



  logger_models.info("----------")
  logger_models.info(f"Model {i}")
  #for logging
  train_model_params_l = copy.deepcopy(train_model_params)
  train_model_params_l['checkpoint_path'] = str(train_model_params_l['checkpoint_path'])
  logger_models.info(json.dumps(train_model_params_l,indent=4))
  logger_events.info(f"Saved Training Parameters of Model {i}")
  logger_events.info(f"Start Training for {train_model_params['epochs']} Epochs")


  #Log via tensorboard
  run_name = f'model{i}_lr={train_model_params["learning_rate"]}_lrmin={train_model_params["min_learning_rate"]}_momentum={train_model_params["momentum"]}_{timestamp()}'
  writer = SummaryWriter(runs_path.joinpath(run_name))

  train_model_landcover(
      run_name,
      model,
      device,
      bagging_set,
      val_set,
      writer,
      **train_model_params
  )


  #SAVE the model
  model_name = f'model_{i}_{timestamp()}.pth'
  torch.save(model.state_dict(),saved_models_path.joinpath(model_name))

  logger_events.info(f"Saved the model as: {model_name}")


  logger_events.info(f"Testing {model_name}")
  
  #TEST the model 
  test_loader = DataLoader(test_set, shuffle=False, drop_last=False, batch_size= 16, num_workers = os.cpu_count(), pin_memory=True)
  val_loader = DataLoader(val_set, shuffle=False, drop_last=False, batch_size= 16, num_workers = os.cpu_count(), pin_memory=True)


  mIOU_test = evaluate_landcover(model,test_loader, device, amp=True, log_images=False)
  mIOU_val = evaluate_landcover(model, val_loader, device, amp= True, log_images=False)

  logger_models.info(f"mIOU on Val: {mIOU_val}")
  logger_models.info(f"mIOU on Test: {mIOU_test}")

  logger_events.info(f"mIOU on Test: {mIOU_test}")

  #delete the model
  free_up_memory(model)
  logger_events.info(f"Deleted {model_name} from GPU")

#endregion














