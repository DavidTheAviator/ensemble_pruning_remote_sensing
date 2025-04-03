import os
import sys
import json
import copy
import logging
from pathlib import Path



import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.v2 as v2


# Add Parent Directory to Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Project-Specific Imports
from Ensemble.normalization import normalizer_pre, normalizer_post
from Ensemble.datasets import FloodDataset
from Ensemble.models import SiamUnet_conc
from Ensemble.utils import free_up_memory, timestamp, StreamToLogger
from Ensemble.evaluation import evaluate_floods
from Ensemble.train import train_model_floods

            
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
#parent dir of Scripts directory
pruning_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

###set up all the folders for the experiment
experiment_folder_path = pruning_dir.joinpath('experiments').joinpath('created_ensembles').joinpath(f"experiment_floods_{timestamp()}")
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

train_model_params = {
    'epochs':25, 
    'batch_size': 16, 
    'learning_rate': 1e-4, 
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
logger_events.addHandler(logging.StreamHandler(sys.stdout)) #to also show this on stdout

#log stdout and stderr
sys.stdout = StreamToLogger(logger_streams,logging.INFO)
sys.stderr = StreamToLogger(logger_streams,logging.ERROR)

logger_events.info("-------------SCRIPT S1GFloods------------------")


logger_events.info("Created Loggers and Folder Structure")
logger_events.info("Starting Main Training Loop")



# list of data augmentation methods (one is always applied)
data_augmentation = [
    v2.GaussianBlur((9,9), sigma=(0.1, 5.0)), # sigma is chosen uniformly at random
    v2.RandomHorizontalFlip(),  # p = 0.5
    v2.RandomVerticalFlip(),    # p = 0.5
    v2.RandomAdjustSharpness(2),  # p = 0.5
]


# load all datasets now normalized
train_set = FloodDataset(train_dir, "train.txt",transform = data_augmentation, normalizer_pre = normalizer_pre, normalizer_post = normalizer_post)
val_set = FloodDataset(val_dir, "val.txt",normalizer_pre = normalizer_pre, normalizer_post = normalizer_post)
test_set = FloodDataset(test_dir, "test.txt",normalizer_pre = normalizer_pre, normalizer_post = normalizer_post)


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
  model = SiamUnet_conc(n_channels=3,n_classes=1)
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
  run_name = f'model{i}_lr={train_model_params["learning_rate"]}_{timestamp()}'
  writer = SummaryWriter(runs_path.joinpath(run_name))

  train_model_floods(
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

  mIOU_test = evaluate_floods(model,test_loader, device, amp=True, log_images=False)
  mIOU_val = evaluate_floods(model, val_loader, device, amp=True, log_images=False)

  logger_models.info(f"mIOU on Val: {mIOU_val}")
  logger_models.info(f"mIOU on Test: {mIOU_test}")

  logger_events.info(f"mIOU on Test: {mIOU_test}")

  #delete the model
  free_up_memory(model)
  logger_events.info(f"Deleted {model_name} from GPU")

#endregion














