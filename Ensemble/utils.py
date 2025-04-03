import os
import datetime
import torch
import GPUtil
from torch.utils.data import DataLoader
import gc





##############
##  GENERAL
##############
#region

def free_up_memory(model):
  """Frees up GPU memory and deletes the model provided"""
  gc.collect()
  torch.cuda.empty_cache()
  del model
  GPUtil.showUtilization()

def timestamp():
  """Creates a timestamp, which can be used for file naming etc."""
  return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


#Source:https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
#not adapted
class StreamToLogger(object):
    """ Fake file-like stream object that redirects writes to a logger instance. """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

#endregion

##############
##  LANDCOVER
##############
#region


def calculate_class_weights_landcover(train_set):
  """Counts the pixels for each class in the landcover (training) set"""

  class_counts = {
      0: 0, # grey, unlabeled
      1: 0, # red, building
      2: 0,  # green, woodlamd
      3: 0, # blue, water
      4: 0 # yellow, road
      } 


  train_loader = DataLoader(train_set, shuffle=False, batch_size=16, num_workers=os.cpu_count(), pin_memory=True)

  for batch in train_loader:
    mask = batch['mask']
    unique_counts = torch.unique(mask, return_counts=True)
    

    class_labels = unique_counts[0].tolist()
    counts = unique_counts[1].tolist()

    for class_label, count in zip(class_labels,counts):
      print(class_label)
      print(count)
      class_counts[class_label] += count

  return class_counts

#endregion

##############
##  FLOODS
##############
#region

def calculate_class_weights_floods(train_set):
  """Counts the pixels for each class in the s1gflood (training) set"""

  class_counts = {
      0: 0, # NON flooded
      1: 0 # flooded area
      }


  train_loader = DataLoader(train_set, shuffle=False, batch_size=16, num_workers=os.cpu_count(), pin_memory=True)

  for batch in train_loader:
    mask = batch['mask']
    unique_counts = torch.unique(mask, return_counts=True)





    class_labels = unique_counts[0].tolist()
    counts = unique_counts[1].tolist()

    for class_label, count in zip(class_labels,counts):
      print(class_label)
      print(count)
      class_counts[class_label] += count



  return class_counts

#endregion