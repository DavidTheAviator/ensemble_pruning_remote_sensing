import torch
from tqdm import tqdm
import torchvision.transforms.v2 as v2





##############
##  GENERAL
##############
#region

#Source: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/7
#Not adapted
class NormalizeInverse(v2.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
    
#endregion


##############
##  LANDCOVER
##############
#region

#Source: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_std_mean.py
#Slightly adapted (Changed some variable names, adapted to my dataloader)
def get_mean_std_landcover(loader):
  """
  Calculate the per channel mean and standard variation of a dataset

  Args:
    loader (torch.utils.data.DataLoader): A Dataloader for the dataset, which is examined. Samples should be in the standard format (B, C, H, W)

  Returns:
    tuple: A tuple containg the per channel mean and standard variation 
  """

  #var[x] = E[X**2] - E[X]**2
  channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0


  for batch in tqdm(loader):
    images = batch['image']

    channels_sum += torch.mean(images, dim=[0,2,3])
    channels_sqrd_sum += torch.mean(images**2, dim=[0,2,3])
    num_batches += 1

  mean = channels_sum / num_batches
  mean_of_sqrd = channels_sqrd_sum/num_batches
  std = (mean_of_sqrd - (mean **2)) ** 0.5

  return mean, std



#Landcover
#Fixed mean and std calculated before (per channel)
mean = torch.tensor([ 94.2679, 101.2565,  87.5951])
std = torch.tensor([41.9003, 34.9157, 28.6940])

#Normalizer/Denormalizer for global use, based on the precalculated values
denormalizer = NormalizeInverse(mean = mean , std= std)
normalizer = v2.Normalize(mean = mean, std = std)

#endregion



##############
##  FLOODS
##############
#region


#Source: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_std_mean.py
#Slightly adapted (Changed some variable names, adapted to my dataloader, made it work with pre/post flood images)
def get_mean_std_floods(loader):
  """
  Calculate the per channel mean and standard variation of a dataset (here both for pre- and post- flood images)

  Args:
    loader (torch.utils.data.DataLoader): A Dataloader for the dataset, which is examined. Samples should be in the standard format (B, C, H, W)

  Returns:
    tuple: A tuple containg the per channel mean and standard variation 
  """

  #var[x] = E[X**2] - E[X]**2
  num_batches = 0
  channels_sum_pre, channels_sqrd_sum_pre = 0, 0
  channels_sum_post, channels_sqrd_sum_post = 0, 0


  for batch in tqdm(loader):
    imgs_pre = batch['image_pre']
    imgs_post = batch['image_post']

    channels_sum_pre += torch.mean(imgs_pre, dim=[0,2,3])
    channels_sqrd_sum_pre += torch.mean(imgs_pre**2, dim=[0,2,3])

    channels_sum_post += torch.mean(imgs_post, dim=[0,2,3])
    channels_sqrd_sum_post += torch.mean(imgs_post**2, dim=[0,2,3])



    num_batches += 1

  mean_pre = channels_sum_pre / num_batches
  mean_of_sqrd_pre = channels_sqrd_sum_pre/num_batches
  std_pre = (mean_of_sqrd_pre - (mean_pre **2)) ** 0.5

  mean_post = channels_sum_post / num_batches
  mean_of_sqrd_post = channels_sqrd_sum_post/num_batches
  std_post = (mean_of_sqrd_post - (mean_post **2)) ** 0.5

  return [mean_pre, std_pre,mean_post, std_post]





#Fixed mean and std calculated before (per channel)
mean_pre = torch.tensor([131.6654, 131.6654, 131.6654])
std_pre = torch.tensor([63.1413, 63.1413, 63.1413])
mean_post = torch.tensor([118.4424, 118.4424, 118.4424])
std_post = torch.tensor([77.5159, 77.5159, 77.5159])


normalizer_pre = v2.Normalize(mean = mean_pre, std = std_pre)
normalizer_post = v2.Normalize(mean = mean_post, std= std_post)

denormalizer_pre = NormalizeInverse(mean = mean_pre, std = std_pre)
denormalizer_post = NormalizeInverse(mean = mean_post, std= std_post)


#endregion