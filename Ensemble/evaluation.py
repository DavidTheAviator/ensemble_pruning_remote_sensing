import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
from .plotting import images_mask_tb_landcover
from .plotting import images_mask_tb_floods





##############
##  GENERAL
##############
#region

def iou(input: Tensor, target: Tensor):
  """
  Calculate the (mean) Intersection over Union (IoU) between two tensors.

 
  Args:
    input (torch.Tensor): A tensor of predicted segmentation masks (one hot encoded) with shape (batch, classes, H, W)
    target (torch.Tensor): A tensor of ground truth segmentation masks with the same shape as input, should also be one hot encoded

    Returns:
        float: The mean IoU value across the batch and classes.
  """


  #transform input and target from shape (B, C, H, W) to (B*C,H,W)
  input = input.flatten(0,1)
  target = target.flatten(0,1)

  assert input.size() == target.size()

  assert input.dim() == 3

  sum_dim = (-1,-2)
  
  #IOU = TP / TP + FP + FN

  #sum up the TP along Height and width, to get a sum for each class, image combination
  tp = (input * target).sum(dim=sum_dim)

  #denom = TP + FP                  TP + FN                  - TP
  denom =  input.sum(dim=sum_dim) + target.sum(dim=sum_dim) - tp

  
  #for cases where denom is zero,  intersection over union is 1
  iou = torch.where(denom == 0, 1, tp/denom)


  return iou.mean()

#endregion



##############
##  LANDCOVER
##############
#region

#Source: https://github.com/milesial/Pytorch-UNet/blob/master/evaluate.py
#Adapted (Replaced dice score with IoU, added the image logging with tensorboard)
@torch.inference_mode()
def evaluate_landcover(net, dataloader, device, amp,log_images=False,val_set=None, global_step=None, writer=None):
    """
    Evaluate the performance of a neural network on the Landcover AI dataset using the IoU metric

    Args:
        net (torch.nn.Module): The neural net to evaluate
        dataloader (torch.utils.data.DataLoader): The dataloader for the validation dataset
        device (torch.device): The device to run the evaluation on (e.g., 'cuda' or 'cpu')
        amp (bool): Bool indicating if automatic mixed precision is used
        log_images (bool): Bool indicating if we want to write images to tensorboard (from the validation set)
        val_set (torch.utils.data.Dataset): The Validation Dataset, from which images for logging are taken
        global_step (int): The global training step used for logging.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer instance for image logging

    Returns:
        float: The average IoU score across the dataset
    """

    net.eval()
    num_val_batches = len(dataloader)
    iou_score = 0


    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

        
            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                
                #has shape BxCxHxW (so e.g. 16x1x256x256)
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                #from BxCxHxW to BxHxW, so we do not have an unnecessary dimension before one hot encoding
                mask_pred = mask_pred.squeeze(1)



                # convert to one-hot format
                #mask get both converted from B, H, W to B,H,W,2, which is reordered to B,2,H,W to stick to the B,C,H,W convention
                mask_true = F.one_hot(mask_true.long(), 2).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.long(), 2).permute(0, 3, 1, 2).float()

                # compute iou score
                iou_score += iou(mask_pred, mask_true)

        
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the iou  score, including background
                iou_score += iou(mask_pred, mask_true)

        if log_images == True:
          assert global_step is not None and writer is not None, 'Writer and a global training step have to be provided for tensorboard logging'

          ####### Log 5 interesting images from the validation set
          interesting_val_images = [48, 47, 60, 61, 64]
          


          manual_batch_images = []
          manual_batch_masks = []
          for i in interesting_val_images:
            manual_batch_images.append(val_set[i]['image'])
            manual_batch_masks.append(val_set[i]['mask'])

          manual_batch_images = torch.stack(manual_batch_images)
          manual_batch_masks = torch.stack(manual_batch_masks)


          image = manual_batch_images.to(device=device, dtype=torch.float32, memory_format= torch.channels_last)
          mask_true = manual_batch_masks.to(device=device, dtype= torch.long)

          mask_pred = net(image)
          # Transform from (B,C,H,W) to (B,H,W), where B is the batch size so in this case len(interesting_val_images)
          mask_pred = mask_pred.argmax(dim=1) 

          #plot the first five
          for i in range(len(interesting_val_images)):
            #pass tensor of img (CxHxW), mask_true (HxW) and mask_predict (HxW) for plotting
            images_mask_tb_landcover(writer, image[i].cpu(), mask_true[i].cpu(), mask_pred[i].cpu(), global_step = global_step, tag = f"Image {i}")




    net.train()
    return iou_score / max(num_val_batches, 1)

#endregion





##############
##  FLOODS
##############
#region




#Source: https://github.com/milesial/Pytorch-UNet/blob/master/evaluate.py
#Adapted (Replaced dice score with IoU, added the image logging with tensorboard, adaptions for loading pre and post images)
@torch.inference_mode()
def evaluate_floods(net, dataloader, device, amp,log_images=False,val_set=None, global_step=None, writer=None):
    """
    Evaluate the performance of a neural network on the S1GFLOOD dataset using the IoU metric

    Args:
        net (torch.nn.Module): The neural net to evaluate
        dataloader (torch.utils.data.DataLoader): The dataloader for the dataset
        device (torch.device): The device to run the evaluation on (e.g., 'cuda' or 'cpu')
        amp (bool): Bool indicating if automatic mixed precision is used
        log_images (bool): Bool indicating if we want to write images to tensorboard (from the validation set)
        val_set (torch.utils.data.Dataset): The Validation Dataset, from which images for logging are taken
        global_step (int): The global training step used for logging.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer instance for image logging

    Returns:
        float: The average IoU score across the dataset
    """
    net.eval()
    num_batches = len(dataloader)
    iou_score = 0


    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_batches, desc='Validation round', unit='batch', leave=False,disable=True):


            image_pre,image_post, mask_true = batch['image_pre'], batch['image_post'], batch['mask']



            # move images and labels to correct device and type
            image_pre = image_pre.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            image_post = image_post.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image_pre,image_post)





            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'

                #has shape BxCxHxW (so e.g. 16x1x256x256)
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                #from BxCxHxW to BxHxW, so we do not have an unnecessary dimension before one hot encoding
                mask_pred = mask_pred.squeeze(1)



                # convert to one-hot format
                #mask get both converted from B, H, W to B,H,W,2, which is reordered to B,2,H,W to stickt to the B,C,H,W convention
                mask_true = F.one_hot(mask_true.long(), 2).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.long(), 2).permute(0, 3, 1, 2).float()

                # compute iou score
                iou_score += iou(mask_pred, mask_true)



            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the iou  score, including background
                iou_score += iou(mask_pred, mask_true)


        if log_images == True:
          assert global_step is not None and writer is not None, 'Writer and a global training step have to be provided for tensorboard logging'

          ####### Log interesting images from the validation set
          interesting_val_images = [0,1,3,4,7,16,20,25,33]
          
          manual_batch_images_pre = []
          manual_batch_images_post = []
          manual_batch_masks = []
          for i in interesting_val_images:
            manual_batch_images_pre.append(val_set[i]['image_pre'])
            manual_batch_images_post.append(val_set[i]['image_post'])
            manual_batch_masks.append(val_set[i]['mask'])

          manual_batch_images_pre = torch.stack(manual_batch_images_pre)
          manual_batch_images_post = torch.stack(manual_batch_images_post)
          manual_batch_masks = torch.stack(manual_batch_masks)


          image_pre = manual_batch_images_pre.to(device=device, dtype=torch.float32, memory_format= torch.channels_last)
          image_post = manual_batch_images_post.to(device=device, dtype=torch.float32, memory_format= torch.channels_last)
          mask_true = manual_batch_masks.to(device=device, dtype= torch.long)

          mask_pred = net(image_pre, image_post)
          mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

          #From B x1 (C)x H x W -> BxHxW
          mask_pred = torch.squeeze(mask_pred,1)


          #plot the first five
          for i in range(len(interesting_val_images)):
            #pass tensor of img_pre (CxHxW), img_post (CxHxW), mask_true (HXW), mask_pred (HxW)
            images_mask_tb_floods(writer, image_pre[i].cpu(),image_post[i].cpu(), mask_true[i].cpu(), mask_pred[i].cpu(), global_step = global_step, tag = f"Image {i}")






    net.train()
    return iou_score / max(num_batches, 1)



#endregion