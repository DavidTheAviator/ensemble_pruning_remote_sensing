import os
import textwrap
import torch
import torch.nn as nn
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm


from .plotting import images_mask_tb_landcover, images_mask_tb_floods
from .evaluation import evaluate_landcover, evaluate_floods






##############
##  LANDCOVER
##############
#region


#Source: https://github.com/milesial/Pytorch-UNet/blob/master/train.py
#Adapted (Added checkpoints for resuming training, Tensorboard Logging instead of weights and biases, Changed optimizer and lr-scheduler)
def train_model_landcover(
        run_name,
        model,
        device,
        train_set,
        val_set,
        writer,
        epochs: int = 25,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-6,
        momentum: float = 0.9,
        t0: float = 10,
        checkpoint_list: list = None,
        checkpoint_path: str = None,
        amp: bool = False,
        weight_decay: float = 1e-8,
        gradient_clipping: float = 1.0,
):
    """
    Trains a neural network model for the LandCover.aI dataset using a training dataset and evaluates it on a 
    validation dataset. Supports checkpointing, TensorBoard logging, mixed precision training, and learning rate scheduling.

    Args:
        run_name (str): Name of the training run (used for checkpoint directory naming).
        model (torch.nn.Module): The neural network model to train.
        device (torch.device): The device to run training on (e.g., 'cuda' or 'cpu').
        train_set (torch.utils.data.Dataset): The training dataset.
        val_set (torch.utils.data.Dataset): The validation dataset.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer instance for logging.
        epochs (int, optional): Number of training epochs (default: 25).
        batch_size (int, optional): Batch size for training (default: 8).
        learning_rate (float, optional): Initial learning rate (default: 1e-3).
        min_learning_rate (float, optional): Minimum learning rate for the cosine annealing scheduler (default: 1e-6).
        momentum (float, optional): Momentum factor for the SGD optimizer (default: 0.9).
        t0 (float, optional): Initial restart period for the cosine annealing scheduler (default: 10).
        checkpoint_list (list, optional): List of epochs at which to save model checkpoints (default: None).
        checkpoint_path (str, optional): Directory path for saving checkpoints (default: None).
        amp (bool, optional): Whether to use automatic mixed precision training (default: False).
        weight_decay (float, optional): Weight decay coefficient for regularization (default: 1e-8).
        gradient_clipping (float, optional): Maximum gradient norm for gradient clipping (default: 1.0).

    Returns:
        None
    """
    

    #pre calculated class counts
    weights = torch.FloatTensor([0.00860509, 0.58491337, 0.01502485, 0.08228819, 0.3091685])
    weights = weights.to(device=device, dtype=torch.float32)



    #length of the data sets
    n_train = len(train_set)
    n_val = len(val_set)

    #data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args) 



    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,  weight_decay=weight_decay, nesterov=True)

    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, t0, T_mult=1, eta_min=min_learning_rate)
    grad_scaler = torch.GradScaler(str(device), enabled=amp)
    criterion = nn.CrossEntropyLoss(weight=weights) if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    

    params_run = textwrap.dedent(f'''
    ### Run Parameters

    | Parameter          | Value          |
    |--------------------|----------------|
    | Epochs             | {epochs}       |
    | Batch size         | {batch_size}   |
    | Learning rate      | {learning_rate}|
    | Training size      | {n_train}      |
    | Validation size    | {n_val}        |
    | Checkpoints        | {checkpoint_list} |
    | Device             | {device.type}  |
    | Mixed Precision    | {amp}          |

    ''')


    writer.add_text("Parameters",params_run,0)
    writer.flush()






    #Log the predictions on 5 interesting images (in each validation round):
    interesting_val_images = [48, 47, 60, 61, 64]



    #add the images to tensorboard before training
    for j,idx in enumerate(interesting_val_images):
      sample = val_set[idx]
      image = sample['image']
      mask_true = sample['mask']
      
      #pass tensor of img (CxHxW), mask_true (HxW) and mask_predict (HxW) for plotting
      images_mask_tb_landcover(writer, image, mask_true, global_step = global_step, tag = f"Image {j}")
    






    # Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        iters = len(train_loader)
        i = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)

                    if model.n_classes == 1:
                        #true_masks has shape BxHxW, so we squeeze masks_pred
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    else:
                        loss = criterion(masks_pred, true_masks)


                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                #Scheduler
                scheduler.step(epoch + i / iters)
                i += 1

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()



                #Logging Train Loss
                writer.add_scalar('Train Loss', loss.item(), global_step=global_step, double_precision=True)
                writer.add_scalar('Learning Rate', scheduler.get_last_lr()[-1], global_step=global_step, double_precision=True)
                writer.flush()


                pbar.set_postfix(**{'loss (batch)': loss.item()})



        #after every epoch make an evaluation round  (which also plots images)
        val_score = evaluate_landcover(model, val_loader, device, amp,log_images=True,val_set=val_set, global_step=global_step, writer=writer)
       



        #Logging
        #Val Score (IoU)
        #Epoch Loss
        writer.add_scalar('Epoch Loss', epoch_loss/len(train_loader),global_step=global_step, double_precision=True)
        writer.add_scalar('Validation Score', val_score, global_step=global_step, double_precision=True)
        writer.flush()



        if checkpoint_list and  checkpoint_path and epoch in checkpoint_list:
            checkpoint_path = Path(checkpoint_path)

            #specifies the path of the directory of this specific run
            run_path = Path(checkpoint_path).joinpath(run_name)

            #create a directory for this run if it does not exist already
            run_path.mkdir(parents=True, exist_ok=True)

            #create a checkpoint object
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': grad_scaler.state_dict(),
                'lr_sched': scheduler
            }

            #save the checkpoint object
            checkpoint_name = f'checkpoint_epoch{epoch}.pth'
            torch.save(checkpoint,run_path.joinpath(checkpoint_name))

#endregion






##############
##  FLOODS
##############
#region



#Source: https://github.com/milesial/Pytorch-UNet/blob/master/train.py
#Adapted (Added checkpoints for resuming training, Tensorboard Logging instead of weights and biases, changed optimizer, updated for change detection)
def train_model_floods(
        run_name,
        model,
        device,
        train_set,
        val_set,
        writer,
        epochs: int = 25,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        checkpoint_list: list = None,
        checkpoint_path: str = None,
        amp: bool = False,
        weight_decay: float = 1e-8,
        gradient_clipping: float = 1.0,
):
    """
    Trains a Fully Convolutional Siamese-Concat model for change detection, specifically for flood detection on S1GFloods. 
    The model is trained on a given dataset and evaluated on a validation set.

    Args:
        run_name (str): Name of the training run, used for organizing checkpoints.
        model (torch.nn.Module): The neural network model to train.
        device (torch.device): The device for training (e.g., 'cuda' or 'cpu').
        train_set (torch.utils.data.Dataset): The dataset used for training.
        val_set (torch.utils.data.Dataset): The dataset used for validation.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for logging images and metrics.
        epochs (int, optional): Number of training epochs (default: 25).
        batch_size (int, optional): Batch size for training (default: 16).
        learning_rate (float, optional): Initial learning rate (default: 1e-3).
        checkpoint_list (list, optional): List of epochs at which model checkpoints should be saved (default: None).
        checkpoint_path (str, optional): Directory path for saving model checkpoints (default: None).
        amp (bool, optional): Enables automatic mixed precision training if True (default: False).
        weight_decay (float, optional): Weight decay regularization factor (default: 1e-8).
        gradient_clipping (float, optional): Maximum gradient norm for gradient clipping (default: 1.0).

    Returns:
        None
    """

    #classes: un-flooded (0), flooded (1), (pre-calculated)
    non_flooded_pix = 190097174 #0 label
    flooded_pix = 91707626  #1 label


    #should be set like this according to docs https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
    #experiments showed it's better not to use it
    pos_weight= torch.tensor(non_flooded_pix/flooded_pix)


    #length of the data sets
    n_train = len(train_set)
    n_val = len(val_set)

    #data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False,  **loader_args)



    # Set up the optimizer, the loss, and the loss scaling for AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=True)
    grad_scaler = torch.GradScaler(str(device), enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    params_run = textwrap.dedent(f'''
    ### Run Parameters

    | Parameter          | Value          |
    |--------------------|----------------|
    | Epochs             | {epochs}       |
    | Batch size         | {batch_size}   |
    | Learning rate      | {learning_rate}|
    | Training size      | {n_train}      |
    | Validation size    | {n_val}        |
    | Checkpoints        | {checkpoint_list} |
    | Device             | {device.type}  |
    | Mixed Precision    | {amp}          |

    ''')


    writer.add_text("Parameters",params_run,0)
    writer.flush()




    #Log the predictions on a few interesting images (in each validation round):
    interesting_val_images = [0,1,3,4,7,16,20,25,33]



    for j,idx in enumerate(interesting_val_images):

      sample = val_set[idx]
      image_pre = sample['image_pre']
      image_post = sample['image_post']
      mask_true = sample['mask']
      #pass tensor of img_pre (CxHxW),img_post (CxHxW) and mask_true (HxW) for plotting
      images_mask_tb_floods(writer, image_pre.cpu(), image_post.cpu(), mask_true.cpu(), global_step = global_step, tag = f"Image {j}")





    # Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        iters = len(train_loader)
        i = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

                images_pre,images_post, masks_true = batch['image_pre'], batch['image_post'],batch['mask']

                assert images_pre.size() == images_post.size(), \
                    f'images_pre has shape: {images_pre.size()} while images_post has shape: {images_post.size()}'


                assert images_pre.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images_pre.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images_pre = images_pre.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                images_post = images_post.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                masks_true = masks_true.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images_pre,images_post)

                    if model.n_classes == 1:
                        #masks_true has shape BxHxW, so we squeeze masks_pred
                        loss = criterion(masks_pred.squeeze(1), masks_true.float())
                    else:

                        loss = criterion(masks_pred, masks_true)



                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()


                

                pbar.update(images_pre.shape[0])
                global_step += 1
                epoch_loss += loss.item()









                #Logging Train Loss 
                writer.add_scalar('Train Loss', loss.item(), global_step=global_step, double_precision=True)
                writer.flush()


                pbar.set_postfix(**{'loss (batch)': loss.item()})



        #after every epoch make an evaluation round  (which also plots images)
        val_score = evaluate_floods(model, val_loader, device, amp,log_images=True,val_set=val_set, global_step=global_step, writer=writer)
        






        #Logging
        #Learning Rate, Val Score, Epoch Loss
        writer.add_scalar('Epoch Loss', epoch_loss/len(train_loader),global_step=global_step, double_precision=True)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'],global_step=global_step, double_precision=True)
        writer.add_scalar('Validation Score', val_score, global_step=global_step, double_precision=True)
        writer.flush()



        if checkpoint_list and  checkpoint_path and epoch in checkpoint_list:
            checkpoint_path = Path(checkpoint_path)

            #specifies the path of the directory of this specific run
            run_path = Path(checkpoint_path).joinpath(run_name)

            #create a directory for this run if it does not exist already
            run_path.mkdir(parents=True, exist_ok=True)

            #create a checkpoint object
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': grad_scaler.state_dict()
               # 'lr_sched': scheduler
            }

            #save the checkpoint object
            checkpoint_name = f'checkpoint_epoch{epoch}.pth'
            torch.save(checkpoint,run_path.joinpath(checkpoint_name))

#endregion






