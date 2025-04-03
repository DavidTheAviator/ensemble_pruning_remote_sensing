import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import sys
import os 

# Add Parent Directory to Path
project_root_dir = os.path.join(os.path.dirname(__file__), "..")
project_root_dir = os.path.abspath(project_root_dir)

sys.path.append(project_root_dir)

# Ensemble imports
from Ensemble.models import SiamUnet_conc, UNet
from .normalization import denormalizer, denormalizer_post, denormalizer_pre
from .inference import predict_img
from .models import UNet, SiamUnet_conc


##############
##  GENERAL
##############
#region


def int_to_rgb(mask, cmap):
    """
    Converts a mask to a 3 channel image, with colors indicated in cmap.

    Args:
      mask (numpy.ndarray): Mask of shape (H,W) , that is filled with integers representing classes.
      cmap (dict): A dictionary representing the color coding for each class, the key is the class label, the value an RGB-tuple

    Returns:
      numpy.ndarray: An array/image of shape (H,W,3), where each clas label is colored according to the color map
    """


    img_colored = np.stack([mask,mask,mask],axis=2)

    #iterate through the class color array
    for class_int, rgb in cmap.items():
        b_mask = mask == class_int

        #on the first channel, write the first color from the tuple, same goes for the other channels
        img_colored[b_mask,0] = rgb[0]
        img_colored[b_mask,1] = rgb[1]
        img_colored[b_mask,2] = rgb[2]

    return img_colored



def plot_img_mask_pred(imgs, mask_true, mask_pred, cmap, dataset_type = 'landcover'):
    """
    Plots an image (or two images),the corresponding mask and the model prediction side by side.

    Args:
      img (numpy.ndarray/Tuple(numpy.ndarray)): Array(s) of shape (H,W,3) representing an image
      mask_true (numpy.ndarray): Mask of shape (H,W) , that is filled with integers representing classes 
      mask_pred (numpy.ndarray): Mask predicted by a model of shape (H,W), that is filled with integers representing classes 
      cmap (dict): A dictionary representing the color coding for each class, the key is the class label, the value an RGB-tuple

    Returns
      None: The figures are just plotted.
    """

    landcover = True if dataset_type == 'landcover' else False

    if landcover:
      mask_true = int_to_rgb(mask_true,cmap)
      mask_pred = int_to_rgb(mask_pred, cmap)


      fig, axs = plt.subplots(1,3)
      im1 = axs[0].imshow(imgs)
      im2 = axs[1].imshow(mask_true)
      im3 = axs[2].imshow(mask_pred)
      axs[0].set_title('Image')
      axs[1].set_title('True')
      axs[2].set_title('Prediction')
    
    else:
      mask_true = int_to_rgb(mask_true,cmap)
      mask_pred = int_to_rgb(mask_pred, cmap)


      fig, axs = plt.subplots(1,4)
      im1 = axs[0].imshow(imgs[0])
      im2 = axs[1].imshow(imgs[1])
      im2 = axs[2].imshow(mask_true)
      im3 = axs[3].imshow(mask_pred)
      axs[0].set_title('Image PRE')
      axs[1].set_title('Image POST')
      axs[2].set_title('True')
      axs[3].set_title('Prediction')
       






def plot_images_dataset(dataset, img_list, model_path, dataset_type='landcover'):
    """
    Loads a trained model and visualizes predictions for a given list of images.

    Args:
      dataset (Dataset): A dataset object containing images and corresponding masks.
      img_list (list of int): List of indices specifying which images to plot.
      model_path (str): Path to the trained model file.
      dataset_type (str, optional): Type of dataset ('landcover' or 'floods'). Defaults to 'landcover'.

    Returns:
      None: The function generates and plots the predicted masks.
    """
    
    landcover = True if dataset_type == 'landcover' else False


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if landcover:
        model_l = UNet(n_channels=3,n_classes=5)
    else:
        model_l = SiamUnet_conc(n_channels=3,n_classes=1)
    
    model_l.load_state_dict(torch.load(model_path,map_location=device))


    for i in img_list:

        sample = dataset[i]

        if landcover:
            img, mask_true = sample['image'].numpy(), sample['mask'].numpy()

            mask_predict = predict_img(model_l, img, device, dataset_type='landcover')


            #do this here, otherwise wrong input gets fed to the models
            img = denormalizer(torch.from_numpy(img)).numpy().astype('int_')

            plot_img_mask_pred(np.moveaxis(img,0,-1), mask_true, mask_predict, cmap_landcover, dataset_type = 'landcover')        
        else:
          

            img_pre,img_post, mask_true = sample['image_pre'].numpy(), sample['image_post'].numpy(), sample['mask'].numpy()

            mask_predict = predict_img(model_l, (img_pre,img_post), device, out_threshold=0.5, dataset_type='floods')

            

            img_pre = denormalizer_pre(torch.from_numpy(img_pre)).numpy().astype('uint8')
            img_post = denormalizer_post(torch.from_numpy(img_post)).numpy().astype('uint8')

            img_pre = np.moveaxis(img_pre,0,-1)
            img_post = np.moveaxis(img_post,0,-1)


            plot_img_mask_pred((img_pre,img_post), mask_true, mask_predict, cmap_floods, dataset_type = 'floods')  
            




def plot_images_dataset_ensemble(dataset,img_list, model_paths, selected_models,fusion_method, dataset_type='landcover'):
    """
    Loads multiple trained models and visualizes ensemble predictions for a given list of images.

    Args:
      dataset (Dataset): A dataset object containing images and corresponding masks.
      img_list (list[int]): List of indices specifying which images to plot.
      model_paths (list[str]): List of file paths to the trained models.
      selected_models (list[int]): List of indices specifying which models to use from model_paths.
      fusion_method (str): The method used to combine model predictions ('vote' for majority voting or 'fusion' for posterior averaging).
      dataset_type (str, optional): Type of dataset ('landcover' or 'floods'). Defaults to 'landcover'.

    Returns:
      None: The function generates and plots the ensemble predictions.
    
    Raises:
      ValueError: If an unsupported fusion method is provided.
      """
    
    landcover = True if dataset_type == 'landcover' else False


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if landcover:
        model_l = UNet(n_channels=3,n_classes=5)
    else:
        model_l = SiamUnet_conc(n_channels=3,n_classes=1)
    
    
    model_paths = [model_paths[i] for i in selected_models]

    models = []
   
    for model_path in model_paths:
        model_l = UNet(n_channels=3,n_classes=5) if landcover else SiamUnet_conc(n_channels=3,n_classes=1)
        model_l.load_state_dict(torch.load(model_path, map_location=device))
        model_l.to(device)
        model_l.eval()
        models.append(model_l)


    for i in img_list:

        sample = dataset[i]

        if landcover:
            img, mask_true = sample['image'].unsqueeze(0), sample['mask'].numpy()

            

            img = img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
           

            mask_pred = [F.softmax(model(img),dim=1) for model in models]
            
            print(f"shape of first list el: {mask_pred[0].shape}")
            
            #majority voting; from https://discuss.pytorch.org/t/majority-voting/207260, not adapted
            if(fusion_method == 'vote'):
                #shape B,H,W
                class_predictions = [torch.argmax(output, dim=1) for output in mask_pred] 
                
                #shape M,B,H,W
                stacked_predictions = torch.stack(class_predictions)

                #shape B,H,W
                fused_pred, _ = torch.mode(stacked_predictions, dim=0)

            
            elif(fusion_method == 'fusion'):
                #posterior averaging
                #ensemble preds has shape (M,B, C, H, W), where M is the number of selected models 
                ensemble_preds = torch.stack(mask_pred)

                #shape (B, C, H, W)
                fused = ensemble_preds.mean(dim=0)

                fused_pred = fused.argmax(dim=1)

            else:
                raise ValueError(f"Unsupported fusion_method: {fusion_method}. Please use 'vote' or 'fusion'.")
            

          

            
            #do this here, otherwise wrong input gets fed to the models
            img = denormalizer(img.squeeze(0)).numpy().astype('int_')

            plot_img_mask_pred(np.moveaxis(img,0,-1), mask_true, fused_pred.squeeze().numpy(), cmap_landcover, dataset_type = 'landcover')        
        else:
          

            img_pre,img_post, mask_true = sample['image_pre'].unsqueeze(0), sample['image_post'].unsqueeze(0), sample['mask'].numpy()



            # move images and labels to correct device and type
            img_pre = img_pre.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            img_post = img_post.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
           


            #make a prediction with every model
            mask_pred = [F.sigmoid(model(img_pre,img_post)) for model in models]

            temp = []
            
            #transform every element in mask predict to shape B,C,H,W, where C=2 and holds the probabilities for label 0 and 1
            for pred in mask_pred:
                ones = torch.ones_like(pred)
                proba_unflooded = torch.sub(ones, pred)

                pred = torch.cat([proba_unflooded,pred],axis=1)
                temp.append(pred)
            

            mask_pred = temp


            #majority voting; from https://discuss.pytorch.org/t/majority-voting/207260, not adapted
            if(fusion_method == 'vote'):
                #shape B,H,W
                class_predictions = [torch.argmax(output, dim=1) for output in mask_pred] 
                
                #shape M,B,H,W
                stacked_predictions = torch.stack(class_predictions)

                #shape B,H,W
                fused_pred, _ = torch.mode(stacked_predictions, dim=0)

            
            elif(fusion_method == 'fusion'):
                #posterior averaging
                #ensemble preds has shape (M,B, C, H, W), where M is the number of selected models 
                ensemble_preds = torch.stack(mask_pred)

                #shape (B, C, H, W)
                fused = ensemble_preds.mean(dim=0)

                fused_pred = fused.argmax(dim=1)

            else:
                raise ValueError(f"Unsupported fusion_method: {fusion_method}. Please use 'vote' or 'fusion'.")
      
            img_pre = denormalizer_pre(img_pre.squeeze(0)).numpy().astype('uint8')
            img_post = denormalizer_post(img_post.squeeze(0)).numpy().astype('uint8')

            img_pre = np.moveaxis(img_pre,0,-1)
            img_post = np.moveaxis(img_post,0,-1)


            plot_img_mask_pred((img_pre,img_post), mask_true, fused_pred.squeeze().numpy(), cmap_floods, dataset_type = 'floods')  
            


#endregion        



##############
##  LANDCOVER
##############
#region

#Color Coding for plotting
cmap_landcover = {0: (220,220,220), # grey, unlabeled
1: (250,128,114),         # red, building
2: (0,255,127),           # green, woodland
3: (0,191,255),           # blue, water
4:(255,255,0)             # yellow, road
}



def plot_img_mask(img, mask, cmap):  
    """
    Plots an image and the corresponding mask side by side.

    Args:
      img (numpy.ndarray): Array of shape (H,W,3) representing an image
      mask (numpy.ndarray): Mask of shape (H,W) , that is filled with integers representing classes (corresponds to the image)
      cmap (dict): A dictionary representing the color coding for each class, the key is the class label, the value an RGB-tuple

    Returns
      None: The figures are just plotted.
    """

    mask = int_to_rgb(mask, cmap)
    fig, axs = plt.subplots(1, 2)
    im1 = axs[0].imshow(img)
    im2 = axs[1].imshow(mask)
    plt.show()




def images_mask_tb_landcover(writer, img, mask_true, mask_predict=None, global_step=None,tag=None):
    """
    Passes an image, a mask (and a mask predicted by the model) to tensorboard for logging

    Args:
      writer (torch.utils.tensorboard.SummaryWriter): The tensorboard writer instace used for logging.
      img (torch.Tensor): Tensor of shape (C,H,W), which is normalized
      mask_true (torch.Tensor, optional): Tensor of shape (H,W), which containts the class labels
      mask_true (torch.Tensor, optional): Tensor of shape (H,W), which containts the predicted class labels
      global_step (int, optional): The current global training step 
      tag (string, optional): The tensorboard tag

    Returns:
      None: The images are just passed to tensorboard for logging.
    """


   
    #because we normalize pixel values in the dataset, we have to denormalize here
    img = denormalizer(img).numpy().astype('uint8')


    #handle the mask tensor
    mask_true = mask_true.numpy()

    #transfer to rgb (and to uint8)
    mask_true= int_to_rgb(mask_true,cmap_landcover).astype('uint8')

    #move channels to first dim (for tensorboard)
    mask_true = np.moveaxis(mask_true, -1, 0)





    if mask_predict is not None:
      #same procedure as with mask true
      mask_predict = np.moveaxis(int_to_rgb(mask_predict.numpy(),cmap_landcover).astype('uint8'),-1,0)
      imgs = np.stack([img,mask_true,mask_predict])
    else:
      imgs = np.stack([img,mask_true])


    writer.add_images(tag,imgs,global_step=global_step)
    writer.flush()

#endregion


##############
##  FLOODS
##############
#region

#Color Coding for plotting
cmap_floods = {0: (220,220,220), # grey, unlabeled
        1: (0,191,255), # blue, flooded
}

def images_mask_tb_floods(writer, img_pre,img_post, mask_true, mask_predict=None, global_step=None,tag=None):
    """
    Passes two images (pre- and post- flood), a mask (and a mask predicted by the model) to tensorboard for logging

    Args:
      writer (torch.utils.tensorboard.SummaryWriter): The tensorboard writer instace used for logging
      img_pre (torch.Tensor): Tensor of shape (C,H,W), which has been normalized (before flooding)
      img_post (torch.Tensor): Tensor of shape (C,H,W), which has been normalized (after flooding)
      mask_true (torch.Tensor, optional): Tensor of shape (H,W), which containts the class labels
      mask_true (torch.Tensor, optional): Tensor of shape (H,W), which containts the predicted class labels
      global_step (int, optional): The current global training step 
      tag (string, optional): The tensorboard tag

    Returns:
      None: The images are just passed to tensorboard for logging.
    """
    #because we normalize pixel values in the dataset, we have to denormalize here
    img_pre = denormalizer_pre(img_pre).numpy().astype('uint8')
    img_post = denormalizer_post(img_post).numpy().astype('uint8')


    #handle the mask tensor
    mask_true = mask_true.numpy()


    #transfer to rgb (and to uint8)
    mask_true= int_to_rgb(mask_true,cmap_floods).astype('uint8')

    #move channels to first dim
    mask_true = np.moveaxis(mask_true, -1, 0)





    if mask_predict is not None:
      #same procedure as with mask true
      mask_predict = np.moveaxis(int_to_rgb(mask_predict.numpy(),cmap_floods).astype('uint8'),-1,0)
      imgs = np.stack([img_pre,img_post, mask_true,mask_predict])
    else:
      imgs = np.stack([img_pre,img_post, mask_true])


    writer.add_images(tag,imgs,global_step=global_step)
    writer.flush()

  #endregion