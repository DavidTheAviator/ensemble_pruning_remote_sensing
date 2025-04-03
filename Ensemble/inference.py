import torch


##############
##  GENERAL
##############
#region


#Source: https://github.com/milesial/Pytorch-UNet/blob/master/predict.py
#Adapted (added landcover/flood logic)
@torch.inference_mode()
def predict_img(net,
                imgs,
                device,
                out_threshold=0.5, 
                dataset_type = 'landcover'
                ):
    """
    Generate a predicted segmentation mask for an input image (Landcover) or image pair (S1GFloods) using a trained neural network.

    Args:
        net (torch.nn.Module): The neural network used for prediction.
        imgs (numpy.ndarray or tuple of numpy.ndarray): The input image(s). A single image for 'landcover' datasets 
            or a tuple (pre-event, post-event) for S1GFloods
        device (torch.device): The device to run inference on (e.g., 'cuda' or 'cpu').
        out_threshold (float, optional): Threshold for converting sigmoid outputs to binary masks. Defaults to 0.5.
        dataset_type (str, optional): The type of dataset, either 'landcover' (single-image input) or 'floods'. Defaults to 'landcover'.

    Returns:
        numpy.ndarray: The predicted segmentation mask with the same spatial dimensions as the input image
    """


    landcover = True if dataset_type == 'landcover' else False
    
    net.eval()
    
    if landcover:
        img = torch.from_numpy(imgs)
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)



       
        # has shape BxCxHxW
        output = net(img).cpu()
        if net.n_classes > 1:
            
            mask = output.argmax(dim=1)
      
        else:
            mask = torch.sigmoid(output) > out_threshold

    else:
        image_pre,image_post= torch.from_numpy(imgs[0]), torch.from_numpy(imgs[1])
        image_pre,image_post = image_pre.unsqueeze(0), image_post.unsqueeze(0)
        image_pre,image_post = image_pre.to(device=device, dtype=torch.float32), image_post.to(device=device, dtype=torch.float32)

       
        # has shape BxCxHxW
        output = net(image_pre, image_post).cpu()

        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:

           
            mask = torch.sigmoid(output) > out_threshold
        
    net.train()

    return mask[0].long().squeeze().numpy()








#endregion

    