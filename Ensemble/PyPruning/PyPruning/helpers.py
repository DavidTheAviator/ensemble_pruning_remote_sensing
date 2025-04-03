from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
import torch
import sys
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_f1_score
import numpy as np
import re


# Add Parent Directory to Path
project_root_dir = os.path.join(os.path.dirname(__file__), "..","..")
project_root_dir = os.path.abspath(project_root_dir)

sys.path.append(project_root_dir)

# Ensemble imports
from Ensemble.models import SiamUnet_conc, UNet
from Ensemble.evaluation import iou


class ProbaTargetLoader:
    """
    A data loader for iterating over probability and target arrays (which are then used for pruning)

    This class is used to load pairs of probability and target arrays from a given directory. 
    The arrays are expected to be stored in files starting with 'ensemble' (for probabilities) 
    and 'target' (for targets). The loader iterates through the matching files in sorted order.

    Args:
        dir_path (str or Path): Directory path containing the 'ensemble' and 'target' files.

    Attributes:
        probas (list): Sorted list of paths to the probability files.
        targets (list): Sorted list of paths to the target files.
        low (int): Starting index for the iteration (always 0).
        current (int): Current index for iteration, initialized to -1.
        high (int): Total number of pairs to iterate over.
        
    Methods:
        __iter__(): Initializes the iterator.
        __next__(): Returns the next pair of probability and target arrays.
        __len__(): Returns the total number of probability-target pairs available.
    """

    def __init__(self,dir_path):
        """
        Initializes the ProbaTargetLoader by loading and sorting the file paths for 
        probability and target arrays from the specified directory.

        Args:
            dir_path (str or Path): Directory containing the 'ensemble' and 'target' files.
        """
        
        l = os.listdir(Path(dir_path))
        self.probas = sorted([dir_path.joinpath(i) for i in l if i.startswith('ensemble')])
        self.targets = sorted([dir_path.joinpath(i) for i in l if i.startswith('target')]) 
        self.low = 0
        self.current = -1
        self.high = len(self.probas)

    def __iter__(self):
        """
        Initializes the iterator for the ProbaTargetLoader.

        Returns:
            ProbaTargetLoader: The iterator object itself.
        """
        return self

    def __next__(self):
        """
        Retrieves the next pair of probability and target arrays.

        Returns:
            tuple: A tuple containing the next probability array and its corresponding target array.

        Raises:
            StopIteration: When all pairs have been iterated over.
        """
        self.current += 1
        if self.current == self.high:
            raise StopIteration
        proba = np.load(self.probas[self.current])
        target = np.load(self.targets[self.current])
        return proba,target

    def __len__(self):
        """
        Returns the total number of probability-target pairs available.

        Returns:
            int: The number of available pairs.
        """
        return self.high


def get_model_paths(experiment_path):
    """
    Returns sorted paths to the saved model files in the 'saved_models' directory of the specified experiment (i.e., folder structure of an initial ensemble)

    Args:
        experiment_path (str or Path): Path to the experiment directory.

    Returns:
        list: Sorted list of model file paths.
    """
    experiment_path = Path(experiment_path)


    models_unsorted = os.listdir(experiment_path.joinpath('saved_models'))
    model_names = sorted(models_unsorted, key=lambda x: int(re.search(r'model_(\d+)', x).group(1)))
    model_paths = [experiment_path.joinpath('saved_models').joinpath(model_name) for model_name in model_names]
    return model_paths



@torch.inference_mode()
def generate_ensemble_proba_target(model_paths, dataset,batch_size, amp, dataset_type = 'landcover'):
    """
    Generates ensemble probabilities and corresponding targets for a given dataset using multiple models.

    Args:
        model_paths (list): List of file paths to the trained models.
        dataset (Dataset): The dataset to generate predictions on.
        batch_size (int): The number of samples per batch.
        amp (bool): Whether to use automatic mixed precision for inference.
        dataset_type (str): The type of dataset ('landcover' or 'floods') to then adjust model architecture.

    Returns:
        tuple: A tuple containing:
            - Tensor: Ensemble probabilities for all models (shape: [num_models, N, C]).
            - Tensor: Flattened target values (shape: [N]).
    """


    #indicate if the dataset is generated from landcover or S1GFLOOD
    if dataset_type == 'landcover':
        landcover = True
    else:
        landcover = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #make a list of all the models
    models = []
   
    for model_path in model_paths:
        model_l = UNet(n_channels=3,n_classes=5) if landcover else SiamUnet_conc(n_channels=3,n_classes=1)
        model_l.load_state_dict(torch.load(model_path, map_location=device))
        #move to cpu initially and always just load for inference
        model_l.to('cpu')
        model_l.eval()
        models.append(model_l)


    dataloader = DataLoader(dataset, shuffle=False,drop_last=False,batch_size=batch_size,num_workers = os.cpu_count(),pin_memory=True)
    num_batches = len(dataloader)


    ensemble_proba = []
    target = []
    # iterate over the dataset
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_batches, desc='Generating Ensemble Proba', unit='batch', leave=False):

            if landcover:
                image, mask_true = batch['image'], batch['mask']

                # move images and labels to correct device and type
                image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                
                #make a prediction with every model
                mask_pred = []
                for model in models:
                    model.to(device)
                    
                    #model(image) is of format B, C, H, W, ; C has the probabilities
                    #softmax is apllied since Unet implementation outputs logits
                    pred = F.softmax(model(image),dim=1)
                    #convert from shape (B,C,H,W), to (B,H,W,C) and finally to (N,C), where N = B*W*H
                    pred = pred.permute(0,2,3,1).flatten(0,2)
                    mask_pred.append(pred)
                    model.to('cpu')
                

                
              
            

                
            
            else:
                image_pre,image_post, mask_true = batch['image_pre'], batch['image_post'], batch['mask']


                # move images and labels to correct device and type
                image_pre = image_pre.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                image_post = image_post.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
             

                

                #make a prediction with every model
                mask_pred = []
                for model in models:
                    model.to(device)
                    pred = F.sigmoid(model(image_pre,image_post))
                    
                    #create Tensor of ones, also of shape B,1,H,W; in this Tensor the probability that the pixel will belong to label 0 is stored
                    ones = torch.ones_like(pred)
                    proba_unflooded = torch.sub(ones, pred)
                    #transform every element in mask predict to shape B,C,H,W, where C=2 and holds the probabilities for label 0 and 1
                    pred = torch.cat([proba_unflooded,pred],axis=1)
                    #convert from shape (B,C,H,W), to (B,H,W,C) and finally to (N,C), where N = B*W*H
                    pred = pred.permute(0,2,3,1).flatten(0,2)
                    mask_pred.append(pred)


                    model.to('cpu')


            #mask pred is the list which has the predictions of shape N,C of the models on the current batch
            #we transform this into one big tensor for the current batch (of shape Num_Models,N,C )
            current_batch_ensemble_proba = torch.stack(mask_pred,dim=0)
            ensemble_proba.append(current_batch_ensemble_proba)


            #flatten all the masks in the batch
            #mask will initially be of shape (B,H,W) transform to (N)
            target.append(mask_true.flatten())

    #merge all the probabilities/targets from the single batches together
    return torch.cat(ensemble_proba,dim=1), torch.cat(target,dim=0)



#Source: https://github.com/milesial/Pytorch-UNet/blob/master/evaluate.py
#Strongly Adapted (Made the function work for ensemble fusion)
@torch.inference_mode()
def evaluate_ensemble_fusion(dataset, model_paths,selected_models, amp, batch_size, metric= 'iou',fusion_method='fusion', dataset_type='landcover'):
    """
    Evaluates the performance of an ensemble of models using various fusion methods and metrics.

    Args:
        dataset (Dataset): The dataset to evaluate the models on.
        model_paths (list): List of paths to the trained models.
        selected_models (list): List of selected model indices to use for evaluation.
        amp (bool): Whether to use automatic mixed precision for inference.
        batch_size (int): The number of samples per batch.
        metric (str): The evaluation metric to use ('iou', 'accuracy', 'f1_macro', 'f1_micro', 'all_eval').
        fusion_method (str): The fusion method to use ('vote', 'fusion', 'none', 'both').
        dataset_type (str): The type of dataset ('landcover' or 'floods') to adjust model architecture.

    Returns:
        float: The average evaluation score based on the selected metric.
    """


    #indicate if the dataset is generated from landcover or S1GFLOOD
    if dataset_type == 'landcover':
        landcover = True
    else:
        landcover = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_paths = [model_paths[i] for i in selected_models]
    
    

    models = []
   
    for model_path in model_paths:
        model_l = UNet(n_channels=3,n_classes=5) if landcover else SiamUnet_conc(n_channels=3,n_classes=1)
        model_l.load_state_dict(torch.load(model_path, map_location=device))
        #move to cpu initially and always just load for inference
        model_l.to('cpu')
        model_l.eval()
        models.append(model_l)
    
    
    


    

    dataloader = DataLoader(dataset, shuffle=False,drop_last=False,batch_size=batch_size,num_workers = os.cpu_count(),pin_memory=True)
    num_batches = len(dataloader)
    eval_score = 0 

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_batches, desc='Evaluate Ensemble', unit='batch', leave=False):

            if landcover:
                image, mask_true = batch['image'], batch['mask']

                # move images and labels to correct device and type
                image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                mask_true = mask_true.to(device=device, dtype=torch.long)



            

                #make a prediction with every model
                mask_pred = []
                for model in models:
                    model.to(device)
                    mask_pred.append(F.softmax(model(image),dim=1))
                    model.to('cpu')
                

                
              
            

                
            
            else:
                image_pre,image_post, mask_true = batch['image_pre'], batch['image_post'], batch['mask']


                # move images and labels to correct device and type
                image_pre = image_pre.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                image_post = image_post.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                mask_true = mask_true.to(device=device, dtype=torch.long)

                

                #make a prediction with every model
                mask_pred = []
                for model in models:
                    model.to(device)
                    pred = F.sigmoid(model(image_pre,image_post))

                    #transform every element in mask predict to shape B,C,H,W, where C=2 and holds the probabilities for label 0 and 1
                    ones = torch.ones_like(pred)
                    proba_unflooded = torch.sub(ones, pred)
                    pred = torch.cat([proba_unflooded,pred],axis=1)
                    mask_pred.append(pred)


                    model.to('cpu')


                

            
    
           

            
            
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

            
            #using the function to evaluate a single model
            elif(fusion_method == 'none'):
                #transform from shape B,C,H,W to B,H,W
                fused_pred = (mask_pred[0]).argmax(dim=1)

            elif(fusion_method == 'both'):
                #shape B,H,W
                class_predictions = [torch.argmax(output, dim=1) for output in mask_pred] 
                
                #shape M,B,H,W
                stacked_predictions = torch.stack(class_predictions)

                #shape B,H,W
                vote_pred, _ = torch.mode(stacked_predictions, dim=0)


                #posterior averaging
                #ensemble preds has shape (M,B, C, H, W), where M is the number of selected models 
                ensemble_preds = torch.stack(mask_pred)

                #shape (B, C, H, W)
                fused = ensemble_preds.mean(dim=0)

                posterior_pred = fused.argmax(dim=1)


            else:
                raise ValueError(f"Unsupported fusion_method: {fusion_method}. Please use 'vote','fusion', 'none' or 'both'.")
            
            


           
    



            if models[0].n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                
                if metric == 'iou':
                    # convert to one-hot format
                    #mask get both converted from B, H, W to B,H,W,2, which is reordered to B,2,H,W to stickt to the B,C,H,W convention
                    mask_true = F.one_hot(mask_true.long(), 2).permute(0, 3, 1, 2).float()
                    mask_pred = F.one_hot(fused_pred.long(), 2).permute(0, 3, 1, 2).float()

                    # compute iou score
                    eval_score += iou(mask_pred, mask_true)
                elif metric == 'accuracy':
                    eval_score += multiclass_accuracy(fused_pred, mask_true, num_classes=2, average='micro', multidim_average='global').item()
                elif metric == 'f1_macro':
                    eval_score += multiclass_f1_score(fused_pred, mask_true, num_classes=2, average='macro', multidim_average='global', zero_division=1).item()
                elif metric == 'f1_micro':
                    eval_score += multiclass_f1_score(fused_pred, mask_true, num_classes=2, average='micro', multidim_average='global', zero_division=1).item()
                elif metric == 'all_eval':
                    
                    
                    # convert to one-hot format
                    #mask get both converted from B, H, W to B,H,W,2, which is reordered to B,2,H,W to stickt to the B,C,H,W convention
                    mask_true_iou = F.one_hot(mask_true.long(), 2).permute(0, 3, 1, 2).float()
                    mask_pred_iou_vote = F.one_hot(vote_pred.long(), 2).permute(0, 3, 1, 2).float()
                    mask_pred_iou_posterior = F.one_hot(posterior_pred.long(), 2).permute(0, 3, 1, 2).float()

                    # compute iou score
                    iou_vote = iou(mask_pred_iou_vote, mask_true_iou).item()

                    f1_macro_vote = multiclass_f1_score(vote_pred, mask_true, num_classes=2, average='macro', multidim_average='global', zero_division=1).item()

                    f1_micro_vote = multiclass_f1_score(vote_pred, mask_true, num_classes=2, average='micro', multidim_average='global', zero_division=1).item()

                    iou_posterior = iou(mask_pred_iou_posterior, mask_true_iou).item()

                    f1_macro_posterior = multiclass_f1_score(posterior_pred, mask_true, num_classes=2, average='macro', multidim_average='global', zero_division=1).item()

                    f1_micro_posterior = multiclass_f1_score(posterior_pred, mask_true, num_classes=2, average='micro', multidim_average='global', zero_division=1).item()


                    eval_score += np.array([iou_vote, f1_macro_vote, f1_micro_vote, iou_posterior, f1_macro_posterior, f1_micro_posterior])



                else:
                    raise ValueError(f"Unsupported metric: {metric}. Please use 'accuracy' or 'iou' ")



        
            else:
                assert mask_true.min() >= 0 and mask_true.max() < models[0].n_classes, 'True mask indices should be in [0, n_classes['
                if metric == 'iou':
                    # convert to one-hot format
                    mask_true = F.one_hot(mask_true, models[0].n_classes).permute(0, 3, 1, 2).float()
                    mask_pred = F.one_hot(fused_pred, models[0].n_classes).permute(0, 3, 1, 2).float()
                    
                    # compute the iou  score, including background
                    eval_score += iou(mask_pred, mask_true)
                elif metric == 'accuracy':
                    eval_score += multiclass_accuracy(fused_pred, mask_true, num_classes=models[0].n_classes, average='micro', multidim_average='global').item()
                elif metric == 'f1_macro':
                    eval_score += multiclass_f1_score(fused_pred, mask_true, num_classes=models[0].n_classes, average='macro', multidim_average='global', zero_division=1).item()
                elif metric == 'f1_micro':
                    eval_score += multiclass_f1_score(fused_pred, mask_true, num_classes=models[0].n_classes, average='micro', multidim_average='global', zero_division=1).item()
                elif metric == 'all_eval':


                    # convert to one-hot format
                    mask_true_iou = F.one_hot(mask_true.long(), models[0].n_classes).permute(0, 3, 1, 2).float()
                    mask_pred_iou_vote = F.one_hot(vote_pred.long(), models[0].n_classes).permute(0, 3, 1, 2).float()
                    mask_pred_iou_posterior = F.one_hot(posterior_pred.long(), models[0].n_classes).permute(0, 3, 1, 2).float()

                    # compute iou score
                    iou_vote = iou(mask_pred_iou_vote, mask_true_iou).item()

                    f1_macro_vote = multiclass_f1_score(vote_pred, mask_true, num_classes=models[0].n_classes, average='macro', multidim_average='global', zero_division=1).item()

                    f1_micro_vote = multiclass_f1_score(vote_pred, mask_true, num_classes=models[0].n_classes, average='micro', multidim_average='global', zero_division=1).item()

                    iou_posterior = iou(mask_pred_iou_posterior, mask_true_iou).item()

                    f1_macro_posterior = multiclass_f1_score(posterior_pred, mask_true, num_classes=models[0].n_classes, average='macro', multidim_average='global', zero_division=1).item()

                    f1_micro_posterior = multiclass_f1_score(posterior_pred, mask_true, num_classes=models[0].n_classes, average='micro', multidim_average='global', zero_division=1).item()


                    eval_score += np.array([iou_vote, f1_macro_vote, f1_micro_vote, iou_posterior, f1_macro_posterior, f1_micro_posterior])
                
                else:
                    raise ValueError(f"Unsupported metric: {metric}. Please use 'accuracy' or 'iou' ")

     




   
    return eval_score / max(num_batches, 1)











           
            




  


   


        









