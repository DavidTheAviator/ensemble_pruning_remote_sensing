import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
import torch
import sys
import numpy as np
import re
from sklearn.cluster import KMeans


# Add Parent Directory to Path
project_root_dir = os.path.join(os.path.dirname(__file__), "..","..")
project_root_dir = os.path.abspath(project_root_dir)

sys.path.append(project_root_dir)



# Ensemble imports
from Ensemble.models import SiamUnet_conc, UNet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helpers import get_model_paths, evaluate_ensemble_fusion



#Source: https://stackoverflow.com/questions/2572916/numpy-smart-symmetric-matrix
#Not Adapted
def make_symmetric(a):
    """
    Make a square matrix symmetric. 

    Args:
        a (numpy.ndarray): A square matrix.

    Returns:
        numpy.ndarray: A symmetric version of the input matrix.
    """
    diag = np.zeros_like(a)
    np.fill_diagonal(diag,np.diag(a))

    return a + a.T - diag


def get_bagging_sets(experiment_path, train_set):
    """
    Load bagging index files and create (bagging) datasets from them

    Args:
        experiment_path (Path): Path to the experiment directory (i.e. of an initial ensemble).
        train_set (Dataset): The full training dataset.

    Returns:
        list: A list of dataset subsets based on bagging indices.
    """

    dataset_path = experiment_path.joinpath('datasets')
    dataset_names = [ds for ds in os.listdir(dataset_path) if ds.endswith('.npy')]
    dataset_names = sorted(dataset_names,key=lambda x: int(re.search(r'dataset_(\d+)',x).group(1)))

    dataset_paths = [experiment_path.joinpath('datasets').joinpath(ds_name) for ds_name in dataset_names]

    bagging_sets = []
    for ds_path in dataset_paths:
        bagging_index = np.load(ds_path)
        bagging_sets.append(torch.utils.data.Subset(train_set,bagging_index))




    return bagging_sets
    



#calculate an accuracy vector for a single model
def calculate_accuracy_vector(experiment_path,selected_model, train_set, batch_size, amp, dataset_type='landcover'):
    """
    Compute an accuracy vector for a single model across bagging sets.

    Args:
        experiment_path (Path): Path to the experiment directory (i.e. of an initial ensemble).
        selected_model (int): Index of the selected model.
        train_set (Dataset): The full training dataset.
        batch_size (int): Batch size for evaluation.
        amp (bool): Whether to use automatic mixed precision.
        dataset_type (str): Type of dataset ('landcover' ord 'floods').

    Returns:
        np.ndarray: Accuracy values for each bagging set.
    """

    accuracy_list = []

    model_paths = get_model_paths(experiment_path)
    bagging_sets = get_bagging_sets(experiment_path, train_set)

    
    for bagging_set in bagging_sets:   
    
        accuracy = evaluate_ensemble_fusion(dataset=bagging_set, 
                                            model_paths=model_paths,
                                            selected_models=selected_model, 
                                            amp=amp, 
                                            batch_size=batch_size, 
                                            metric= 'accuracy',
                                            fusion_method='none', 
                                            dataset_type=dataset_type)
        accuracy_list.append(accuracy)
    
    return np.asarray(accuracy_list)



                
def calculate_accuracy_matrix(experiment_path,train_set,batch_size,amp,dataset_type):
    """
    Compute a (symmetric) accuracy matrix for all models in the experiment.
    
    Args:
        experiment_path (Path): Path to the experiment directory (i.e. of an initial ensemble).
        train_set (Dataset): The training dataset.
        batch_size (int): Batch size for evaluation.
        amp (bool): Whether to use automatic mixed precision.
        dataset_type (str): Type of dataset used (i.e. 'landcover' or 'floods').

    Returns:
        numpy.ndarray: Symmetric accuracy matrix.
    """

    model_paths = get_model_paths(experiment_path)
    num_models = len(model_paths)

    accuracy_vectors = []

    for i in range(num_models):
        
        acc_vector = calculate_accuracy_vector(experiment_path=experiment_path,
                                               selected_model=[i], 
                                               train_set=train_set, 
                                               batch_size=batch_size, 
                                               amp=amp, 
                                               dataset_type=dataset_type)
        accuracy_vectors.append(acc_vector)



    accuracy_matrix = np.zeros(shape=(num_models,num_models))
    
    #fill just the upper triangle of the matrix
    # the diagonal is zero by definition and the matrix is symmetric
    for i in range(num_models):
        for j in range(i+1,num_models):
            accuracy_matrix[i,j] = np.sqrt(np.dot(acc_vector[i],acc_vector[j])/num_models)

    return make_symmetric(accuracy_matrix)




# Source:https://github.com/scikit-learn-contrib/DESlib/blob/master/deslib/util/diversity.py#L157-L180; Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#slightly adapted (converted loop to vector operation)
def _process_predictions(y, y_pred1, y_pred2):
    """Pre-process the predictions of a pair of base classifiers for the
    computation of the diversity measures

    Parameters
    ----------
    y : array of shape (n_samples):
        class labels of each sample.

    y_pred1 : array of shape (n_samples):
              predicted class labels by the classifier 1 for each sample.

    y_pred2 : array of shape (n_samples):
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    N00 : Percentage of samples that both classifiers predict the wrong label

    N10 : Percentage of samples that only classifier 2 predicts the wrong label

    N10 : Percentage of samples that only classifier 1 predicts the wrong label

    N11 : Percentage of samples that both classifiers predict the correct label
    """
    size_y = len(y)
    if size_y != len(y_pred1) or size_y != len(y_pred2):
        raise ValueError(
            'The vector with class labels must have the same size.')

    N00, N10, N01, N11 = 0.0, 0.0, 0.0, 0.0

    n11 = np.logical_and(y_pred1 == y,y_pred2 == y)
    n10 = np.logical_and(y_pred1 == y, y_pred2 != y)
    n01 = np.logical_and(y_pred1 != y, y_pred2 == y)
    n00 = np.logical_and(y_pred1 != y, y_pred2 != y)
    
    N11 = n11.sum()
    N10 = n10.sum()
    N01 = n01.sum()
    N00 = n00.sum()



    return N00 / size_y, N10 / size_y, N01 / size_y, N11 / size_y




# Source:https://github.com/scikit-learn-contrib/DESlib/blob/master/deslib/util/diversity.py#L157-L180; Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#Slightly adapted (added the cases for a Q-statistic of one from the Spectral Clustering Pruning paper)
def Q_statistic(y, y_pred1, y_pred2):
    """Calculates the Q-statistics diversity measure between a pair of
    classifiers. The Q value is in a range [-1, 1]. Classifiers that tend to
    classify the same object correctly will have positive values of Q, and
    Q = 0 for two independent classifiers.

    Parameters
    ----------
    y : array of shape (n_samples):
        class labels of each sample.

    y_pred1 : array of shape (n_samples):
              predicted class labels by the classifier 1 for each sample.

    y_pred2 : array of shape (n_samples):
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    Q : The q-statistic measure between two classifiers
    """
    size_y = len(y)

    N00, N10, N01, N11 = _process_predictions(y, y_pred1, y_pred2)
    
    if N00 == 1 or N00 == 0:
        return 1
    if N01 == 1 or N01 == 0:
        return 1
    if N10 == 1 or N10 == 0:
        return 1
    if N11 == 1 or N11 == 0:
        return 1
    
    Q = ((N11 * N00) - (N01 * N10)) / ((N11 * N00) + (N01 * N10))
    return Q


    

                    
                
           
#Source: https://github.com/milesial/Pytorch-UNet/blob/master/evaluate.py
#Adapted (Made the function work for ensemble fusion)
@torch.inference_mode()
def calculate_diversity(dataset, model_paths,selected_models, amp, batch_size, dataset_type='landcover'):
    """
    Compute the diversity between two selected models using the Q-statistic.

    Args:
        dataset (Dataset): The dataset used for evaluation.
        model_paths (list): List of paths to all available models.
        selected_models (list): Indices of the two models to evaluate (the diversity between them).
        amp (bool): Whether to use automatic mixed precision.
        batch_size (int): Batch size for inference.
        dataset_type (str): Type of dataset ('landcover' or 'floods'). Defaults to 'landcover'.

    Returns:
        float: The computed Q-statistic measuring the diversity.
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
    q_stat = 0 

    
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_batches, desc='Diversity Calculation round', unit='batch', leave=False):

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


            
            #shape B,H,W
            preds = [torch.argmax(output, dim=1) for output in mask_pred] 

            q_stat += Q_statistic(mask_true.flatten().cpu().numpy(), preds[0].flatten().cpu().numpy(),preds[1].flatten().cpu().numpy())

            
    
     




   
    return q_stat / max(num_batches, 1)



def calculate_diversity_matrix(experiment_path,train_set,batch_size,amp,dataset_type):    
    """
    Compute the diversity matrix for all models using the Q-statistic. 
    The matrix quantifies pairwise diversity between models based on their predictions 
    on different bagging datasets.

    Args:
        experiment_path (Path): Path to the experiment directory (i.e. of an initial ensemble).
        train_set (Dataset): The training dataset.
        batch_size (int): Batch size for evaluation.
        amp (bool): Whether to use automatic mixed precision.
        dataset_type (str): Type of dataset ('landcover' or 'floods').

    Returns:
        np.ndarray: A symmetric matrix of shape (num_models, num_models) representing 
                    pairwise model diversity.
    """

    model_paths = get_model_paths(experiment_path)
    num_models = len(model_paths)


    bagging_sets = get_bagging_sets(experiment_path, train_set)
    



    diversity_matrix = np.zeros(shape=(num_models,num_models))
    
    #fill just the upper triangle of the matrix
    # the diagonal is zero by definition and the matrix is symmetric
    for i in range(num_models):
        for j in range(i+1,num_models):
            ds_i = bagging_sets[i]
            ds_j = bagging_sets[j]
            q_ij = calculate_diversity(dataset=ds_i,
                                       model_paths= model_paths,
                                       selected_models= [i,j],
                                       amp=amp, 
                                       batch_size=batch_size, 
                                       dataset_type=dataset_type)
            q_ji = calculate_diversity(dataset=ds_j,
                                       model_paths= model_paths,
                                       selected_models= [i,j],
                                       amp=amp, 
                                       batch_size=batch_size, 
                                       dataset_type=dataset_type)
            

            q_ij = (1-q_ij)/2
            q_ji = (1-q_ji)/2

            diversity_matrix[i,j] = np.sqrt(q_ij*q_ji)

    return make_symmetric(diversity_matrix)





def calculate_similarity_matrix(experiment_path,train_set,batch_size,amp,dataset_type,similarity_type = 'gm',lambd= None):
    """
    Compute the similarity matrix by combining the accuracy and diversity matrices.
    The similarity can be computed using the geometric mean ('gm') or a weighted 
    sum ('lambda') of accuracy and diversity.

    Args:
        experiment_path (Path): Path to the experiment directory (i.e. of an initial ensemble).
        train_set (Dataset): The training dataset.
        batch_size (int): Batch size for evaluation.
        amp (bool): Whether to use automatic mixed precision.
        dataset_type (str): Type of dataset ('landcover' or 'floods').
        similarity_type (str: Similarity computation method ('gm' or 'lambda'). Defaults to 'gm'.
        lambd (float, optional): Weighting factor for accuracy in 'lambda' mode (between 0 and 1).

    Returns:
        np.ndarray: A matrix of shape (num_models, num_models) representing model similarity.

    Raises:
        ValueError: If an invalid similarity type is provided or 'lambda' is chosen without a valid lambd.
    """ 
    diversity_matrix = calculate_diversity_matrix(experiment_path,train_set,batch_size,amp,dataset_type)
    accuracy_matrix = calculate_accuracy_matrix(experiment_path,train_set,batch_size,amp,dataset_type)

    if similarity_type == 'gm':
        return np.sqrt(accuracy_matrix * diversity_matrix)
    elif similarity_type == 'lambda' and lambd is not None:
        return (lambd * accuracy_matrix) + ((1-lambd) * diversity_matrix)
    
    else:
        raise ValueError(f"Similarity Type has to be 'gm' or 'lambda' but is {similarity_type}, if it is lambda lambd between 0 and 1 has to be set" )




def spectral_clustering(experiment_path,train_set,batch_size,amp,dataset_type,similarity_type = 'gm',lambd= None):
    """
    Perform spectral clustering on models based on their similarity, as laid out in "A spectral clustering based ensemble pruning approach" (Zhang and Cao, 2014). 

    Args:
        experiment_path (Path): Path to the experiment directory (i.e. directory of an initial ensemble) containing model paths.
        train_set (Dataset): The training dataset used for evaluation.
        batch_size (int): Batch size for evaluation.
        amp (bool): Whether to use automatic mixed precision.
        dataset_type (str): Type of dataset ('landcover' or 'floods').
        similarity_type (str): Method to calculate similarity ('gm' or 'lambda'). Defaults to 'gm'.
        lambd (float, optional): Weighting factor for accuracy in 'lambda' mode (between 0 and 1).

    Returns:
        np.ndarray: Indices of models belonging to the cluster with higher average similarity.
    """


    num_models = len(get_model_paths(experiment_path))
    
    W = calculate_similarity_matrix(experiment_path=experiment_path,
                                    train_set=train_set,
                                    batch_size=batch_size,
                                    amp=amp,
                                    dataset_type=dataset_type,
                                    similarity_type = similarity_type,
                                    lambd= lambd)

    D = np.diag(W.sum(axis=0))

    I = np.eye(num_models)

    D_sqrt_inv = np.linalg.inv(np.sqrt(D))

    L = I - np.linalg.multi_dot([D_sqrt_inv,W,D_sqrt_inv])


    eigenvalues, eigenvectors = np.linalg.eig(L)

    #from https://stackoverflow.com/questions/8092920/sort-eigenvalues-and-associated-eigenvectors-after-using-numpy-linalg-eig-in-pyt
    ids = eigenvalues.argsort()  
    eigenvalues = eigenvalues[ids]
    eigenvectors = eigenvectors[:,ids]

    v_1 = eigenvectors[:,0]
    v_2 = eigenvectors[:,1]

    clustering_matrix = np.concatenate([v_1[:,np.newaxis],v_2[:,np.newaxis]], axis=1)

    kmeans = KMeans(n_clusters=2,init='random').fit(clustering_matrix)

    #assign the indices of the classifiers to
    sub_graph_ids_a = np.nonzero(kmeans.labels_==0)[0]
    sub_graph_ids_b = np.nonzero(kmeans.labels_==1)[0]



    #compute the averaged similarity of the clusters
    # Formula: Sum of Similarity Weights in Subgraph / Number of edges in Subgraph 
    edges_a,edges_b, weight_sum_a, weight_sum_b = 0.0,0.0,0.0,0.0
    for i in range(num_models):
        #we just need to consider every edge once, so it is sufficient to examine the upper matrix triangle
        for j in range(i+1, num_models):
            #edge in sub graph a
            if i in sub_graph_ids_a and j in sub_graph_ids_a:
                edges_a += 1
                weight_sum_a += W[i,j]
            elif i in sub_graph_ids_b and j in sub_graph_ids_b:
                edges_b += 1
                weight_sum_b += W[i,j]
            else:
                pass 

    if edges_a > 0:
        average_similarity_a = weight_sum_a/edges_a
    else:
        average_similarity_a = 0 

    if edges_b > 0:
        average_similarity_b = weight_sum_b/edges_b
    else:
        average_similarity_b = 0 

    #choose cluster with higher average similarity 
    if average_similarity_a > average_similarity_b:
        return sub_graph_ids_a
    else:
        return sub_graph_ids_b


   












class SpectralClusterPruningClassifier:
    """
    Minimal class structure for the Spectral Clustering Pruning algorithm.

    Methods:
        prune_(experiment_path, train_set, batch_size, amp, dataset_type, similarity_type='gm', lambd=None):
            Prunes the ensemble by applying spectral clustering and selecting a cluster based on similarity.
    """
  
    def __init__(self):
        """
        Initializes the SpectralClusterPruningClassifier.
        """
        pass

    def prune_(self, experiment_path,train_set,batch_size,amp,dataset_type,similarity_type = 'gm',lambd= None):
        """Wrapper for the spectral_clustering function"""
        
        return spectral_clustering(experiment_path,train_set,batch_size,amp,dataset_type,similarity_type = similarity_type,lambd= lambd)

        