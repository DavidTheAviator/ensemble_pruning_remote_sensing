import os
import sys
import numpy as np
from pathlib import Path
import torch
import itertools
import pandas as pd
import time
import shutil

# Add Parent Directory to Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Project-Specific Imports
from Ensemble.utils import timestamp
from PyPruning.PyPruning.helpers import generate_ensemble_proba_target, evaluate_ensemble_fusion, get_model_paths
from PyPruning.PyPruning.GreedyPruningClassifier import GreedyPruningClassifier, margin_distance_minimization, uwa, diaces, sdacc, dftwo, accrein
from PyPruning.PyPruning.RankPruningClassifier import RankPruningClassifier, individual_contribution_ordering, orientation_ordering, individual_margin_diversity
from PyPruning.PyPruning.SpectralClusterPruningClassifier import SpectralClusterPruningClassifier



def rounded_df_copy(df, col_names_rounding):
    """
    Creates a copy of a DataFrame with specified columns rounded to three decimal places.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col_names_rounding (list of str): List of column names to round.

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns rounded to three decimal places.
    """

    df_to_save = df.copy(deep=True)
    for col_name in col_names_rounding:
        df_to_save[col_name] = df_to_save[col_name].astype(float).round(3)
    return df_to_save




def prune_single_ensemble_experiment(logger, 
                                     experiment_path,
                                     train_set, 
                                     val_set, 
                                     test_set, 
                                     sub_sample_size,
                                     n_estimators,
                                     calculate_spectral, 
                                     calculate_optimal,
                                     calculate_benchmarks,
                                     batch_size, 
                                     amp,
                                     n_jobs, 
                                     dataset_type = 'landcover'):
    """
    Execute a single pruning run (one initial ensemble, fixed ensemble size, different pruning methods) 
    and evaluate the obtained ensembles on the test set (via mIoU, macro F1 and micro F1).
    
    Args:
        logger: Logging object for tracking experiment progress and output.
        experiment_path (str): Path to the directory where the experiment results will be stored.
        train_set (Dataset): The training dataset used for model training (and in some algorithms for pruning).
        val_set (Dataset): The validation dataset used for validation of the models (and for pruning).
        test_set (Dataset): The test dataset used for evaluating the final performance of the obtained subensembles.
        sub_sample_size (int): Number of random samples (i.e., images) selected from the validation set for pruning.
        n_estimators (int): The number of classifiers to be selected for the ensemble after pruning (i.e., subensemble size)
        calculate_spectral (bool): Whether to compute Spectral Clustering Pruning
        calculate_optimal (bool): Whether to calculate the optimal subensemble (based on the validation set).
        calculate_benchmarks (bool): Whether to compute benchmark metrics for model evaluation.
        batch_size (int): The number of samples processed in each batch 
        amp (bool): Whether to use automatic mixed precision
        n_jobs (int): Number of parallel jobs to use
        dataset_type (str): Type of dataset being used, e.g., 'landcover' or 'floods'. Default is 'landcover'.

    Returns:
        pd.DataFrame: DataFrame summarizing the evaluation results for each pruning strategy and its variations.

    """
    ####################
    ##  SET- UP
    ####################
    #region

    time_total_start = time.time()
    time_ensemble_proba_start = time.time()


    experiment_path = Path(experiment_path)
    model_paths = get_model_paths(experiment_path)
    num_models = len(model_paths)
    
    
    #sub samples a Pruning Set from the VAL Set
    rng = np.random.default_rng()
    sample_index = rng.choice(np.arange(len(val_set)),size=sub_sample_size,replace=False)
    pruning_set = torch.utils.data.Subset(val_set, sample_index)

    logger.info(f"Images at these inidces chosen randomly for pruning: {sample_index}")
    
    logger.info("Generating the Ensemble Probability Table .....")

    #generate the ensemble proba table which is used for most pruning algos , which has shape (Classifiers, Sample_Number, Classes) or (M,N,C)
    ensemble_proba, target = generate_ensemble_proba_target(model_paths, pruning_set,batch_size= batch_size, amp=amp, dataset_type=dataset_type)
    ensemble_proba = ensemble_proba.cpu().numpy()
    target = target.numpy()

    logger.info("DONE")

    logger.info("Generating the Ensemble Probability Table List (for batched versions of algorithms) .....")

    if dataset_type == 'landcover':
        metric_batch_size = 8
    else:
        metric_batch_size = 32
    
   

    val_set_ids = np.arange(len(val_set))
    rng.shuffle(val_set_ids)
    ensemble_proba_batches = np.split(val_set_ids, np.arange(metric_batch_size, len(val_set), metric_batch_size))

   

    #in this folder the ensemble_proba arrays get saved temporarily
    temp_path = experiment_path.joinpath('..').joinpath('..').joinpath('temp')
    proba_target_path = temp_path.joinpath(f"ensemble_proba_{timestamp()}")

    if not os.path.exists(proba_target_path):
        os.makedirs(proba_target_path)

    #save ensemble proba and target for every batch
    for i, batch_index in enumerate(ensemble_proba_batches):
        batch_pruning_set = torch.utils.data.Subset(val_set, batch_index)
        ensemble_proba_batch, target_batch = generate_ensemble_proba_target(model_paths, batch_pruning_set,batch_size= 8, amp=amp, dataset_type=dataset_type)
        ensemble_proba_batch = ensemble_proba_batch.cpu().numpy()
        target_batch = target_batch.numpy()
        
        
        #save the arrays in the folder
        np.save(proba_target_path.joinpath(f'ensemble_proba_batch_{i}.npy'), ensemble_proba_batch) # save
        np.save(proba_target_path.joinpath(f'target_batch_{i}.npy'),target_batch)

    logger.info("DONE")
    

    time_ensemble_proba_end = time.time()
    #endregion

    ####################
    ##  RANKING BASED
    ####################
    #region

    ######Not Batched
    time_rank_normal_start = time.time()

    logger.info("Performing ORIENTATION ORDERING (OO) .... ")     
    start_oo = time.time()

    ###ORIENTATION ORDERING (OO)
    #pre calculate the reference vector for better performance 
    c_sig = 2.0*(ensemble_proba.argmax(axis=2) == target[np.newaxis,:]) -1.0
    
    #ensemble signature vector
    c_ens = c_sig.mean(axis=0)
    o = np.ones(len(c_ens), dtype=float)


    lamb = np.dot(-o,c_ens)/np.dot(c_ens,c_ens)
    c_ref = o + lamb * c_ens

    oo_cl = RankPruningClassifier(n_estimators, metric = orientation_ordering, n_jobs = n_jobs, metric_options = {'c_ref': c_ref})
    oo_ids, _ = oo_cl.prune_(proba= ensemble_proba, target=target)
    
    end_oo = time.time()

    logger.info("DONE")
    logger.info(f"oo_ids: {oo_ids}")
  
    logger.info("Creating Votes Table.....")
    start_votes = time.time()

    ###calculate Vote array of shape (N,C) here for higher efficiency  
    V = np.zeros(ensemble_proba.shape)
    idx = ensemble_proba.argmax(axis=2)
    V[np.arange(ensemble_proba.shape[0])[:,None],np.arange(ensemble_proba.shape[1]),idx] = 1
    V = V.sum(axis=0)
    end_votes = time.time()
    logger.info("DONE")

    logger.info("Performing Ensemble pruning via individual contribution ordering (EPIC) .... ")
    start_epic = time.time()
    ###Ensemble pruning via individual contribution ordering (EPIC)
    
    
    epic_cl = RankPruningClassifier(n_estimators, metric = individual_contribution_ordering, n_jobs = n_jobs, metric_options = {'V': V})
    epic_ids, _ = epic_cl.prune_(proba= ensemble_proba, target=target)
    end_epic = time.time()

    logger.info("DONE")
    logger.info(f"epic_ids: {epic_ids}")

    logger.info("Performing Margin & Diversity based ordering Ensemble Pruning (MDEP) .... ")
    start_mdep = time.time()

    ###Margin & Diversity based ordering Ensemble Pruning (MDEP)
    mdep_cl = RankPruningClassifier(n_estimators, metric = individual_margin_diversity, n_jobs = n_jobs, metric_options = {'alpha': 0.2, 'V': V})
    mdep_ids, _ = mdep_cl.prune_(proba= ensemble_proba, target=target)

    end_mdep = time.time()
    logger.info("DONE")
    logger.info(f"mdep_ids: {mdep_ids}")

    time_rank_normal_end = time.time()


    ######Batched
    time_rank_batched_start = time.time()

    logger.info("Performing ORIENTATION ORDERING (OO) - IN BATCHES .... ")     
    start_oo_batches = time.time()

    ###ORIENTATION ORDERING (OO)
    oo_batches_cl = RankPruningClassifier(n_estimators, metric = orientation_ordering, n_jobs = n_jobs)
    oo_batches_ids, _ = oo_batches_cl.prune_(proba= None, target=None, proba_target_path= proba_target_path)
    end_oo_batches = time.time()

    logger.info("DONE")
    logger.info(f"oo__batches_ids: {oo_batches_ids}")
  
    logger.info("Performing Ensemble pruning via individual contribution ordering (EPIC) - IN BATCHES .... ")
    start_epic_batches = time.time()

    ###Ensemble pruning via individual contribution ordering (EPIC)
    epic_batches_cl = RankPruningClassifier(n_estimators, metric = individual_contribution_ordering, n_jobs = n_jobs)
    epic_batches_ids, _ = epic_batches_cl.prune_(proba= None, target=None, proba_target_path= proba_target_path)
    end_epic_batches = time.time()

    logger.info("DONE")
    logger.info(f"epic_batches_ids: {epic_batches_ids}")

    logger.info("Performing Margin & Diversity based ordering Ensemble Pruning (MDEP) - IN BATCHES.... ")
    start_mdep_batches = time.time()

    ###Margin & Diversity based ordering Ensemble Pruning (MDEP)
    mdep_batches_cl = RankPruningClassifier(n_estimators, metric = individual_margin_diversity, n_jobs = n_jobs, metric_options = {'alpha': 0.2})
    mdep_batches_ids, _ = mdep_batches_cl.prune_(proba= None, target=None, proba_target_path= proba_target_path)

    end_mdep_batches = time.time()
    logger.info("DONE")
    logger.info(f"mdep_batches_ids: {mdep_batches_ids}")

    time_rank_batched_end = time.time()
    #endregion

    ####################
    ##  GREEDY
    ####################
    #region
    ######Not Batched

    time_greedy_normal_start = time.time()

    logger.info("Performing Margin Distance Minimization (MDM) .... ")
    start_mdm = time.time()

    ###Margin Distance Minimization (MDM)
    mdm_cl = GreedyPruningClassifier(n_estimators, metric=margin_distance_minimization)
    mdm_ids,_ = mdm_cl.prune_(proba= ensemble_proba, target=target)

    end_mdm = time.time()
    logger.info("DONE")
    logger.info(f"mdm_ids: {mdm_ids}")

    logger.info("Performing Uncertainty Weighted Accuracy (UWA) .... ")
    start_uwa = time.time()
    
    ###Uncertainty Weighted Accuracy (UWA)
    uwa_cl = GreedyPruningClassifier(n_estimators, metric=uwa)
    uwa_ids,_ = uwa_cl.prune_(proba= ensemble_proba, target=target)

    end_uwa = time.time()
    logger.info("DONE")
    logger.info(f"uwa_ids: {uwa_ids}")


    logger.info("Performing Diversity-Accuracy Measure for Homogenous Ensemble Selection (DIACES) .... ")
    start_diaces = time.time()

    ###Diversity-Accuracy Measure for Homogenous Ensemble Selection (DIACES)

    diaces_cl = GreedyPruningClassifier(n_estimators, metric=diaces,  metric_options = {'alpha':0.8})
    diaces_ids,_ = diaces_cl.prune_(proba= ensemble_proba, target=target)

    end_diaces = time.time()
    logger.info("DONE")
    logger.info(f"diaces_ids: {diaces_ids}")

   
    logger.info("Performing simultaneous diversity & accuracy (SDACC) .... ")
    start_sdacc = time.time()

    ###Simultaneous diversity & accuracy (SDACC)
    sdacc_cl = GreedyPruningClassifier(n_estimators, metric=sdacc)
    sdacc_ids,_ = sdacc_cl.prune_(proba= ensemble_proba, target=target)

    end_sdacc = time.time()
    logger.info("DONE")
    logger.info(f"sdacc_ids: {sdacc_ids}")

    
    
    logger.info("Performing Diversity-focused-two (DFTWO) .... ")
    start_dftwo = time.time()


    ###Diversity-focused-two (DFTWO)
    dftwo_cl = GreedyPruningClassifier(n_estimators, metric=dftwo)
    dftwo_ids,_ = dftwo_cl.prune_(proba= ensemble_proba, target=target)

    end_dftwo = time.time()
    logger.info("DONE")
    logger.info(f"dftwo_ids: {dftwo_ids}")



    logger.info("Performing Accuracy-reinforcement (ACCREIN) .... ")
    start_accrein = time.time()

    ###Accuracy-reinforcement (ACCREIN)
    accrein_cl = GreedyPruningClassifier(n_estimators, metric=accrein)
    accrein_ids,_ = accrein_cl.prune_(proba= ensemble_proba, target=target)

    end_accrein = time.time()
    logger.info("DONE")
    logger.info(f"accrein_ids: {accrein_ids}")

    time_greedy_normal_end = time.time()

    ###### Batched
    time_greedy_batched_start = time.time()
    
    logger.info("Performing Margin Distance Minimization (MDM)  - IN BATCHES .... ")
    start_mdm_batches = time.time()

    ###Margin Distance Minimization (MDM)
    mdm_batches_cl = GreedyPruningClassifier(n_estimators, metric=margin_distance_minimization)
    mdm_batches_ids,_ = mdm_batches_cl.prune_(proba= None, target=None, proba_target_path= proba_target_path)


    end_mdm_batches = time.time()
    logger.info("DONE")
    logger.info(f"mdm_batches_ids: {mdm_batches_ids}")

    logger.info("Performing Uncertainty Weighted Accuracy (UWA) - IN BATCHES .... ")
    start_uwa_batches = time.time()
    
    ###Uncertainty Weighted Accuracy (UWA)
    uwa_batches_cl = GreedyPruningClassifier(n_estimators, metric=uwa)
    uwa_batches_ids,_ = uwa_batches_cl.prune_(proba= None, target=None, proba_target_path= proba_target_path)

    end_uwa_batches = time.time()
    logger.info("DONE")
    logger.info(f"uwa_batches_ids: {uwa_batches_ids}")


    logger.info("Performing Diversity-Accuracy Measure for Homogenous Ensemble Selection (DIACES)  - IN BATCHES .... ")
    start_diaces_batches = time.time()

    ###Diversity-Accuracy Measure for Homogenous Ensemble Selection (DIACES)
    diaces_batches_cl = GreedyPruningClassifier(n_estimators, metric=diaces,  metric_options = {'alpha':1})
    diaces_batches_ids,_ = diaces_batches_cl.prune_(proba= None, target=None, proba_target_path= proba_target_path)


    end_diaces_batches = time.time()
    logger.info("DONE")
    logger.info(f"diaces_batches_ids: {diaces_batches_ids}")


    logger.info("Performing simultaneous diversity & accuracy (SDACC) - IN BATCHES .... ")
    start_sdacc_batches = time.time()

    ###Simultaneous diversity & accuracy (SDACC)
    sdacc_batches_cl = GreedyPruningClassifier(n_estimators, metric=sdacc)
    sdacc_batches_ids,_ = sdacc_batches_cl.prune_(proba= None, target=None, proba_target_path= proba_target_path)

    end_sdacc_batches = time.time()
    logger.info("DONE")
    logger.info(f"sdacc_batches_ids: {sdacc_batches_ids}")



    logger.info("Performing Diversity-focused-two (DFTWO) - IN BATCHES .... ")
    start_dftwo_batches = time.time()


    ###Diversity-focused-two (DFTWO)
    dftwo_batches_cl = GreedyPruningClassifier(n_estimators, metric=dftwo)
    dftwo_batches_ids,_ = dftwo_batches_cl.prune_(proba= None, target=None, proba_target_path= proba_target_path)

    end_dftwo_batches = time.time()
    logger.info("DONE")
    logger.info(f"dftwo_batches_ids: {dftwo_batches_ids}")



    logger.info("Performing Accuracy-reinforcement (ACCREIN) - IN BATCHES .... ")
    start_accrein_batches = time.time()

    ###Accuracy-reinforcement (ACCREIN)
    accrein_batches_cl = GreedyPruningClassifier(n_estimators, metric=accrein)
    accrein_batches_ids,_ = accrein_batches_cl.prune_(proba= None, target=None, proba_target_path= proba_target_path)

    end_accrein_batches = time.time()
    logger.info("DONE")
    logger.info(f"accrein_batches_ids: {accrein_batches_ids}")


    #Remove the folder with temp files now
    shutil.rmtree(proba_target_path)


    time_greedy_batched_end = time.time()
    #endregion

    ####################
    ##  "OTHER"
    ####################
    #region

    if calculate_spectral:
        logger.info("Performing Spectral clustering based ensemble pruning approach (SCP) .... ")
        start_scp = time.time()
       
        ###Spectral clustering based ensemble pruning approach (SCP)
        scp_cl = SpectralClusterPruningClassifier()
        scp_ids = scp_cl.prune_(experiment_path,train_set,batch_size=batch_size,amp=amp,dataset_type=dataset_type,similarity_type = 'gm',lambd= None)



        end_scp = time.time()
        logger.info("DONE")
        logger.info(f"scp_ids: {scp_ids}")
    else:
        scp_ids = np.nan 
        logger.info(f"NOT Performing Spectral clustering based ensemble pruning approach (SCP),option was disactivated in function call")

    ####################
    ##  OPTIMAL (best on based IOU on the pruning set (i.e. the whole val set)
    ####################
    

    if calculate_optimal:
        logger.info(f"Obtaining Optimal Subensemble of size {n_estimators} .... ")
        start_optimal = time.time()
        optimal_ids = -1
        optimal_score_val_set = 0

        #possible sub-ensembles 
        possible_combinations = list(itertools.combinations(range(num_models), n_estimators))

        for sub_ids in possible_combinations:
            new_score = evaluate_ensemble_fusion(val_set, model_paths,selected_models=sub_ids,amp =True, batch_size=batch_size, metric= 'iou',fusion_method='fusion', dataset_type=dataset_type)
            
            #could be the pruning set too 
            #new_score = evaluate_ensemble_fusion(pruning_set, model_paths,selected_models=sub_ids,amp =True, batch_size=batch_size, metric= 'iou',fusion_method='fusion', dataset_type=dataset_type)

            if new_score > optimal_score_val_set:
                optimal_score_val_set = new_score
                optimal_ids = sub_ids
            

        end_optimal = time.time()
        logger.info("DONE")
        logger.info(f"optimal_ids: {optimal_ids}")
    else: 
        optimal_ids = np.nan
        logger.info(f"NOT Obtaining Optimal Subensemble,option was disactivated in function call")

    


    ####################
    ##  RANDOM
    ####################
    logger.info(f"Obtaining Random Subensemble of size {n_estimators} .... ")
    start_random = time.time()
    random_ids = rng.choice(np.arange(num_models),size=n_estimators,replace=False)
    end_random = time.time()
    logger.info("DONE")
    logger.info(f"random_ids: {random_ids}")

    #endregion
    ####################
    ##  EVALUATION
    ####################
    #region

    logger.info("BEGINNING EVALUATION")

    #NON Batched Versions of Ranking/Greedy Algos:
    time_eval_normal_start = time.time()


    oo_scores = evaluate_ensemble_fusion(test_set, model_paths,oo_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    oo_score_vote_iou =              oo_scores[0] 
    oo_score_fusion_iou =            oo_scores[3] 
    oo_score_vote_f1_micro =         oo_scores[2] 
    oo_score_fusion_f1_micro =       oo_scores[5] 
    oo_score_vote_f1_macro =         oo_scores[1] 
    oo_score_fusion_f1_macro =       oo_scores[4] 
    time_oo = end_oo-start_oo


    logger.info("ORIENTATION ORDERING (OO)")
    logger.info(f"oo_score_vote_iou: {oo_score_vote_iou}")
    logger.info(f"oo_score_fusion_iou: {oo_score_fusion_iou}")

   
    epic_scores = evaluate_ensemble_fusion(test_set, model_paths,epic_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    epic_score_vote_iou =              epic_scores[0] 
    epic_score_fusion_iou =            epic_scores[3] 
    epic_score_vote_f1_micro =         epic_scores[2] 
    epic_score_fusion_f1_micro =       epic_scores[5] 
    epic_score_vote_f1_macro =         epic_scores[1] 
    epic_score_fusion_f1_macro =       epic_scores[4] 
    time_epic = end_epic -start_epic + (end_votes-start_votes)

    logger.info("Ensemble pruning via individual contribution ordering (EPIC)")
    logger.info(f"epic_score_vote_iou: {epic_score_vote_iou}")
    logger.info(f"epic_score_fusion_iou: {epic_score_fusion_iou}")

    mdep_scores = evaluate_ensemble_fusion(test_set, model_paths,mdep_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    mdep_score_vote_iou =              mdep_scores[0] 
    mdep_score_fusion_iou =            mdep_scores[3] 
    mdep_score_vote_f1_micro =         mdep_scores[2] 
    mdep_score_fusion_f1_micro =       mdep_scores[5] 
    mdep_score_vote_f1_macro =         mdep_scores[1] 
    mdep_score_fusion_f1_macro =       mdep_scores[4] 
    time_mdep = end_mdep-start_mdep + (end_votes-start_votes)


    logger.info("Margin & Diversity based ordering Ensemble Pruning (MDEP)")
    logger.info(f"mdep_score_vote_iou: {mdep_score_vote_iou}")
    logger.info(f"mdep_score_fusion_iou: {mdep_score_fusion_iou}")

    mdm_scores = evaluate_ensemble_fusion(test_set, model_paths,mdm_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    mdm_score_vote_iou =              mdm_scores[0] 
    mdm_score_fusion_iou =            mdm_scores[3] 
    mdm_score_vote_f1_micro =         mdm_scores[2] 
    mdm_score_fusion_f1_micro =       mdm_scores[5] 
    mdm_score_vote_f1_macro =         mdm_scores[1] 
    mdm_score_fusion_f1_macro =       mdm_scores[4] 
    time_mdm = end_mdm-start_mdm

    logger.info("Margin Distance Minimization (MDM)")
    logger.info(f"mdm_score_vote_iou: {mdm_score_vote_iou}")
    logger.info(f"mdm_score_fusion_iou: {mdm_score_fusion_iou}")

    uwa_scores = evaluate_ensemble_fusion(test_set, model_paths,uwa_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    uwa_score_vote_iou =              uwa_scores[0] 
    uwa_score_fusion_iou =            uwa_scores[3] 
    uwa_score_vote_f1_micro =         uwa_scores[2] 
    uwa_score_fusion_f1_micro =       uwa_scores[5] 
    uwa_score_vote_f1_macro =         uwa_scores[1] 
    uwa_score_fusion_f1_macro =       uwa_scores[4] 
    time_uwa = end_uwa-start_uwa


    logger.info("Uncertainty Weighted Accuracy (UWA)")
    logger.info(f"uwa_score_vote_iou: {uwa_score_vote_iou}")
    logger.info(f"uwa_score_fusion_iou: {uwa_score_fusion_iou}")

    diaces_scores = evaluate_ensemble_fusion(test_set, model_paths,diaces_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    diaces_score_vote_iou =              diaces_scores[0] 
    diaces_score_fusion_iou =            diaces_scores[3] 
    diaces_score_vote_f1_micro =         diaces_scores[2] 
    diaces_score_fusion_f1_micro =       diaces_scores[5] 
    diaces_score_vote_f1_macro =         diaces_scores[1] 
    diaces_score_fusion_f1_macro =       diaces_scores[4] 
    time_diaces = end_diaces -start_diaces

    logger.info("Diversity-Accuracy Measure for Homogenous Ensemble Selection (DIACES)")
    logger.info(f"diaces_score_vote_iou: {diaces_score_vote_iou}")
    logger.info(f"diaces_score_fusion_iou: {diaces_score_fusion_iou}")



    sdacc_scores = evaluate_ensemble_fusion(test_set, model_paths,sdacc_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    sdacc_score_vote_iou =              sdacc_scores[0] 
    sdacc_score_fusion_iou =            sdacc_scores[3] 
    sdacc_score_vote_f1_micro =         sdacc_scores[2] 
    sdacc_score_fusion_f1_micro =       sdacc_scores[5] 
    sdacc_score_vote_f1_macro =         sdacc_scores[1] 
    sdacc_score_fusion_f1_macro =       sdacc_scores[4] 
    time_sdacc = end_sdacc -start_sdacc

    logger.info("Simultaneous diversity & accuracy (SDACC)")
    logger.info(f"sdacc_score_vote_iou: {sdacc_score_vote_iou}")
    logger.info(f"sdacc_score_fusion_iou: {sdacc_score_fusion_iou}")


    dftwo_scores = evaluate_ensemble_fusion(test_set, model_paths,dftwo_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    dftwo_score_vote_iou =              dftwo_scores[0] 
    dftwo_score_fusion_iou =            dftwo_scores[3] 
    dftwo_score_vote_f1_micro =         dftwo_scores[2] 
    dftwo_score_fusion_f1_micro =       dftwo_scores[5] 
    dftwo_score_vote_f1_macro =         dftwo_scores[1] 
    dftwo_score_fusion_f1_macro =       dftwo_scores[4] 
    time_dftwo = end_dftwo -start_dftwo

    logger.info("Diversity-focused-two (DFTWO)")
    logger.info(f"dftwo_score_vote_iou: {dftwo_score_vote_iou}")
    logger.info(f"dftwo_score_fusion_iou: {dftwo_score_fusion_iou}")


    accrein_scores = evaluate_ensemble_fusion(test_set, model_paths,accrein_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    accrein_score_vote_iou =              accrein_scores[0] 
    accrein_score_fusion_iou =            accrein_scores[3] 
    accrein_score_vote_f1_micro =         accrein_scores[2] 
    accrein_score_fusion_f1_micro =       accrein_scores[5] 
    accrein_score_vote_f1_macro =         accrein_scores[1] 
    accrein_score_fusion_f1_macro =       accrein_scores[4] 
    time_accrein = end_accrein -start_accrein

    logger.info("Accuracy-reinforcement (ACCREIN)")
    logger.info(f"accrein_score_vote_iou: {accrein_score_vote_iou}")
    logger.info(f"accrein_score_fusion_iou: {accrein_score_fusion_iou}")

    time_eval_normal_end = time.time()

    #Batched Versions of Ranking/Greedy Algos:

    time_eval_batched_start = time.time()

    oo_batches_scores = evaluate_ensemble_fusion(test_set, model_paths,oo_batches_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    oo_batches_score_vote_iou =              oo_batches_scores[0] 
    oo_batches_score_fusion_iou =            oo_batches_scores[3] 
    oo_batches_score_vote_f1_micro =         oo_batches_scores[2] 
    oo_batches_score_fusion_f1_micro =       oo_batches_scores[5] 
    oo_batches_score_vote_f1_macro =         oo_batches_scores[1] 
    oo_batches_score_fusion_f1_macro =       oo_batches_scores[4] 
    time_oo_batches = end_oo_batches-start_oo_batches


    logger.info("ORIENTATION ORDERING (oo_batches)")
    logger.info(f"oo_batches_score_vote_iou: {oo_batches_score_vote_iou}")
    logger.info(f"oo_batches_score_fusion_iou: {oo_batches_score_fusion_iou}")

    epic_batches_scores = evaluate_ensemble_fusion(test_set, model_paths,epic_batches_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    epic_batches_score_vote_iou =              epic_batches_scores[0] 
    epic_batches_score_fusion_iou =            epic_batches_scores[3] 
    epic_batches_score_vote_f1_micro =         epic_batches_scores[2] 
    epic_batches_score_fusion_f1_micro =       epic_batches_scores[5] 
    epic_batches_score_vote_f1_macro =         epic_batches_scores[1] 
    epic_batches_score_fusion_f1_macro =       epic_batches_scores[4] 
    time_epic_batches = end_epic_batches -start_epic_batches + (end_votes-start_votes)

    logger.info("Ensemble pruning via individual contribution ordering (epic_batches)")
    logger.info(f"epic_batches_score_vote_iou: {epic_batches_score_vote_iou}")
    logger.info(f"epic_batches_score_fusion_iou: {epic_batches_score_fusion_iou}")

    mdep_batches_scores = evaluate_ensemble_fusion(test_set, model_paths,mdep_batches_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    mdep_batches_score_vote_iou =              mdep_batches_scores[0] 
    mdep_batches_score_fusion_iou =            mdep_batches_scores[3] 
    mdep_batches_score_vote_f1_micro =         mdep_batches_scores[2] 
    mdep_batches_score_fusion_f1_micro =       mdep_batches_scores[5] 
    mdep_batches_score_vote_f1_macro =         mdep_batches_scores[1] 
    mdep_batches_score_fusion_f1_macro =       mdep_batches_scores[4] 
    time_mdep_batches = end_mdep_batches-start_mdep_batches + (end_votes-start_votes)

    logger.info("Margin & Diversity based ordering Ensemble Pruning (mdep_batches)")
    logger.info(f"mdep_batches_score_vote_iou: {mdep_batches_score_vote_iou}")
    logger.info(f"mdep_batches_score_fusion_iou: {mdep_batches_score_fusion_iou}")

    mdm_batches_scores = evaluate_ensemble_fusion(test_set, model_paths,mdm_batches_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    mdm_batches_score_vote_iou =              mdm_batches_scores[0] 
    mdm_batches_score_fusion_iou =            mdm_batches_scores[3] 
    mdm_batches_score_vote_f1_micro =         mdm_batches_scores[2] 
    mdm_batches_score_fusion_f1_micro =       mdm_batches_scores[5] 
    mdm_batches_score_vote_f1_macro =         mdm_batches_scores[1] 
    mdm_batches_score_fusion_f1_macro =       mdm_batches_scores[4] 
    time_mdm_batches = end_mdm_batches-start_mdm_batches

    logger.info("Margin Distance Minimization (mdm_batches)")
    logger.info(f"mdm_batches_score_vote_iou: {mdm_batches_score_vote_iou}")
    logger.info(f"mdm_batches_score_fusion_iou: {mdm_batches_score_fusion_iou}")

    uwa_batches_scores = evaluate_ensemble_fusion(test_set, model_paths,uwa_batches_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    uwa_batches_score_vote_iou =              uwa_batches_scores[0] 
    uwa_batches_score_fusion_iou =            uwa_batches_scores[3] 
    uwa_batches_score_vote_f1_micro =         uwa_batches_scores[2] 
    uwa_batches_score_fusion_f1_micro =       uwa_batches_scores[5] 
    uwa_batches_score_vote_f1_macro =         uwa_batches_scores[1] 
    uwa_batches_score_fusion_f1_macro =       uwa_batches_scores[4] 
    time_uwa_batches = end_uwa_batches-start_uwa_batches


    logger.info("Uncertainty Weighted Accuracy (uwa_batches)")
    logger.info(f"uwa_batches_score_vote_iou: {uwa_batches_score_vote_iou}")
    logger.info(f"uwa_batches_score_fusion_iou: {uwa_batches_score_fusion_iou}")

    diaces_batches_scores = evaluate_ensemble_fusion(test_set, model_paths,diaces_batches_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    diaces_batches_score_vote_iou =              diaces_batches_scores[0] 
    diaces_batches_score_fusion_iou =            diaces_batches_scores[3] 
    diaces_batches_score_vote_f1_micro =         diaces_batches_scores[2] 
    diaces_batches_score_fusion_f1_micro =       diaces_batches_scores[5] 
    diaces_batches_score_vote_f1_macro =         diaces_batches_scores[1] 
    diaces_batches_score_fusion_f1_macro =       diaces_batches_scores[4] 
    time_diaces_batches = end_diaces_batches - start_diaces_batches

    logger.info("Diversity-Accuracy Measure for Homogenous Ensemble Selection (diaces_batches)")
    logger.info(f"diaces_batches_score_vote_iou: {diaces_batches_score_vote_iou}")
    logger.info(f"diaces_batches_score_fusion_iou: {diaces_batches_score_fusion_iou}")


    sdacc_batches_scores = evaluate_ensemble_fusion(test_set, model_paths,sdacc_batches_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    sdacc_batches_score_vote_iou =              sdacc_batches_scores[0] 
    sdacc_batches_score_fusion_iou =            sdacc_batches_scores[3] 
    sdacc_batches_score_vote_f1_micro =         sdacc_batches_scores[2] 
    sdacc_batches_score_fusion_f1_micro =       sdacc_batches_scores[5] 
    sdacc_batches_score_vote_f1_macro =         sdacc_batches_scores[1] 
    sdacc_batches_score_fusion_f1_macro =       sdacc_batches_scores[4] 
    time_sdacc_batches = end_sdacc_batches -start_sdacc_batches

    logger.info("Simultaneous diversity & accuracy (SDACC) (sdacc_batches)")
    logger.info(f"sdacc_batches_score_vote_iou: {sdacc_batches_score_vote_iou}")
    logger.info(f"sdacc_batches_score_fusion_iou: {sdacc_batches_score_fusion_iou}")


    dftwo_batches_scores = evaluate_ensemble_fusion(test_set, model_paths,dftwo_batches_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    dftwo_batches_score_vote_iou =              dftwo_batches_scores[0] 
    dftwo_batches_score_fusion_iou =            dftwo_batches_scores[3] 
    dftwo_batches_score_vote_f1_micro =         dftwo_batches_scores[2] 
    dftwo_batches_score_fusion_f1_micro =       dftwo_batches_scores[5] 
    dftwo_batches_score_vote_f1_macro =         dftwo_batches_scores[1] 
    dftwo_batches_score_fusion_f1_macro =       dftwo_batches_scores[4] 
    time_dftwo_batches = end_dftwo_batches -start_dftwo_batches

    logger.info("Diversity-focused-two (DFTWO) (dftwo_batches)")
    logger.info(f"dftwo_batches_score_vote_iou: {dftwo_batches_score_vote_iou}")
    logger.info(f"dftwo_batches_score_fusion_iou: {dftwo_batches_score_fusion_iou}")


    accrein_batches_scores = evaluate_ensemble_fusion(test_set, model_paths,accrein_batches_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    accrein_batches_score_vote_iou =              accrein_batches_scores[0] 
    accrein_batches_score_fusion_iou =            accrein_batches_scores[3] 
    accrein_batches_score_vote_f1_micro =         accrein_batches_scores[2] 
    accrein_batches_score_fusion_f1_micro =       accrein_batches_scores[5] 
    accrein_batches_score_vote_f1_macro =         accrein_batches_scores[1] 
    accrein_batches_score_fusion_f1_macro =       accrein_batches_scores[4] 
    time_accrein_batches = end_accrein_batches -start_accrein_batches

    logger.info("Accuracy-reinforcement (ACCREIN) (accrein_batches)")
    logger.info(f"accrein_batches_score_vote_iou: {accrein_batches_score_vote_iou}")
    logger.info(f"accrein_batches_score_fusion_iou: {accrein_batches_score_fusion_iou}")




    time_eval_batched_end = time.time()



    ####BENCHMARKS (Whole Ensemble, Random Ensemble, Best/Average/Worst Single Classifier etc.)
    time_eval_other_start = time.time()
    
    if calculate_benchmarks:
        all_models_ids =  list(range(num_models))

        all_models_scores = evaluate_ensemble_fusion(test_set, model_paths,all_models_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
        all_models_score_vote_iou =              all_models_scores[0] 
        all_models_score_fusion_iou =            all_models_scores[3] 
        all_models_score_vote_f1_micro =         all_models_scores[2] 
        all_models_score_fusion_f1_micro =       all_models_scores[5] 
        all_models_score_vote_f1_macro =         all_models_scores[1] 
        all_models_score_fusion_f1_macro =       all_models_scores[4] 
    else: 
        all_models_ids              =          np.nan
        all_models_score_vote_iou =            np.nan 
        all_models_score_fusion_iou =          np.nan 
        all_models_score_vote_f1_micro =       np.nan 
        all_models_score_fusion_f1_micro =     np.nan 
        all_models_score_vote_f1_macro =       np.nan 
        all_models_score_fusion_f1_macro =     np.nan 
    

    logger.info("Whole Ensemble")
    logger.info(f"all_models_score_vote_iou: {all_models_score_vote_iou}")
    logger.info(f"all_models_score_fusion_iou: {all_models_score_fusion_iou}")

    
    random_scores = evaluate_ensemble_fusion(test_set, model_paths,random_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    random_score_vote_iou =              random_scores[0] 
    random_score_fusion_iou =            random_scores[3] 
    random_score_vote_f1_micro =         random_scores[2] 
    random_score_fusion_f1_micro =       random_scores[5] 
    random_score_vote_f1_macro =         random_scores[1] 
    random_score_fusion_f1_macro =       random_scores[4] 
    time_random = end_random - start_random


    logger.info("Random Ensemble")
    logger.info(f"random_score_vote_iou: {random_score_vote_iou}")
    logger.info(f"random_score_fusion_iou: {random_score_fusion_iou}")

    if calculate_benchmarks:
        #determine worst, average and best, single score
        individual_scores_iou = []
        individual_scores_f1_micro = []
        individual_scores_f1_macro= []
        average_score_f1_micro = 0
        average_score_f1_macro = 0
        average_score_iou = 0
        for i in range(num_models):
            scores = evaluate_ensemble_fusion(test_set, model_paths,[i],amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
            score_iou =            scores[3] 
            score_f1_micro =       scores[5]
            score_f1_macro =       scores[4] 
            individual_scores_iou.append((i,score_iou))
            individual_scores_f1_micro.append((i,score_f1_micro))
            individual_scores_f1_macro.append((i,score_f1_macro))
            average_score_f1_micro += score_f1_micro
            average_score_f1_macro += score_f1_macro 
            average_score_iou += score_iou
            


        average_score_f1_micro = average_score_f1_micro/num_models
        average_score_f1_macro = average_score_f1_macro/num_models
        average_score_iou = average_score_iou/num_models

        
        best_single_id_iou, best_single_score_iou = max(individual_scores_iou, key= lambda t: t[1])
        worst_single_id_iou, worst_single_score_iou = min(individual_scores_iou, key= lambda t: t[1])

        best_single_id_f1_micro, best_single_score_f1_micro = max(individual_scores_f1_micro, key= lambda t: t[1])
        worst_single_id_f1_micro, worst_single_score_f1_micro = min(individual_scores_f1_micro, key= lambda t: t[1])

        best_single_id_f1_macro, best_single_score_f1_macro = max(individual_scores_f1_macro, key= lambda t: t[1])
        worst_single_id_f1_macro, worst_single_score_f1_macro = min(individual_scores_f1_macro, key= lambda t: t[1])
    else:
        average_score_f1_micro =        np.nan
        average_score_f1_macro =        np.nan
        average_score_iou =             np.nan

        
        best_single_id_iou, best_single_score_iou = np.nan, np.nan
        worst_single_id_iou, worst_single_score_iou = np.nan, np.nan
        best_single_id_f1_micro, best_single_score_f1_micro = np.nan, np.nan
        worst_single_id_f1_micro, worst_single_score_f1_micro = np.nan, np.nan
        best_single_id_f1_macro, best_single_score_f1_macro = np.nan, np.nan
        worst_single_id_f1_macro, worst_single_score_f1_macro = np.nan, np.nan
    
    
    logger.info("BEST SINGLE MODEL")
    logger.info(f"best_single_score: {best_single_score_iou}")

    logger.info("WORST SINGLE MODEL")
    logger.info(f"worst_single_score: {worst_single_score_iou}")

    logger.info("AVERAGE SINGLE MODEL")
    logger.info(f"average_single_score: {average_score_iou}")




    
    #performance of optimal ensemble on test set
    if calculate_optimal:
        optimal_scores = evaluate_ensemble_fusion(test_set, model_paths,optimal_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
        optimal_score_vote_iou =              optimal_scores[0] 
        optimal_score_fusion_iou =            optimal_scores[3] 
        optimal_score_vote_f1_micro =         optimal_scores[2] 
        optimal_score_fusion_f1_micro =       optimal_scores[5] 
        optimal_score_vote_f1_macro =         optimal_scores[1] 
        optimal_score_fusion_f1_macro =       optimal_scores[4] 
        time_optimal = end_optimal - start_optimal
        len_optimal_ids = len(optimal_ids)
        
        logger.info("Optimal Ensemble")
        logger.info(f"optimal_score_vote_iou: {optimal_score_vote_iou}")
        logger.info(f"optimal_score_fusion_iou: {optimal_score_fusion_iou}")
        
    
    else:
        optimal_score_vote_iou = np.nan
        optimal_score_fusion_iou = np.nan
        optimal_score_vote_f1_micro = np.nan
        optimal_score_fusion_f1_micro = np.nan
        optimal_score_vote_f1_macro = np.nan
        optimal_score_fusion_f1_macro = np.nan
        time_optimal = np.nan
        len_optimal_ids = np.nan



    #performance of Spectral Clustering Pruning 
    if calculate_spectral:
        scp_scores = evaluate_ensemble_fusion(test_set, model_paths,scp_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
        scp_score_vote_iou =              scp_scores[0] 
        scp_score_fusion_iou =            scp_scores[3] 
        scp_score_vote_f1_micro =         scp_scores[2] 
        scp_score_fusion_f1_micro =       scp_scores[5] 
        scp_score_vote_f1_macro =         scp_scores[1] 
        scp_score_fusion_f1_macro =       scp_scores[4] 
        time_scp = end_scp - start_scp
        len_scp_ids = len(scp_ids)
        logger.info("Spectral clustering based ensemble pruning approach (SCP) .... ")
        logger.info(f"scp_score_vote_iou: {scp_score_vote_iou}")
        logger.info(f"scp_score_fusion_iou: {scp_score_fusion_iou}")
    else:
        scp_score_vote_iou = np.nan
        scp_score_fusion_iou = np.nan
        scp_score_vote_f1_micro = np.nan
        scp_score_fusion_f1_micro = np.nan
        scp_score_vote_f1_macro = np.nan
        scp_score_fusion_f1_macro  = np.nan
        time_scp = np.nan
        len_scp_ids = np.nan


    time_eval_other_end = time.time()
    time_total_end = time.time()

    logger.info("EVALUATION DONE")
    logger.info("Track Results in eval df")

    logger.info("TIME OVERVIEW")
    logger.info(f"ENSEMBLE PROBA CREATION:              {time_ensemble_proba_end-time_ensemble_proba_start}")
    logger.info(f"RANKING (NORMAL):                     {time_rank_normal_end-time_rank_normal_start}")
    logger.info(f"RANKING (BATCHED):                    {time_rank_batched_end-time_rank_batched_start}")
    logger.info(f"GREEDY (NORMAL):                      {time_greedy_normal_end-time_greedy_normal_start}")
    logger.info(f"GREEDY (BATCHED):                     {time_greedy_batched_end-time_greedy_batched_start}")
    logger.info(f"EVAL (NORMAL):                        {time_eval_normal_end- time_eval_normal_start}")
    logger.info(f"EVAL (BATCHED):                       {time_eval_batched_end- time_eval_batched_start}")
    logger.info(f"EVAL (OTHER):                         {time_eval_other_end- time_eval_other_start}")
    logger.info(f"TOTAL:                                {time_total_end- time_total_start}")


    



   



    # COLUMNS:                                              Chosen IDS  -                                   Ensemble Size                     Score Vote IOU -                     Score Fusion IOU                           Score Vote F1 MACRO                            Score Fusion F1 MACRO                   Score Vote F1 MICRO -                             Score Fusion F1 MICRO                            TIME

    eval_dict = {                                                                                       
        'OO':                                               [oo_ids,                                       len(oo_ids),                       oo_score_vote_iou,                  oo_score_fusion_iou,                        oo_score_vote_f1_macro,                         oo_score_fusion_f1_macro,               oo_score_vote_f1_micro,                         oo_score_fusion_f1_micro,                         time_oo                         ],
        'EPIC':                                             [epic_ids,                                     len(epic_ids),                     epic_score_vote_iou,                epic_score_fusion_iou,                      epic_score_vote_f1_macro,                       epic_score_fusion_f1_macro,             epic_score_vote_f1_micro,                       epic_score_fusion_f1_micro,                       time_epic                       ],
        'MDEP':                                             [mdep_ids,                                     len(mdep_ids),                     mdep_score_vote_iou,                mdep_score_fusion_iou,                      mdep_score_vote_f1_macro,                       mdep_score_fusion_f1_macro,             mdep_score_vote_f1_micro,                       mdep_score_fusion_f1_micro,                       time_mdep                       ],
        'MDM':                                              [mdm_ids,                                      len(mdm_ids),                      mdm_score_vote_iou,                 mdm_score_fusion_iou,                       mdm_score_vote_f1_macro,                        mdm_score_fusion_f1_macro,              mdm_score_vote_f1_micro,                        mdm_score_fusion_f1_micro,                        time_mdm                        ],
        'UWA':                                              [uwa_ids,                                      len(uwa_ids),                      uwa_score_vote_iou,                 uwa_score_fusion_iou,                       uwa_score_vote_f1_macro,                        uwa_score_fusion_f1_macro,              uwa_score_vote_f1_micro,                        uwa_score_fusion_f1_micro,                        time_uwa                        ],
        'DIACES':                                           [diaces_ids,                                   len(diaces_ids),                   diaces_score_vote_iou,              diaces_score_fusion_iou,                    diaces_score_vote_f1_macro,                     diaces_score_fusion_f1_macro,           diaces_score_vote_f1_micro,                     diaces_score_fusion_f1_micro,                     time_diaces                     ],
        'SDACC':                                            [sdacc_ids,                                    len(sdacc_ids),                    sdacc_score_vote_iou,               sdacc_score_fusion_iou,                     sdacc_score_vote_f1_macro,                      sdacc_score_fusion_f1_macro,            sdacc_score_vote_f1_micro,                      sdacc_score_fusion_f1_micro,                      time_sdacc                      ],
        'DFTWO':                                            [dftwo_ids,                                    len(dftwo_ids),                    dftwo_score_vote_iou,               dftwo_score_fusion_iou,                     dftwo_score_vote_f1_macro,                      dftwo_score_fusion_f1_macro,            dftwo_score_vote_f1_micro,                      dftwo_score_fusion_f1_micro,                      time_dftwo                      ],
        'ACCREIN':                                          [accrein_ids,                                  len(accrein_ids),                  accrein_score_vote_iou,             accrein_score_fusion_iou,                   accrein_score_vote_f1_macro,                    accrein_score_fusion_f1_macro,          accrein_score_vote_f1_micro,                    accrein_score_fusion_f1_micro,                    time_accrein                    ],
        'OO-BATCHES':                                       [oo_batches_ids,                               len(oo_batches_ids),               oo_batches_score_vote_iou,          oo_batches_score_fusion_iou,                oo_batches_score_vote_f1_macro,                 oo_batches_score_fusion_f1_macro,       oo_batches_score_vote_f1_micro,                 oo_batches_score_fusion_f1_micro,                 time_oo_batches                 ],
        'EPIC-BATCHES':                                     [epic_batches_ids,                             len(epic_batches_ids),             epic_batches_score_vote_iou,        epic_batches_score_fusion_iou,              epic_batches_score_vote_f1_macro,               epic_batches_score_fusion_f1_macro,     epic_batches_score_vote_f1_micro,               epic_batches_score_fusion_f1_micro,               time_epic_batches               ],
        'MDEP-BATCHES':                                     [mdep_batches_ids,                             len(mdep_batches_ids),             mdep_batches_score_vote_iou,        mdep_batches_score_fusion_iou,              mdep_batches_score_vote_f1_macro,               mdep_batches_score_fusion_f1_macro,     mdep_batches_score_vote_f1_micro,               mdep_batches_score_fusion_f1_micro,               time_mdep_batches               ],
        'MDM-BATCHES':                                      [mdm_batches_ids,                              len(mdm_batches_ids),              mdm_batches_score_vote_iou,         mdm_batches_score_fusion_iou,               mdm_batches_score_vote_f1_macro,                mdm_batches_score_fusion_f1_macro,      mdm_batches_score_vote_f1_micro,                mdm_batches_score_fusion_f1_micro,                time_mdm_batches                ],
        'UWA-BATCHES':                                      [uwa_batches_ids,                              len(uwa_batches_ids),              uwa_batches_score_vote_iou,         uwa_batches_score_fusion_iou,               uwa_batches_score_vote_f1_macro,                uwa_batches_score_fusion_f1_macro,      uwa_batches_score_vote_f1_micro,                uwa_batches_score_fusion_f1_micro,                time_uwa_batches                ],
        'DIACES-BATCHES':                                   [diaces_batches_ids,                           len(diaces_batches_ids),           diaces_batches_score_vote_iou,      diaces_batches_score_fusion_iou,            diaces_batches_score_vote_f1_macro,             diaces_batches_score_fusion_f1_macro,   diaces_batches_score_vote_f1_micro,             diaces_batches_score_fusion_f1_micro,             time_diaces_batches             ],
        'SDACC-BATCHES':                                    [sdacc_batches_ids,                            len(sdacc_batches_ids),            sdacc_batches_score_vote_iou,       sdacc_batches_score_fusion_iou,             sdacc_batches_score_vote_f1_macro,              sdacc_batches_score_fusion_f1_macro,    sdacc_batches_score_vote_f1_micro,              sdacc_batches_score_fusion_f1_micro,              time_sdacc_batches              ],
        'DFTWO-BATCHES':                                    [dftwo_batches_ids,                            len(dftwo_batches_ids),            dftwo_batches_score_vote_iou,       dftwo_batches_score_fusion_iou,             dftwo_batches_score_vote_f1_macro,              dftwo_batches_score_fusion_f1_macro,    dftwo_batches_score_vote_f1_micro,              dftwo_batches_score_fusion_f1_micro,              time_dftwo_batches              ],
        'ACCREIN-BATCHES':                                  [accrein_batches_ids,                          len(accrein_batches_ids),          accrein_batches_score_vote_iou,     accrein_batches_score_fusion_iou,           accrein_batches_score_vote_f1_macro,            accrein_batches_score_fusion_f1_macro,  accrein_batches_score_vote_f1_micro,            accrein_batches_score_fusion_f1_micro,            time_accrein_batches            ],
        'SCP':                                              [scp_ids,                                      len_scp_ids,                       scp_score_vote_iou,                 scp_score_fusion_iou,                       scp_score_vote_f1_macro,                        scp_score_fusion_f1_macro,              scp_score_vote_f1_micro,                        scp_score_fusion_f1_micro,                        time_scp                        ],
        'RANDOM ENSEMBLE':                                  [random_ids,                                   len(random_ids),                   random_score_vote_iou,              random_score_fusion_iou,                    random_score_vote_f1_macro,                     random_score_fusion_f1_macro,           random_score_vote_f1_micro,                     random_score_fusion_f1_micro,                     np.nan                          ],
        'ALL MODELS':                                       [list(range(num_models)),                      num_models,                        all_models_score_vote_iou,          all_models_score_fusion_iou,                all_models_score_vote_f1_macro,                 all_models_score_fusion_f1_macro,       all_models_score_vote_f1_micro,                 all_models_score_fusion_f1_micro,                 np.nan                          ],
        'BEST SINGLE MODEL (IOU) (on test set)':            [best_single_id_iou,                           np.nan,                            best_single_score_iou,              best_single_score_iou,                      np.nan,                                         np.nan,                                 np.nan,                                         np.nan,                                           np.nan                          ],
        'WORST SINGLE MODEL (IOU) (on test set)':           [worst_single_id_iou,                          np.nan,                            worst_single_score_iou,             worst_single_score_iou,                     np.nan,                                         np.nan,                                 np.nan,                                         np.nan,                                           np.nan                          ],
        'BEST SINGLE MODEL (F1 MICRO) (on test set)':       [best_single_id_f1_micro,                      np.nan,                            np.nan,                             np.nan,                                     np.nan,                                         np.nan,                                 best_single_score_f1_micro,                     best_single_score_f1_micro,                       np.nan                          ],
        'WORST SINGLE MODEL (F1 MICRO) (on test set)':      [worst_single_id_f1_micro,                     np.nan,                            np.nan,                             np.nan,                                     np.nan,                                         np.nan,                                 worst_single_score_f1_micro,                    worst_single_score_f1_micro,                      np.nan                          ],
        'BEST SINGLE MODEL (F1 MACRO) (on test set)':       [best_single_id_f1_macro,                      np.nan,                            np.nan,                             np.nan,                                     best_single_score_f1_macro,                     best_single_score_f1_macro,             np.nan,                                         np.nan,                                           np.nan                          ],
        'WORST SINGLE MODEL (F1 MACRO) (on test set)':      [worst_single_id_f1_macro,                     np.nan,                            np.nan,                             np.nan,                                     worst_single_score_f1_macro,                    worst_single_score_f1_macro,            np.nan,                                         np.nan,                                           np.nan                          ],
        'AVERAGE SINGLE MODEL (on test set)':               [np.nan,                                       np.nan,                            average_score_iou,                  average_score_iou,                          average_score_f1_macro,                         average_score_f1_macro,                 average_score_f1_micro,                         average_score_f1_micro,                           np.nan                          ],
        'OPTIMAL ENSEMBLE (based on pruning set)':          [optimal_ids,                                  len_optimal_ids,                  optimal_score_vote_iou,             optimal_score_fusion_iou,                   optimal_score_vote_f1_macro,                    optimal_score_fusion_f1_macro,          optimal_score_vote_f1_micro,                    optimal_score_fusion_f1_micro,                    time_optimal                    ]
    }
    

    eval_df = pd.DataFrame.from_dict(eval_dict, orient= 'index', columns = ['Chosen Models','Ensemble Size','IoU - VOTING', 'IoU POSTERIOR AVERAGE', 'F1 MACRO - VOTING', 'F1 MACRO POSTERIOR AVERAGE','F1 MICRO - VOTING', 'F1 MICRO POSTERIOR AVERAGE','Execution Time'])


    return eval_df

    #endregion




def exploratory_mdep_alpha(logger, experiment_path, val_set, test_set, sub_sample_size,n_estimators, batch_size, amp,n_jobs, dataset_type = 'landcover'):
    """
    Performs exploratory experiments for the MDEP metric for different alpha values. 

    Args:
        logger (logging.Logger): Logger instance for logging the progress of the experiment.
        experiment_path (str or Path): Path to the experiment directory containing the models.
        val_set (torch.utils.data.Dataset): Validation dataset used for generating the pruning set.
        test_set (torch.utils.data.Dataset): Test dataset used for evaluating the pruned ensemble.
        sub_sample_size (int): Number of samples (i.e., images) to randomly select from the validation set for pruning.
        n_estimators (int): Number of classifiers to be selected from the ensemble.
        batch_size (int): Batch size for processing the dataset during ensemble probability generation and evaluation.
        amp (bool): Whether to use automatic mixed precision during evaluation.
        n_jobs (int): Number of parallel jobs to use during pruning and evaluation.
        dataset_type (strl): Type of dataset ('landcover' or 'floods'). Default is 'landcover'.

    Returns:
        pd.DataFrame: DataFrame containing the evaluation results for different alpha values in the MDEP experiment. 
    """
   
    experiment_path = Path(experiment_path)
    model_paths = get_model_paths(experiment_path)
    num_models = len(model_paths)
    
    
    #sub samples a Pruning Set from the VAL Set
    rng = np.random.default_rng()
    sample_index = rng.choice(np.arange(len(val_set)),size=sub_sample_size,replace=False)
    pruning_set = torch.utils.data.Subset(val_set, sample_index)

    logger.info(f"Images at these inidces chosen randomly for pruning: {sample_index}")
    
    logger.info("Generating the Ensemble Probability Table .....")

    #generate the ensemble proba table which is used for most pruning algos , which has shape (Classifiers, Sample_Number, Classes) or (M,N,C)
    ensemble_proba, target = generate_ensemble_proba_target(model_paths, pruning_set,batch_size= batch_size, amp=amp, dataset_type=dataset_type)
    ensemble_proba = ensemble_proba.cpu().numpy()
    target = target.numpy()

    logger.info("DONE")


    ####################
    ##  MDEP
    #################### 

    #between 0 and 1 (according to paper)
    alphas = [0.1,0.2,0.4,0.8,0.9]


    logger.info("Creating Votes Table.....")
   

    ###calculate Vote array of shape (N,C) here for higher efficiency  
    V = np.zeros(ensemble_proba.shape)
    idx = ensemble_proba.argmax(axis=2)
    V[np.arange(ensemble_proba.shape[0])[:,None],np.arange(ensemble_proba.shape[1]),idx] = 1
    V = V.sum(axis=0)
    
    logger.info("DONE")
    
    logger.info("Performing Margin & Diversity based ordering Ensemble Pruning (MDEP) .... ")
    

    mdep_ids = []
    for alpha in alphas:
        ###Margin & Diversity based ordering Ensemble Pruning (MDEP)
        mdep_cl = RankPruningClassifier(n_estimators, metric = individual_margin_diversity, n_jobs = n_jobs, metric_options = {'alpha': alpha, 'V': V})
        mdep_id, _ = mdep_cl.prune_(proba= ensemble_proba, target=target)

        
        logger.info("DONE")
        logger.info(f"mdep_id: {mdep_id}; alpha: {alpha}")
        mdep_ids.append(mdep_id)


    fusion_scores = []
    vote_scores = []
    for eval_id in mdep_ids:
        mdep_score_vote_iou = evaluate_ensemble_fusion(test_set, model_paths,eval_id,amp=amp, batch_size=batch_size, metric= 'iou',fusion_method='vote', dataset_type=dataset_type)
        mdep_score_fusion_iou = evaluate_ensemble_fusion(test_set, model_paths,eval_id,amp=amp, batch_size=batch_size, metric= 'iou',fusion_method='fusion', dataset_type=dataset_type)
        

        mdep_score_vote_iou = mdep_score_vote_iou.item()
        mdep_score_fusion_iou = mdep_score_fusion_iou.item()
       

        logger.info("Margin & Diversity based ordering Ensemble Pruning (MDEP)")
        logger.info(f"mdep_score_vote_iou: {mdep_score_vote_iou}")
        logger.info(f"mdep_score_fusion_iou: {mdep_score_fusion_iou})")

        fusion_scores.append(mdep_score_fusion_iou)
        vote_scores.append(mdep_score_vote_iou)

    eval_dict = {}
    for i, alpha in enumerate(alphas):
        eval_dict[f'MDEP - {alpha}'] = [mdep_ids[i], len(mdep_ids[i]), vote_scores[i], fusion_scores[i], alpha]
    
    eval_df = pd.DataFrame.from_dict(eval_dict, orient= 'index', columns = ['Chosen Models','Ensemble Size', 'IoU - VOTING', 'IoU POSTERIOR AVERAGE','Alpha'])


    return eval_df


def exploratory_diaces_alpha(logger, experiment_path, val_set, test_set, sub_sample_size,n_estimators, batch_size, amp,n_jobs, dataset_type = 'landcover'):
    """
    Performs exploratory experiments for the DIACES algorithm (different values for alpha)

    Args:
        logger (logging.Logger): Logger instance for logging progress and results of the analysis.
        experiment_path (str or Path): Path to the experiment directory containing the models.
        val_set (torch.utils.data.Dataset): Validation dataset used for generating the pruning set.
        test_set (torch.utils.data.Dataset): Test dataset used for evaluating the pruned ensemble.
        sub_sample_size (int): Number of samples (i.e., images) to randomly select from the validation set for pruning.
        n_estimators (int): Number of classifiers to be selected from the ensemble for pruning.
        batch_size (int): Batch size for processing the datasets during pruning and evaluation.
        amp (bool): Whether to use automatic mixed precision
        n_jobs (int): Number of parallel jobs to use during pruning and evaluation.
        dataset_type (str): Type of dataset ('landcover' or 'floods'). Default is 'landcover'.

    Returns:
        pd.DataFrame: DataFrame summarizing the evaluation results for each alpha value
    """
    experiment_path = Path(experiment_path)
    model_paths = get_model_paths(experiment_path)
    num_models = len(model_paths)
    
    
    #sub samples a Pruning Set from the VAL Set
    rng = np.random.default_rng()
    sample_index = rng.choice(np.arange(len(val_set)),size=sub_sample_size,replace=False)
    pruning_set = torch.utils.data.Subset(val_set, sample_index)

    logger.info(f"Images at these inidces chosen randomly for pruning: {sample_index}")
    
    logger.info("Generating the Ensemble Probability Table .....")

    #generate the ensemble proba table which is used for most pruning algos , which has shape (Classifiers, Sample_Number, Classes) or (M,N,C)
    ensemble_proba, target = generate_ensemble_proba_target(model_paths, pruning_set,batch_size= batch_size, amp=amp, dataset_type=dataset_type)
    ensemble_proba = ensemble_proba.cpu().numpy()
    target = target.numpy()

    logger.info("DONE")


    ####################
    ##  DIACES
    #################### 
    
    
    alphas = [0.1,0.2,0.4,0.8,0.9]


    
    diaces_ids = []
    for alpha in alphas:
        ###Diversity-Accuracy Measure for Homogenous Ensemble Selection (DIACES)
        diaces_cl = GreedyPruningClassifier(n_estimators, metric=diaces,  metric_options = {'alpha':1})
        diaces_id,_ = diaces_cl.prune_(proba= ensemble_proba, target=target)

        logger.info("DONE")
        logger.info(f"diaces_ids: {diaces_id}")

        
        diaces_ids.append(diaces_id)


    fusion_scores = []
    vote_scores = []
    for eval_id in diaces_ids:
        diaces_score_vote_iou = evaluate_ensemble_fusion(test_set, model_paths,eval_id,amp=amp, batch_size=batch_size, metric= 'iou',fusion_method='vote', dataset_type=dataset_type)
        diaces_score_fusion_iou = evaluate_ensemble_fusion(test_set, model_paths,eval_id,amp=amp, batch_size=batch_size, metric= 'iou',fusion_method='fusion', dataset_type=dataset_type)
        

        diaces_score_vote_iou = diaces_score_vote_iou.item()
        diaces_score_fusion_iou = diaces_score_fusion_iou.item()
       

        logger.info("Diversity-Accuracy Measure for Homogenous Ensemble Selection (DIACES)")
        logger.info(f"diaces_score_vote_iou: {diaces_score_vote_iou}")
        logger.info(f"diaces_score_fusion_iou: {diaces_score_fusion_iou})")

        fusion_scores.append(diaces_score_fusion_iou)
        vote_scores.append(diaces_score_vote_iou)

    eval_dict = {}
    for i, alpha in enumerate(alphas):
        eval_dict[f'DIACES - {alpha}'] = [diaces_ids[i], len(diaces_ids[i]), vote_scores[i], fusion_scores[i], alpha]
    
    eval_df = pd.DataFrame.from_dict(eval_dict, orient= 'index', columns = ['Chosen Models','Ensemble Size', 'IoU - VOTING', 'IoU POSTERIOR AVERAGE','Alpha'])


    return eval_df


def exploratory_spectral_gm_lambda(logger, experiment_path,train_set, val_set, test_set, sub_sample_size,batch_size, amp,dataset_type = 'landcover'):
    """
    Performs an exploratory experiment for the Spectral Clustering-based ensemble pruning method (SCP) using different lambda values/the geometric mean.

    Args:
        logger (logging.Logger): Logger instance for logging progress and results of the analysis.
        experiment_path (str or Path): Path to the experiment directory containing the models.
        train_set (torch.utils.data.Dataset): Training dataset used for pruning.
        val_set (torch.utils.data.Dataset): Validation dataset used for generating the pruning set.
        test_set (torch.utils.data.Dataset): Test dataset used for evaluating the pruned ensemble.
        sub_sample_size (int): Number of samples to randomly select from the validation set for pruning.
        batch_size (int): Batch size for processing the datasets during pruning and evaluation.
        amp (bool): Whether to use automatic mixed precision during evaluation.
        dataset_type (str): Type of dataset ('landcover' or 'floods'). Default is 'landcover'.

    Returns:
        pd.DataFrame: DataFrame summarizing the evaluation results for each lambda value/geometric mean
    """
    
    experiment_path = Path(experiment_path)
    model_paths = get_model_paths(experiment_path)
    num_models = len(model_paths)
    
    
    #sub samples a Pruning Set from the VAL Set
    rng = np.random.default_rng()
    sample_index = rng.choice(np.arange(len(val_set)),size=sub_sample_size,replace=False)
    pruning_set = torch.utils.data.Subset(val_set, sample_index)

    logger.info(f"Images at these inidces chosen randomly for pruning: {sample_index}")
    
    logger.info("Generating the Ensemble Probability Table .....")

    #generate the ensemble proba table which is used for most pruning algos , which has shape (Classifiers, Sample_Number, Classes) or (M,N,C)
    ensemble_proba, target = generate_ensemble_proba_target(model_paths, pruning_set,batch_size= batch_size, amp=amp, dataset_type=dataset_type)
    ensemble_proba = ensemble_proba.cpu().numpy()
    target = target.numpy()

    logger.info("DONE")
    
    
    logger.info("Performing Spectral clustering based ensemble pruning approach (SCP) .... ")
       
    
    ####################
    ##  Spectral clustering based ensemble pruning approach (SCP)
    #################### 
    


    lambdas = [-1,0.2,0.4,0.8]
       
    scp_ids = []
    for lamb in lambdas:
       
        if lamb == -1:
            scp_cl = SpectralClusterPruningClassifier()
            scp_id = scp_cl.prune_(experiment_path,train_set,batch_size=batch_size,amp=amp,dataset_type=dataset_type,similarity_type = 'gm',lambd= None)
        else:
            scp_cl = SpectralClusterPruningClassifier()
            scp_id = scp_cl.prune_(experiment_path,train_set,batch_size=batch_size,amp=amp,dataset_type=dataset_type,similarity_type = 'lambda',lambd= lamb)

        logger.info("DONE")
        logger.info(f"scp_ids: {scp_id}")

        
        scp_ids.append(scp_id)


    fusion_scores = []
    vote_scores = []
    for eval_id in scp_ids:
        scp_score_vote_iou = evaluate_ensemble_fusion(test_set, model_paths,eval_id,amp=amp, batch_size=batch_size, metric= 'iou',fusion_method='vote', dataset_type=dataset_type)
        scp_score_fusion_iou = evaluate_ensemble_fusion(test_set, model_paths,eval_id,amp=amp, batch_size=batch_size, metric= 'iou',fusion_method='fusion', dataset_type=dataset_type)
        

        scp_score_vote_iou = scp_score_vote_iou.item()
        scp_score_fusion_iou = scp_score_fusion_iou.item()
        

        logger.info("Diversity-Accuracy Measure for Homogenous Ensemble Selection (DIACES)")
        logger.info(f"scp_score_vote_iou: {scp_score_vote_iou}")
        logger.info(f"scp_score_fusion_iou: {scp_score_fusion_iou})")

        fusion_scores.append(scp_score_fusion_iou)
        vote_scores.append(scp_score_vote_iou)

    eval_dict = {}
    for i, lamb in enumerate(lambdas):
        if lamb == -1:
            eval_dict[f'SCP - gm'] = [scp_ids[i], len(scp_ids[i]), vote_scores[i], fusion_scores[i], lamb]
        else:
            eval_dict[f'SCP - {lamb}'] = [scp_ids[i], len(scp_ids[i]), vote_scores[i], fusion_scores[i], lamb]

    eval_df = pd.DataFrame.from_dict(eval_dict, orient= 'index', columns = ['Chosen Models','Ensemble Size', 'IoU - VOTING', 'IoU POSTERIOR AVERAGE','Alpha'])


    return eval_df


def prune_single_ensemble_scp(logger, 
                              experiment_path,
                              train_set, 
                              test_set, 
                              batch_size, 
                              amp,
                              dataset_type = 'landcover'):
    

    """
    Performs (only) Spectral Clustering-based ensemble pruning (SCP) on a single initial ensemble, evaluates the pruned ensemble,
    and returns a DataFrame summarizing the results, including performance metrics and execution time.

    Args:
        logger (logging.Logger): Logger instance for logging progress and results.
        experiment_path (str or Path): Path to the experiment directory containing the models.
        train_set (torch.utils.data.Dataset): Training dataset used for pruning.
        test_set (torch.utils.data.Dataset): Test dataset used for evaluating the pruned ensemble.
        batch_size (int): Batch size for processing the datasets during pruning and evaluation.
        amp (bool): Whether to use automatic mixed precision during evaluation.
        dataset_type (str: Type of dataset ('landcover' or 'floods'). Default is 'landcover'.

    Returns:
        pd.DataFrame: DataFrame summarizing the evaluation results for the pruned ensemble
    """



    experiment_path = Path(experiment_path)
    model_paths = get_model_paths(experiment_path)

    
    



    ####################
    ##  "OTHER"
    #################### 
    logger.info("Performing Spectral clustering based ensemble pruning approach (SCP) .... ")
    start_scp = time.time()

    scp_cl = SpectralClusterPruningClassifier()
    scp_ids = scp_cl.prune_(experiment_path,train_set,batch_size=batch_size,amp=amp,dataset_type=dataset_type,similarity_type = 'gm',lambd= None)

    end_scp = time.time()
    logger.info("DONE")
    logger.info(f"scp_ids: {scp_ids}")
    
    

    ####################
    ##  EVALUATION
    ####################

    logger.info("BEGINNING EVALUATION")



    #performance of Spectral Clustering Pruning 
    scp_scores = evaluate_ensemble_fusion(test_set, model_paths,scp_ids,amp=amp, batch_size=batch_size, metric= 'all_eval',fusion_method='both', dataset_type=dataset_type)
    scp_score_vote_iou =              scp_scores[0] 
    scp_score_fusion_iou =            scp_scores[3] 
    scp_score_vote_f1_micro =         scp_scores[2] 
    scp_score_fusion_f1_micro =       scp_scores[5] 
    scp_score_vote_f1_macro =         scp_scores[1] 
    scp_score_fusion_f1_macro =       scp_scores[4] 
    time_scp = end_scp - start_scp
    len_scp_ids = len(scp_ids)
    logger.info("Spectral clustering based ensemble pruning approach (SCP) .... ")
    logger.info(f"scp_score_vote_iou: {scp_score_vote_iou}")
    logger.info(f"scp_score_fusion_iou: {scp_score_fusion_iou}")
    


    logger.info("EVALUATION DONE")
    logger.info("Track Results in eval df")

  



    # COLUMNS:                                              Chosen IDS  -                                   Ensemble Size                     Score Vote IOU -                     Score Fusion IOU                           Score Vote F1 MACRO                            Score Fusion F1 MACRO                   Score Vote F1 MICRO -                             Score Fusion F1 MICRO                            TIME
    eval_dict = {                                                                                       
        'SCP':                                              [scp_ids,                                      len_scp_ids,                       scp_score_vote_iou,                 scp_score_fusion_iou,                       scp_score_vote_f1_macro,                        scp_score_fusion_f1_macro,              scp_score_vote_f1_micro,                        scp_score_fusion_f1_micro,                        time_scp                        ],
    }
    

    eval_df = pd.DataFrame.from_dict(eval_dict, orient= 'index', columns = ['Chosen Models','Ensemble Size','IoU - VOTING', 'IoU POSTERIOR AVERAGE', 'F1 MACRO - VOTING', 'F1 MACRO POSTERIOR AVERAGE','F1 MICRO - VOTING', 'F1 MICRO POSTERIOR AVERAGE','Execution Time'])


    return eval_df
