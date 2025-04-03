import os
import logging
import sys
import pandas as pd
from pathlib import Path

# Add Parent Directory to Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Ensemble.utils import timestamp, StreamToLogger
from Ensemble.local_datasets import train_set_floods_local, val_set_floods_local, test_set_floods_local 
from Ensemble.local_datasets import train_set_landcover_local, val_set_landcover_local, test_set_landcover_local 

from Scripts.pruning_experiment_helper import prune_single_ensemble_experiment, rounded_df_copy



file_parent_path = os.path.dirname(__file__)
experiment_parent_path = Path(os.path.join(file_parent_path,"..","experiments","created_ensembles"))

#ADAPT INITIAL ENSEMBLE PATHS HERE
landcover_experiments = ['experiment_landcover_2025-01-28_01-25-06', 'experiment_landcover_2025-01-31_02-36-41', 'experiment_landcover_2025-02-04_11-47-30']
flood_experiments = ['experiment_floods_2025-01-29_13-01-26', 'experiment_floods_2025-01-29_15-51-00', 'experiment_floods_2025-01-29_18-21-56']

landcover_paths = []
flood_paths = []
for lc, fl in zip(landcover_experiments,flood_experiments):
    landcover_paths.append(experiment_parent_path.joinpath(lc))
    flood_paths.append(experiment_parent_path.joinpath(fl))


#create the sub folder for the  ensemble size pruning experiment
pruning_exp_folder_path = Path(os.path.join(file_parent_path,"..","experiments","pruning_ensemble_size",f"floods_lancover_ensemble_size_{timestamp()}"))

if not os.path.exists(pruning_exp_folder_path):
    os.makedirs(pruning_exp_folder_path)





#####CHANGE PARAMTERS HERE
batch_size_floods = 16
batch_size_landcover= 8 
sub_sample_size_floods = 128
sub_sample_size_landcover= 32
amp =True
n_jobs = -1




####LOGGERS
streams_logger = logging.getLogger('streams')
exp_logger = logging.getLogger('exp')
exp_logger.setLevel(logging.INFO)
streams_logger.setLevel(logging.INFO)
exp_fh = logging.FileHandler(pruning_exp_folder_path.joinpath('exp.log'))
exp_fh.setLevel(logging.INFO)
streams_fh = logging.FileHandler(pruning_exp_folder_path.joinpath('streams.log'))
streams_fh.setLevel(logging.INFO)


formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
exp_fh.setFormatter(formatter)

streams_logger.addHandler(streams_fh)
exp_logger.addHandler(exp_fh)
exp_logger.addHandler(logging.StreamHandler(sys.stdout))

#log stdout and stderr
sys.stdout = StreamToLogger(streams_logger,logging.INFO)
sys.stderr = StreamToLogger(streams_logger,logging.ERROR) #to also show this on stdout


####################
##  EXPERIMENT (EFFECTS OF ENSEMBLE SIZE)
####################


landcover_dfs = []
floods_dfs = []


#create a foler to save all the results already during creation, in case an error occurs during experiment execution
backup_results_path = pruning_exp_folder_path.joinpath("backup_results")
if not os.path.exists(backup_results_path):
    os.makedirs(backup_results_path)

#column names for rounding 
col_names_round = ['IoU - VOTING', 'IoU POSTERIOR AVERAGE', 'F1 MACRO - VOTING', 'F1 MACRO POSTERIOR AVERAGE','F1 MICRO - VOTING', 'F1 MICRO POSTERIOR AVERAGE','Execution Time']


experiment_path_list = [0,1,2]

for path_id in experiment_path_list:
    for n_estimators in range(1,10):
        if n_estimators == 1:
            #calculate_spectral = True 
            calculate_spectral = False
            calculate_benchmarks = True
        else:
            calculate_spectral = False
            calculate_benchmarks = False

        exp_logger.info(f"For initial ensemble {path_id} Creating Sub Ensemble of Size {n_estimators}")
        exp_logger.info(f"Landcover: {landcover_paths[path_id]}")
        exp_logger.info(f"Floods: {flood_paths[path_id]}")

    
        
        ######LANDCOVER
        landcover_df = prune_single_ensemble_experiment(logger= exp_logger, 
                                                        experiment_path=landcover_paths[path_id] ,
                                                        train_set= train_set_landcover_local, 
                                                        val_set = val_set_landcover_local, 
                                                        test_set = test_set_landcover_local, 
                                                        sub_sample_size = sub_sample_size_landcover ,
                                                        n_estimators = n_estimators,
                                                        calculate_spectral= calculate_spectral, 
                                                        calculate_optimal=False,
                                                        calculate_benchmarks = calculate_benchmarks,
                                                        batch_size= batch_size_landcover , 
                                                        amp=amp ,
                                                        n_jobs = n_jobs, 
                                                        dataset_type = 'landcover')

        #### FLOODS
        floods_df = prune_single_ensemble_experiment(logger= exp_logger, 
                                                    experiment_path=flood_paths[path_id] ,
                                                    train_set= train_set_floods_local, 
                                                    val_set = val_set_floods_local, 
                                                    test_set = test_set_floods_local, 
                                                    sub_sample_size = sub_sample_size_floods ,
                                                    n_estimators = n_estimators,
                                                    calculate_spectral= calculate_spectral, 
                                                    calculate_optimal=False,
                                                    calculate_benchmarks = calculate_benchmarks,
                                                    batch_size= batch_size_floods , 
                                                    amp=amp ,
                                                    n_jobs = n_jobs, 
                                                    dataset_type = 'floods')

        
        #append dfs to list 
        landcover_dfs.append(landcover_df)
        floods_dfs.append(floods_df)

        #backup the results
        lc_df_rounded = rounded_df_copy(landcover_df,col_names_round)
        fl_df_rounded = rounded_df_copy(floods_df,col_names_round)
        lc_df_rounded.to_excel(backup_results_path.joinpath(f"landcover_ensemble_{path_id}_size_{n_estimators}.xlsx"))
        fl_df_rounded.to_excel(backup_results_path.joinpath(f"floods_ensemble_{path_id}_size_{n_estimators}.xlsx"))






        

    ######SAVE THE EXPERIMENT RESULTS 
    floods_results_path = pruning_exp_folder_path.joinpath("floods_results")
    landcover_results_path = pruning_exp_folder_path.joinpath("landcover_results")

    if not os.path.exists(floods_results_path):
        os.makedirs(floods_results_path)
    if not os.path.exists(landcover_results_path):
        os.makedirs(landcover_results_path)





    #save the single dfs in files
    for i, (lc_df, fl_df) in enumerate(zip(landcover_dfs,floods_dfs)):
        lc_df_rounded = rounded_df_copy(lc_df,col_names_round)
        fl_df_rounded = rounded_df_copy(fl_df,col_names_round)
        lc_df_rounded.to_excel(landcover_results_path.joinpath(f"landcover_ensemble_{path_id}_size_{i+1}.xlsx"))
        fl_df_rounded.to_excel(floods_results_path.joinpath(f"floods_ensemble_{path_id}_size_{i+1}.xlsx"))

    #save the dfs below each other in two files
    startrow_lc = 0
    with pd.ExcelWriter(landcover_results_path.joinpath(f"landcover_ensemble_{path_id}_all_sizes.xlsx")) as writer:
        for df in landcover_dfs:
            df_rounded = rounded_df_copy(df,col_names_round)
            df_rounded.to_excel(writer, engine="xlsxwriter", startrow=startrow_lc)
            startrow_lc += (df.shape[0] + 2)

    startrow_fl = 0
    with pd.ExcelWriter(floods_results_path.joinpath(f"floods_ensemble_{path_id}_all_sizes.xlsx")) as writer:
        for df in floods_dfs:
            df_rounded = rounded_df_copy(df,col_names_round)
            df_rounded.to_excel(writer, engine="xlsxwriter", startrow=startrow_fl)
            startrow_fl += (df.shape[0] + 2)
    
    landcover_df = []
    floods_dfs = []

        
    