import os
import logging
import sys
from pathlib import Path

# Add Parent Directory to Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Ensemble.utils import timestamp, StreamToLogger
from Ensemble.local_datasets import train_set_floods_local, val_set_floods_local, test_set_floods_local 
from Ensemble.local_datasets import train_set_landcover_local, val_set_landcover_local, test_set_landcover_local 
from Scripts.pruning_experiment_helper import  exploratory_mdep_alpha, exploratory_diaces_alpha, exploratory_spectral_gm_lambda


file_parent_path = os.path.dirname(__file__)
experiment_parent_path = Path(os.path.join(file_parent_path,"..","experiments","created_ensembles"))


landcover_experiments = ['experiment_landcover_2025-01-28_01-25-06', 'experiment_landcover_2025-01-31_02-36-41', 'experiment_landcover_2025-02-04_11-47-30']
flood_experiments = ['experiment_floods_2025-01-29_13-01-26', 'experiment_floods_2025-01-29_15-51-00', 'experiment_floods_2025-01-29_18-21-56']

landcover_paths = []
flood_paths = []
for lc, fl in zip(landcover_experiments,flood_experiments):
    landcover_paths.append(experiment_parent_path.joinpath(lc))
    flood_paths.append(experiment_parent_path.joinpath(fl))


#create the sub folder for the  exploratory pruning experiment
pruning_exp_folder_path = Path(os.path.join(file_parent_path,"..","experiments","pruning_exploratory",f"floods_lancover_exploratory_{timestamp()}"))

if not os.path.exists(pruning_exp_folder_path):
    os.makedirs(pruning_exp_folder_path)





#####CHANGE PARAMTERS HERE
batch_size_floods = 16
batch_size_landcover= 8 
sub_sample_size_floods = 256
sub_sample_size_landcover= 32
amp =True
n_jobs = 4




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
##  MDEP (Alpha )
####################
exp_logger.info("MDEP (Alpha ) started")

######LANDCOVER
mdep_alpha_lc_2 = exploratory_mdep_alpha(logger=exp_logger, 
                                         experiment_path=landcover_paths[0],
                                         val_set = val_set_landcover_local, 
                                         test_set = test_set_landcover_local, 
                                         sub_sample_size = sub_sample_size_landcover ,
                                         n_estimators = 2,
                                         batch_size= batch_size_landcover , 
                                         amp=amp ,
                                         n_jobs = n_jobs, 
                                         dataset_type = 'landcover')

mdep_alpha_lc_3 = exploratory_mdep_alpha(logger=exp_logger, 
                                         experiment_path=landcover_paths[0],
                                         val_set = val_set_landcover_local, 
                                         test_set = test_set_landcover_local, 
                                         sub_sample_size = sub_sample_size_landcover ,
                                         n_estimators = 3,
                                         batch_size= batch_size_landcover , 
                                         amp=amp ,
                                         n_jobs = n_jobs, 
                                         dataset_type = 'landcover')



######FLOODS
mdep_alpha_fl_2 = exploratory_mdep_alpha(logger=exp_logger, 
                                         experiment_path=flood_paths[0],
                                         val_set = val_set_floods_local, 
                                         test_set = test_set_floods_local, 
                                         sub_sample_size = sub_sample_size_floods ,
                                         n_estimators = 2,
                                         batch_size= batch_size_floods, 
                                         amp=amp ,
                                         n_jobs = n_jobs, 
                                         dataset_type = 'floods')

mdep_alpha_fl_3 = exploratory_mdep_alpha(logger=exp_logger, 
                                         experiment_path=flood_paths[0],
                                         val_set = val_set_floods_local, 
                                         test_set = test_set_floods_local, 
                                         sub_sample_size = sub_sample_size_floods ,
                                         n_estimators = 3,
                                         batch_size= batch_size_floods, 
                                         amp=amp ,
                                         n_jobs = n_jobs, 
                                         dataset_type = 'floods')



####################
##  DIACES (Alpha )
####################


######LANDCOVER
diaces_alpha_lc_2 = exploratory_diaces_alpha(logger=exp_logger, 
                                             experiment_path=landcover_paths[0],
                                             val_set = val_set_landcover_local, 
                                             test_set = test_set_landcover_local, 
                                             sub_sample_size = sub_sample_size_landcover ,
                                             n_estimators = 2,
                                             batch_size= batch_size_landcover , 
                                             amp=amp ,
                                             n_jobs = n_jobs, 
                                             dataset_type = 'landcover')

diaces_alpha_lc_3 = exploratory_diaces_alpha(logger=exp_logger, 
                                             experiment_path=landcover_paths[0],
                                             val_set = val_set_landcover_local, 
                                             test_set = test_set_landcover_local, 
                                             sub_sample_size = sub_sample_size_landcover ,
                                             n_estimators = 3,
                                             batch_size= batch_size_landcover , 
                                             amp=amp ,
                                             n_jobs = n_jobs, 
                                             dataset_type = 'landcover')



######FLOODS
diaces_alpha_fl_2 = exploratory_diaces_alpha(logger=exp_logger, 
                                             experiment_path=flood_paths[0],
                                             val_set = val_set_floods_local, 
                                             test_set = test_set_floods_local, 
                                             sub_sample_size = sub_sample_size_floods ,
                                             n_estimators = 2,
                                             batch_size= batch_size_floods, 
                                             amp=amp ,
                                             n_jobs = n_jobs, 
                                             dataset_type = 'floods')

diaces_alpha_fl_3 = exploratory_diaces_alpha(logger=exp_logger, 
                                             experiment_path=flood_paths[0],
                                             val_set = val_set_floods_local, 
                                             test_set = test_set_floods_local, 
                                             sub_sample_size = sub_sample_size_floods ,
                                             n_estimators = 3,
                                             batch_size= batch_size_floods, 
                                             amp=amp ,
                                             n_jobs = n_jobs, 
                                             dataset_type = 'floods')



####################
##  SPECTRAL (gm vs lambda)
####################

######LANDCOVER
#not calculated because run time is very high
""""
spectral_gm_lambda_landcover = exploratory_spectral_gm_lambda(logger=exp_logger, 
                                             experiment_path=landcover_paths[0],
                                             train_set =val_set_landcover_local,
                                             val_set = val_set_landcover_local, 
                                             test_set = test_set_landcover_local, 
                                             sub_sample_size = sub_sample_size_landcover ,
                                             batch_size= batch_size_landcover, 
                                             amp=amp ,
                                             dataset_type = 'landcover')
"""
                                        

######FLOODS
spectral_gm_lambda_floods = exploratory_spectral_gm_lambda(logger=exp_logger, 
                                             experiment_path=flood_paths[0],
                                             train_set = train_set_floods_local,
                                             val_set = val_set_floods_local, 
                                             test_set = test_set_floods_local, 
                                             sub_sample_size = sub_sample_size_floods ,
                                             batch_size= batch_size_floods, 
                                             amp=amp ,
                                             dataset_type = 'floods')
    

######SAVE THE EXPERIMENT RESULTS 
floods_results_path = pruning_exp_folder_path.joinpath("floods_results")
landcover_results_path = pruning_exp_folder_path.joinpath("landcover_results")

if not os.path.exists(floods_results_path):
    os.makedirs(floods_results_path)
if not os.path.exists(landcover_results_path):
    os.makedirs(landcover_results_path)



mdep_alpha_lc_2.to_excel(landcover_results_path.joinpath("mdep_alpha_lc_2.xlsx"))
mdep_alpha_lc_3.to_excel(landcover_results_path.joinpath("mdep_alpha_lc_3.xlsx"))
mdep_alpha_fl_2.to_excel(floods_results_path.joinpath("mdep_alpha_fl_2.xlsx"))
mdep_alpha_fl_3.to_excel(floods_results_path.joinpath("mdep_alpha_fl_3.xlsx"))
diaces_alpha_lc_2.to_excel(landcover_results_path.joinpath("diaces_alpha_lc_2.xlsx"))
diaces_alpha_lc_3.to_excel(landcover_results_path.joinpath("diaces_alpha_lc_3.xlsx"))
diaces_alpha_fl_2.to_excel(floods_results_path.joinpath("diaces_alpha_fl_2.xlsx"))
diaces_alpha_fl_3.to_excel(floods_results_path.joinpath("diaces_alpha_fl_3.xlsx"))
#spectral_gm_lambda_landcover.to_excel(landcover_results_path.joinpath("spectral_gm_lambda_landcover.xlsx"))
spectral_gm_lambda_floods.to_excel(floods_results_path.joinpath("spectral_gm_lambda_floods.xlsx"))