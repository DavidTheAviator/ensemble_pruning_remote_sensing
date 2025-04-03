
# Less is More - An Empirical Study of Ensemble Pruning Methods in Remote Sensing

This is the code for the Bachelor Thesis **Less is More - An Empirical Study of Ensemble Pruning Methods in Remote Sensing** by David Tzafrir.  
It includes code to:

- Train ensembles of CNNs on the [LandCover.aI](https://landcover.ai.linuxpolska.com/) and [S1GFloods](https://github.com/Tamer-Saleh/S1GFlood-Detection) datasets.
- Prune an initial ensemble to a subensemble of a fixed size, comparing 10 different pruning methods with each other (based on a subsample of the validation set or computation in batches). The implemented pruning algorithms are:

    * Orientation Ordering (OO) (see [Pruning in ordered bagging ensembles](https://doi.org/10.1145/1143844.1143921) [Martínez-Muñoz and Suárez, 2006])
    * Ensemble Pruning via Individual Contribution Ordering (EPIC) (see [Ensemble pruning via individual contribution ordering](https://doi.org/10.1145/1835804.1835914) [Lu et al., 2010])
    * Margin & Diversity Ensemble Pruning (MDEP) (see [Margin & diversity based ordering ensemble pruning](https://doi.org/10.1016/j.neucom.2017.06.052) [Guo et al., 2018])
    * Margin Distance Minimization (MDM) (see [An Analysis of Ensemble Pruning Techniques Based on Ordered Aggregation](https://doi.org/10.1109/TPAMI.2008.78) [Martínez-Muñoz et al., 2009])
    * Uncertainty Weighted Accuracy (UWA) (see [An ensemble uncertainty aware measure for directed hill climbing ensemble pruning](https://doi.org/10.1007/s10994-010-5172-0) [Partalas et al.,2010])
    * Diversity Accuracy Ensemble Selection (DIACES) (see [A Diversity-Accuracy Measure for Homogenous Ensemble Selection](https://doi.org/10.9781/ijimai.2018.06.005) [Zouggar and Adla, 2019])
    * Simultaneous Diversity and Accuracy (SDACC) (see [Considering diversity and accuracy simultaneously for ensemble pruning](https://doi.org/10.1016/j.asoc.2017.04.058) [Dai et al., 2017])
    * Diversity Focused Two (DFTWO) (see [Considering diversity and accuracy simultaneously for ensemble pruning](https://doi.org/10.1016/j.asoc.2017.04.058) [Dai et al., 2017])
    * Accuracy Reinforcement (ACCREIN) (see [Considering diversity and accuracy simultaneously for ensemble pruning](https://doi.org/10.1016/j.asoc.2017.04.058) [Dai et al., 2017])

---

## Installation / Reproducing Results

The code was executed on a machine with Python 3.12.3, all dependencies are listed in the requirements.txt file. Follow these steps for installation and to reproduce the results:

1. Clone the repository:
   ```bash
   git clone https://github.com/DavidTheAviator/ensemble_pruning_remote_sensing.git
   cd ensemble_pruning_remote_sensing
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the datasets (S1GFloods [\[Download Link\]](https://drive.google.com/file/d/1W-gnUU-AaYbJ8KMdfnbrI7ySHkiKjOvo/view) and LandCover.aI [\[Download Link\]](https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip)) and save them in the /Datasets/ directory. It should have the following structure after completing this step (rename the datasets manually):

    ```
    /Datasets/
    ├── data_floods/
    │   ├── test/
    │   ├── train/
    │   └── val/
    └── data_landcover/
        ├── images/
        ├── masks/
        ├── output/
        ├── split.py
        ├── test.txt
        ├── train.txt
        └── val.txt
    ```

5. Pass the indexing file (for S1GFloods) provided in this repo to the data_floods/test, train and val folders:

    ```bash
    cd Datasets
    mv ./test.txt ./data_floods/test/
    mv ./train.txt ./data_floods/train/
    mv ./val.txt ./data_floods/val/
    ```

6. Ensembles can be trained via the train_ensemble_floods.py & train_ensemble_landcover.py Scripts.
7. Execute the pruning experiment_size.py script to perform the actual pruning experiment (with your specified initial ensembles/size ranges).

---

## Code Structure

```
/Datasets/                              # Datasets (need to be downloaded from source and named as specified above in the Installation Section)
/Ensemble/                              # All files needed for the creation of the initial ensemble
/experiments/                           # All experiment results will be saved here
/experiments/created_ensemble           # Folder to save the experiments for initial ensembles
/experiments/pruning_exploratory        # Results from the pruning_experiment_exploratory.py Script will be saved here
/experiments/pruning_ensemble_size      # Results from the pruning_experiment_size.py Script will be saved here
/experiments/temp                       # Used for saving probabilities and target arrays for computation in batches
/requirements.txt                       # Python package dependencies
README.md                               # This file
```

---

## Acknowledgments

Code, which has been taken from other sources, has been marked. The code for many of the pruning algorithms (and the module structure) is based on the [PyPruning](https://github.com/sbuschjaeger/PyPruning) library by Sebastian Buschjäger. The Unet implementation used is taken from the [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) GitHub Repository by Alexandre Milesi. The FC-siam-conc implementation used was made by one of the authors of the paper himself (Rodrigo Caye Daudt) and can be found in the GitHub Repository [fully_convolutional_change_detection](https://github.com/rcdaudt/fully_convolutional_change_detection).

Docstrings for functions were created with ChatGPT and then revised and corrected.
