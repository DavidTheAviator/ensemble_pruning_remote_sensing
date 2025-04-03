import numpy as np
from joblib import Parallel,delayed
from tqdm import tqdm

from .PruningClassifier import PruningClassifier
from .helpers import ProbaTargetLoader



def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2' """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



#Source: https://github.com/sbuschjaeger/PyPruning/blob/master/PyPruning/RankPruningClassifier.py
#Adapted (Implemented vectorized computations insted of loop)
def individual_margin_diversity(i, ensemble_proba, target, alpha = 0.2, V=None):
    """
    Computes the individual margin diversity score as described in the paper "Margin & diversity based ordering ensemble pruning" (Guo et al., 2018).

    Args:
        i (int): Index of the classifier being evaluated.
        ensemble_proba (numpy.ndarray): Array of shape (M, N, C) containing class probability predictions from M classifiers 
                                        for N samples across C classes.
        target (numpy.ndarray): Array of true class labels for N samples.
        alpha (float, optional): Weighting factor for balancing margin and diversity scores. Defaults to 0.2.
        V (numpy.ndarray, optional): Array of shape (N, C) representing the votes for each class per sample. 
                                     If None, it will be computed internally. Defaults to None.

    Returns:
        numpy.ndarray: The individual margin diversity score for the classifier i
    """

    iproba = ensemble_proba[i,:,:]
    n = iproba.shape[0]
    predictions = iproba.argmax(axis=1)
    m = ensemble_proba.shape[0]
    
    
    
    if V is None:
        #V is eventually a matrix of shape (N,C) which denotes the votes for each class per sample
        #ideally calculated before once for better performance
        V = np.zeros(ensemble_proba.shape)
        idx = ensemble_proba.argmax(axis=2)
        V[np.arange(ensemble_proba.shape[0])[:,None],np.arange(ensemble_proba.shape[1]),idx] = 1
        V = V.sum(axis=0)

   

   
   
    #tracking where the classifier is predicting right wrong on the n samples
    right_preds_c = (predictions == target).astype(float)

    #votes for the correct class on each sample
    v_correct = V[np.arange(n),target]

    #max votes on each sample (excluding the correct class)
    V_correct_masked = np.ma.array(V,mask=False)
    V_correct_masked.mask[np.arange(n), target] = True
    v_pseudo_max = V_correct_masked.max(axis=1)

    #calculate the margin score 
    fm = v_correct - v_pseudo_max
    
    #in the case where the correct class and the class with the "max" votes have the same number of votes, we adapt  the value to make log calculation possible
    fm[fm== 0] = 0.01
    fm = np.log(np.abs(fm*1/m))


    #if the correct class receives 0 votes, this value will not be taken into account for the metric (since right_preds_c * MDM )
    #still in order to not raise Warnings, it is set to 0.01
    v_correct[v_correct==0] = 0.01

    #calculate diversity score
    fd = np.log(v_correct*1/m)


    MDM = alpha*fm + (1-alpha)*fd 
    MDM = right_preds_c * MDM 
    MDM = MDM.sum(axis=0)

    return MDM 


    

#Source: https://github.com/sbuschjaeger/PyPruning/blob/master/PyPruning/RankPruningClassifier.py
#Adapted (Implemented vectorized computations insted of loop)
def individual_contribution_ordering(i, ensemble_proba, target, V=None):
    """
    Computes the EPIC score as laid out in "Ensemble pruning via individual contribution ordering" (Lu et. al., 2010)

    Args:
        i (int): Index of the classifier being evaluated.
        ensemble_proba (numpy.ndarray): Array of shape (M, N, C) containing class probability predictions 
                                        from M classifiers for N samples across C classes.
        target (numpy.ndarray): Array of true class labels for N samples.
        V (numpy.ndarray, optional): Array of shape (N, C) representing the votes for each class per sample. 
                                     If None, it will be computed internally. Defaults to None.

    Returns:
        float: The negative individual contribution of the classifier i
    """
    
    iproba = ensemble_proba[i,:,:]
    n = iproba.shape[0]

    #prediction of the classifier
    pred = iproba.argmax(axis=1)

    if V is None:
        #V is eventually a matrix of shape (N,C) which denotes the votes for each class per sample
        #ideally calculated before once for better performance
        V = np.zeros(ensemble_proba.shape)
        idx = ensemble_proba.argmax(axis=2)
        V[np.arange(ensemble_proba.shape[0])[:,None],np.arange(ensemble_proba.shape[1]),idx] = 1
        V = V.sum(axis=0)


    #fuse results by majority voting
    maj = V.argmax(axis=1)
    


    #partitions along axis = -1, so axis=1 in this case, so that 
    #everything left of the second largest element is smaller or equal, anything right of it
    # larger or equal
    partition = np.partition(V, -2)
    
    #highest number of votes for a class (per sample)
    v_max = partition[:,-1]

    #second highest number of votes for a class (per sample)
    v_sec = partition[:,-2]

    #number of votes for the prediction of the classifier
    v_c = V[np.arange(n),pred]


    #number of votes for the correct class
    v_correct = V[np.arange(n),target]
    
    #classifier predicts correctly and in minority group
    alpha = np.logical_and(pred == target, maj != target)

    
    #classifier predicts correctly and in majority group
    beta = np.logical_and(pred == target, maj == target)

    #classifier predicts false
    theta = (pred != target)

    IC = alpha.astype(float) * (2*v_max- v_c) + beta.astype(float) * v_sec + theta.astype(float) * (v_correct-v_c - v_max)
    IC = IC.sum(axis= 0)

    return -1.0 * IC


    

#Source: https://github.com/sbuschjaeger/PyPruning/blob/master/PyPruning/RankPruningClassifier.py
#Adapted (Reference vector is now computed as laid out in the paper, distance measure adapted )
def orientation_ordering(i, ensemble_proba, target, c_ref = None):
    """
    Computes the Orientation Ordering metric as laid out in "Pruning in ordered bagging ensembles" (Martínez-Muñoz and Suárez, 2006)

    Args:
        i (int): Index of the classifier being evaluated.
        ensemble_proba (numpy.ndarray): Array of shape (M, N, C) containing class probability predictions 
                                        from M classifiers for N samples across C classes.
        target (numpy.ndarray): Array of true class labels for N samples.
        c_ref (numpy.ndarray, optional): Reference vector for orientation calculation. 
                                         If None, it is computed internally. Defaults to None.

    Returns:
        float: The computed orientation ordering score for the classifier.

    Raises:
        AssertionError: If the reference vector or the classifier signature vector contains only zeros.
    """



    #for performance reasons c_ref should be calculated beforehand (only once)
    if c_ref is None:
        c_sig = 2.0*(ensemble_proba.argmax(axis=2) == target[np.newaxis,:]) -1.0
        #ensemble signature vector
        c_ens = c_sig.mean(axis=0)
        o = np.ones(len(c_ens), dtype=float)


        lamb = np.dot(-o,c_ens)/np.dot(c_ens,c_ens)
        c_ref = o + lamb * c_ens

    
    c_i = 2.0* (ensemble_proba[i,:,:].argmax(axis=1) == target) -1.0


    assert np.all(c_ref==np.zeros(1)) == False and np.all(c_i==np.zeros(1)) == False, 'One of the vector contains just zeros'

    return angle_between(c_i,c_ref)





#Source https://github.com/sbuschjaeger/PyPruning/blob/master/PyPruning/RankPruningClassifier.py
# Adapted (Added the logic/functions for computation in batches)
class RankPruningClassifier(PruningClassifier):
    """
    Rank-based pruning method for ensemble classifiers.

    This approach ranks classifiers based on a given metric and selects the top `n_estimators` classifiers.
    The metric is always minimized, so actually the top `n_estimators`, are the ones with the lowest metric values.
   

    Args:
        n_estimators (int): Number of classifiers to select (default is 5).
        metric (function): Function to rank classifiers, with lower scores being better.
        n_jobs (int): Number of threads for parallel metric computation (default is 8).
        metric_options (dict or None): Additional options passed to the metric function (default is None).

    Attributes:
        n_estimators (int): The number of classifiers to be selected.
        metric (function): The metric function used for ranking classifiers.
        n_jobs (int): The number of parallel jobs for metric computation.
        metric_options (dict): Additional options for the metric function.

    Methods:
        batched_metric_rank(proba, target, proba_target_path, n_received):
            Computes and averages scores for classifiers over batches.
        
        prune_(proba, target, proba_target_path=None):
            Prunes the ensemble by selecting the top-ranked classifiers.
    """

    def __init__(self, n_estimators = 5, metric = None, n_jobs = 8, metric_options = None):
        """
        Initialize the RankPruningClassifier.

        Args:
            n_estimators (int, optional): The number of classifiers to select. Defaults to 5.
            metric (function, optional): The metric function used to rank classifiers. Must return lower scores for better classifiers (the metric is always minimized).
            n_jobs (int, optional): The number of threads to use for parallel metric computation. Defaults to 8.
            metric_options (dict, optional): Additional options to be passed to the metric function. Defaults to None.

        Returns:
            None
        """

        super().__init__()

        assert metric is not None, "You must provide a valid metric!"
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.metric = metric

        if metric_options is None:
            self.metric_options = {}
        else:
            self.metric_options = metric_options


    # it is assumed that Parallel keeps the order of evaluations regardless of its backend (see eg. https://stackoverflow.com/questions/56659294/does-joblib-parallel-keep-the-original-order-of-data-passed)
    # But for safety measures we also return the index of the current model
    def _metric(self, i, ensemble_proba, target, additional_options):
        """
        Computes the metric score for a given classifier (used for parallel computation).

        Args:
            i (int): The index of the classifier being evaluated.
            ensemble_proba (numpy.ndarray): The predicted probabilities for the ensemble.
            target (numpy.ndarray): The true target values for evaluation.
            additional_options (dict): Additional options to be passed to the metric function.

        Returns:
            tuple: A tuple containing the classifier index and its metric score.
        """

        return (i, self.metric(i, ensemble_proba, target, **additional_options))


    def batched_metric_rank(self, proba, target,proba_target_path, n_received):
        """
        Evaluates classifiers using the ranking metric, supporting both batched and non-batched computations.

        This function computes the score of each classifier based on the given metric and returns their 
        ranking scores. If batch processing is used, the scores are computed iteratively across batches 
        and then averaged.

        Args:
            proba (numpy.ndarray): The predicted probabilities for the ensemble (for non-batched computation).
            target (numpy.ndarray): The true target values for evaluation (for non-batched computation).
            proba_target_path (str or None): Path to the batch directory, which stores proba and target arrays 
                                            (if batch processing is used, check the ProbaTargetLoader for more details).
            n_received (int): The total number of classifiers in the ensemble.

        Returns:
            list: A list of computed metric scores for each classifier.
        """



        #calculate the batched version of the algorithm
        if proba_target_path is not None:
            batch_scores = []

            
            for proba, target in tqdm(ProbaTargetLoader(proba_target_path),total=len(ProbaTargetLoader(proba_target_path)), desc="Batched Metric Calculation",unit='batch', leave=False):
                scores = Parallel(n_jobs=self.n_jobs, backend="threading")(
                    delayed(self._metric) (i, proba, target, self.metric_options) for i in range(n_received)
                )

                

                #sort scores by i in ascending order (altough in theory joblib keeps ordering)
                scores.sort(key=lambda t: t[0])

            
                batch_scores.append(scores)
            
            averaged_scores = {}

            for i in range(n_received):
                averaged_scores[i] = []

            for score in batch_scores:
                for (i,metric) in score:
                    averaged_scores[i].append(metric)

            final_scores = []
            for i in averaged_scores:
                final_scores.append(np.mean(averaged_scores[i]))
            


            return final_scores

        else:
            scores = Parallel(n_jobs=self.n_jobs, backend="threading")(
                    delayed(self.metric) (i, proba, target, **self.metric_options) for i in range(n_received)
                )
            return scores





    
    def prune_(self, proba, target, proba_target_path=None):
        """
        Prunes the ensemble by selecting the top-ranked (i.e.,lowest metric values) classifiers based on a given metric. 
        If batch processing is used, the computation is performed iteratively (over the whole dataset)

        Args:
            proba (numpy.ndarray): The predicted probabilities for the ensemble (for non-batched computation).
            target (numpy.ndarray): The true target values for evaluation (for non-batched computation).
            proba_target_path (str or None, optional): Path to the batch directory, which stores proba and target arrays 
                                                    (if batch processing is used, check the ProbaTargetLoader for more details).

        Returns:
            tuple: A tuple containing the list of selected classifier indices and their corresponding weights.
        """
        if(proba_target_path is None):
            n_received = proba.shape[0]
        else:
            proba_target_loader = ProbaTargetLoader(proba_target_path)
            first_proba,_ = next(proba_target_loader)
            n_received = first_proba.shape[0]
       
        if self.n_estimators >= n_received:
            return range(0, n_received), [1.0 / n_received for _ in range(n_received)]
        
        single_scores = self.batched_metric_rank(proba, target, proba_target_path, n_received)
        single_scores = np.array(single_scores)

        return np.argpartition(single_scores, self.n_estimators)[:self.n_estimators], [1.0 / self.n_estimators for _ in range(self.n_estimators)]
        