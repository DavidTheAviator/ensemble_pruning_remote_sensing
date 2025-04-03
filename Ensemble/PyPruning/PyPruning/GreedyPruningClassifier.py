import numpy as np
from joblib import Parallel,delayed
from .PruningClassifier import PruningClassifier
from tqdm import tqdm
from .helpers import ProbaTargetLoader





#Source https://github.com/sbuschjaeger/PyPruning/blob/master/PyPruning/GreedyPruningClassifier.py
#Slightly Adapted (Taking the mean of the signature vectors, choice of p etc.)
def margin_distance_minimization(i, ensemble_proba, selected_models, target):
    """
    Computes the margin distance minimization criterion for selecting classifiers in an ensemble,
    as laid out in the paper "An Analysis of Ensemble Pruning Techniques Based on Ordered Aggregation" (Martínez-Muñoz et. al., 2009)

    Args:
        i (int): Index of the classifier being evaluated.
        ensemble_proba (numpy.ndarray): A (M, N, C) array with class probability predictions from M classifiers.
        selected_models (list): Indices of already selected classifiers.
        target (numpy.ndarray): Array of true class labels for N samples.

    Returns:
        float: The computed margin distance value
    """


    #the amount of classifiers 
    M = ensemble_proba.shape[0]
 

    iproba = ensemble_proba[i,:,:]
    sub_proba = ensemble_proba[selected_models, :, :]
    
    #signature vector of the selected model
    c_sig_selected = (2 * (iproba.argmax(axis=1) == target) - 1.0)
    c_sig_selected = c_sig_selected[np.newaxis,:]

    #no other classifiers selected up to this point 
    if(sub_proba.size == 0):
        #array of "all" signature vectors just consist of one
        c_sigs = c_sig_selected 
    #there are classifiers which have been selected already 
    else:
        #calculate the reference vectors of the selected classifiers
        c_sigs = 2.0*(sub_proba.argmax(axis=2) == target[np.newaxis,:]) - 1.0
        
        #add reference vector of the not selected classifier
        c_sigs = np.concatenate((c_sig_selected,c_sigs),axis=0)

        

    #normalize sum of signature vectors (works also if only one signature vector in the array)
    c_sigs = np.sum(c_sigs,axis=0) * (1/M)

    
    u = len(selected_models) + 1
    
    
    #other option (as suggested in the paper)
    #p =  2* math.sqrt(2*u) * 1/M
    p =  0.075
    
    o = np.full(len(target), p)

    return np.linalg.norm(o - c_sigs)


#Implementation by authors: http://mlkd.csd.auth.gr/ensemblepruning.html
#Adapted for Pytghon/NumPy
def uwa(i, ensemble_proba, selected_models, target):
    """
    Computes the Uncertainty Weighted Accuracy metric from the paper "An ensemble uncertainty aware measure for directed hill climbing ensemble pruning" (Partalas et. al.,2010)

    Args:
        i (int): Index of the classifier being evaluated.
        ensemble_proba (numpy.ndarray): Array of shape (M, N, C) containing class probability predictions
                                        from M classifiers for N samples across C classes.
        selected_models (list): Indices of the classifiers already selected in the ensemble.
        target (numpy.ndarray): Array of true class labels for N samples.

    Returns:
        float: The negative UWA score
    """
    
    
    
    S =  len(selected_models)
    N = len(target)
    iproba = ensemble_proba[i,:,:]
    
    # shape S x N X C
    sub_proba = ensemble_proba[selected_models, :, :]

    #in implementation by authors accuracy is used for the first iteration
    if(S== 0):
        return -np.count_nonzero(iproba.argmax(axis=1) == target) * 1/N
    else:
        ipred = (iproba.argmax(axis=1) == target)

        
        sub_pred = (sub_proba.sum(axis=0).argmax(axis=1) == target)


        #calculate ratio of correct models 
        sub_individual_pred = sub_proba.argmax(axis=2) == target[np.newaxis, :] 
        prop_correct = np.count_nonzero(sub_individual_pred, axis = 0) * 1/S
        prop_false = 1 - prop_correct
    
        #classifier and ensemble predict correctly
        e_tt = np.logical_and(ipred, sub_pred)
        #classifer correct, but ensemble false
        e_tf = np.logical_and(ipred, np.logical_not(sub_pred))
        #classifier false and ensemble false
        e_ff = np.logical_and(np.logical_not(ipred), np.logical_not(sub_pred))
        #classifier false and ensemble correct 
        e_ft = np.logical_and(np.logical_not(ipred), sub_pred)

        e_tf = e_tf.astype(float) * prop_correct
        e_ft = - e_ft.astype(float) *prop_false
        e_tt = e_tt.astype(float) * prop_false
        e_ff = - e_ff.astype(float) * prop_correct

        
        uwa = np.sum(np.stack([e_tf,e_ft,e_tt,e_ff]))

        return -uwa




def diaces(i, ensemble_proba, selected_models, target, alpha):
    """
    Computes the DIACES (DIversity and ACcuracy for Ensemble Selection) metric from the paper 
    "A Diversity-Accuracy Measure for Homogenous Ensemble Selection" (Zouggar and Adla, 2019)

    Args:
        i (int): Index of the classifier being evaluated.
        ensemble_proba (numpy.ndarray): Array of shape (M, N, C) containing class probability 
                                        predictions from M classifiers for N samples across C classes.
        selected_models (list): Indices of the classifiers already selected in the ensemble.
        target (numpy.ndarray): Array of true class labels for N samples.
        alpha (float): Weighting factor for balancing diversity and accuracy.

    Returns:
        float: The computed DIACES score
    """
   
    if(len(selected_models)== 0):
    
        #paper does not go into much detail how to choose the first classifier
        #it should be a random classifier from the group of classifiers with "medium performance" (measured by accuracy)
        M = ensemble_proba.shape[0]
        n = ensemble_proba.shape[1]
        
        assert M>=3, "The diaces method requires at least three classifiers in the initial ensemble"


        


        classifier_correct_preds = ((ensemble_proba.argmax(axis=2) == target[np.newaxis,:]) * 1.0).sum(axis=1)
        classifier_accuracy = classifier_correct_preds * 1/n
        ids_classifiers = np.argsort(classifier_accuracy)
        medium_group = np.array_split(ids_classifiers,3)[1]
        

        #assign a random negative number to a medium classifier, so that one of the medium classifiers will be choosen (since it has the lowest metric value)
        if(i in medium_group):
            rng = np.random.default_rng()
            return rng.choice(np.arange(-M,0))
        else:
            #if the classifier is not in the medium group it will surely not be chosen, since the metric is minimized
            return 1
        



    else:
        subset_ids = selected_models+ [i]
        subset_proba = ensemble_proba[subset_ids, :,:]


        #matrix of shape M,N which has entry 1.0 at i,j if classifier i predicted wrongly on sample j
        subset_errors = (subset_proba.argmax(axis=2) != target[np.newaxis,:]) *1.0 

        sample_errors = subset_errors.sum(axis=0)
        classifier_errors = subset_errors.sum(axis=1)


        n = ensemble_proba.shape[1]    
        M = ensemble_proba.shape[0]
        total_errors = sample_errors.sum()



        theta = sample_errors * 1/total_errors
        classifier_errors_ratio = classifier_errors * 1/n 


        c = (theta **2).sum()
        e = (classifier_errors_ratio ** 2).sum()

        c_star = (n*total_errors*c - total_errors)/(n*M -total_errors)
        e_star = (M*(n**2)*e- (total_errors**2))/(total_errors*M*n- total_errors)



        S = c_star + alpha * e_star 

        return S 


def sdacc(i, ensemble_proba, selected_models, target):
    """
    Computes the simultaneous diversity & accuracy (SDAcc) metric from the paper "Considering diversity and accuracy simultaneously for ensemble pruning" (Dai et. al., 2017)

    Args:
        i (int): Index of the classifier being evaluated.
        ensemble_proba (numpy.ndarray): Array of shape (M, N, C) containing class probability 
                                        predictions from M classifiers for N samples across C classes.
        selected_models (list): Indices of the classifiers already selected in the ensemble.
        target (numpy.ndarray): Array of true class labels for N samples.
        alpha (float): Weighting factor for balancing diversity and accuracy.

    Returns:
        float: The computed SDAcc score
    """
    
    S =  len(selected_models)
    N = len(target)

    #shape N X C
    iproba = ensemble_proba[i,:,:]
    
    
    # shape  S x N X C
    sub_proba = ensemble_proba[selected_models, :, :]

    #accuracy is used as metric in first iteration (we take negative accuracy since the metric is always minimized)
    if(S== 0):
        return -np.count_nonzero(iproba.argmax(axis=1) == target) * 1/N
    else:
        ipred = (iproba.argmax(axis=1) == target)

        #we use posterior averaging to determine the decision of the subensemble
        sub_pred = (sub_proba.sum(axis=0).argmax(axis=1) == target)


        #calculate ratio of correct models 
        sub_individual_pred = sub_proba.argmax(axis=2) == target[np.newaxis, :] 
        prop_correct = np.count_nonzero(sub_individual_pred, axis = 0) * 1/S
        prop_false = 1 - prop_correct
    
        #classifier and ensemble predict correctly
        e_tt = np.logical_and(ipred, sub_pred)
        #classifer correct, but ensemble false
        e_tf = np.logical_and(ipred, np.logical_not(sub_pred))
        #classifier false and ensemble false
        e_ff = np.logical_and(np.logical_not(ipred), np.logical_not(sub_pred))
        #classifier false and ensemble correct 
        e_ft = np.logical_and(np.logical_not(ipred), sub_pred)

        e_tf = e_tf.astype(float) * prop_false
        e_ft = - e_ft.astype(float) *prop_false
        e_tt = e_tt.astype(float) * prop_false
        e_ff = - e_ff.astype(float) * prop_correct

        
        sdacc = np.sum(np.stack([e_tf,e_ft,e_tt,e_ff]))

        return -sdacc




def dftwo(i, ensemble_proba, selected_models, target):
    """
    Computes the diversity-focused-two (DFTwo) metric from the paper "Considering diversity and accuracy simultaneously for ensemble pruning" (Dai et. al., 2017)

    Args:
        i (int): Index of the classifier being evaluated.
        ensemble_proba (numpy.ndarray): Array of shape (M, N, C) containing class probability 
                                        predictions from M classifiers for N samples across C classes.
        selected_models (list): Indices of the classifiers already selected in the ensemble.
        target (numpy.ndarray): Array of true class labels for N samples.
        alpha (float): Weighting factor for balancing diversity and accuracy.

    Returns:
        float: The computed DFTwo score
    """


    S =  len(selected_models)
    N = len(target)

    #shape N X C
    iproba = ensemble_proba[i,:,:]
    
    
    # shape  S x N X C
    sub_proba = ensemble_proba[selected_models, :, :]

    #accuracy is used as metric in first iteration (we take negative accuracy since the metric is always minimized)
    if(S== 0):
        return -np.count_nonzero(iproba.argmax(axis=1) == target) * 1/N
    else:
        ipred = (iproba.argmax(axis=1) == target)

        #we use posterior averaging to determine the decision of the subensemble
        sub_pred = (sub_proba.sum(axis=0).argmax(axis=1) == target)


        #calculate ratio of correct models 
        sub_individual_pred = sub_proba.argmax(axis=2) == target[np.newaxis, :] 
        prop_correct = np.count_nonzero(sub_individual_pred, axis = 0) * 1/S
        prop_false = 1 - prop_correct
    
       
        #classifer correct, but ensemble false
        e_tf = np.logical_and(ipred, np.logical_not(sub_pred))
        #classifier false and ensemble correct 
        e_ft = np.logical_and(np.logical_not(ipred), sub_pred)

        e_tf = e_tf.astype(float) * prop_false
        e_ft = - e_ft.astype(float) *prop_false
        

        
        dftwo = np.sum(np.stack([e_tf,e_ft]))

        return -dftwo


def accrein(i, ensemble_proba, selected_models, target):
    """
    Computes the accuracy-reinforcement (AccRein) metric from the paper "Considering diversity and accuracy simultaneously for ensemble pruning" (Dai et. al., 2017)

    Args:
        i (int): Index of the classifier being evaluated.
        ensemble_proba (numpy.ndarray): Array of shape (M, N, C) containing class probability 
                                        predictions from M classifiers for N samples across C classes.
        selected_models (list): Indices of the classifiers already selected in the ensemble.
        target (numpy.ndarray): Array of true class labels for N samples.
        alpha (float): Weighting factor for balancing diversity and accuracy.

    Returns:
        float: The computed AccRein score
    """
        
    S =  len(selected_models)
    N = len(target)

    #shape N X C
    iproba = ensemble_proba[i,:,:]
    
    
    # shape  S x N X C
    sub_proba = ensemble_proba[selected_models, :, :]

    #accuracy is used as metric in first iteration (we take negative accuracy since the metric is always minimized)
    if(S== 0):
        return -np.count_nonzero(iproba.argmax(axis=1) == target) * 1/N
    else:
        ipred = (iproba.argmax(axis=1) == target)

        #we use posterior averaging to determine the decision of the subensemble
        sub_pred = (sub_proba.sum(axis=0).argmax(axis=1) == target)


        #calculate ratio of correct models 
        sub_individual_pred = sub_proba.argmax(axis=2) == target[np.newaxis, :] 
        prop_correct = np.count_nonzero(sub_individual_pred, axis = 0) * 1/S
        prop_false = 1 - prop_correct
    
        #classifier and ensemble predict correctly
        e_tt = np.logical_and(ipred, sub_pred)
        #classifer correct, but ensemble false
        e_tf = np.logical_and(ipred, np.logical_not(sub_pred))
        #classifier false and ensemble correct 
        e_ft = np.logical_and(np.logical_not(ipred), sub_pred)

        e_tf = e_tf.astype(float) * prop_false
        e_ft = - e_ft.astype(float) *prop_false
        e_tt = e_tt.astype(float) * prop_false

        
        accrein = np.sum(np.stack([e_tf,e_ft,e_tt]))

        return -accrein





#Source https://github.com/sbuschjaeger/PyPruning/blob/master/PyPruning/GreedyPruningClassifier.py
# Adapted (Added the logic/functions for computation in batches)
class GreedyPruningClassifier(PruningClassifier):
    """
    Greedy-based pruning method for ensemble classifiers.

    This approach orders classifiers based on their performance, considering both 
    individual performance (according to a certain metric) and the already selected sub-ensemble. In each round, 
    it greedily selects the classifier that minimizes a predefined loss function.

    As just stated, the metric is always minimized in this implementation.

    Args:
        n_estimators (int): Number of classifiers to select (default is 5).
        metric (function): Function to score classifiers, with lower scores being better 
                            (default is margin_distance_minimization).
        n_jobs (int): Number of threads for parallel metric computation (default is 8).
        metric_options (dict or None): Additional options passed to the metric function (default is None).

    Attributes:
        n_estimators (int): The number of estimators to be selected.
        metric (function): The metric function used for classifier selection.
        n_jobs (int): The number of parallel jobs for metric computation.
        metric_options (dict): Additional options for the metric function.

    Methods:
        batched_metric_greedy(proba, target, proba_target_path, selected_models, not_selected_models):
            Computes and averages scores for classifiers over batches.
        
        prune_(proba, target, proba_target_path=None):
            Prunes the ensemble by selecting the best classifiers using the greedy method.
    """


    def __init__(self, n_estimators = 5, metric = margin_distance_minimization, n_jobs = 8, metric_options = None):
        """
        Initialize the GreedyPruningClassifier.

        Args:
            n_estimators (int, optional): The number of classifiers to select. Defaults to 5.
            metric (function, optional): The metric function to score classifiers. Defaults to margin_distance_minimization.
            n_jobs (int, optional): The number of threads to use for parallel metric computation. Defaults to 8.
            metric_options (dict, optional): Additional options for the metric function. Defaults to None.

        Returns:
            None
        """

        super().__init__()

        assert metric is not None, "You did not provide a valid metric for model selection. Please do so"
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.metric = metric

        if metric_options is None:
            self.metric_options = {}
        else:
            self.metric_options = metric_options



    def _metric(self, i, ensemble_proba, selected_models, target, additional_options):
        """
        Compute the metric score for a given classifier (used for parallel computation)

        Args:
            i (int): The index of the classifier being evaluated.
            ensemble_proba (array-like): The predicted probabilities for the ensemble.
            selected_models (list): List of already selected classifiers for the ensemble.
            target (array-like): The true target values for evaluation.
            additional_options (dict): Additional options to be passed to the metric function.

        Returns:
            tuple: A tuple containing the classifier index and its metric score.
        """
        return (i, self.metric(i, ensemble_proba, selected_models, target, **additional_options))





    def batched_metric_greedy(self, proba, target, proba_target_path, selected_models,not_selected_models):
        """
        Evaluate classifiers using the greedy metric selection, supporting both batched and non-batched computations.
        This function represents one iteration of the whole greedy algorithm (where a certain number of models have been selected already and the others are up for selection)

      

        Args:
            proba (numpy.ndarray): The predicted probabilities for the ensemble (for non-batched computation)
            target (numpy.ndarray): The true target values for evaluation (for non-batched computation)
            proba_target_path (str or None): Path to the batch directory, which stores proba and target arrays (if batch processing is used, check the ProbaTargetLoader for more details)
            selected_models (list): List of already selected classifiers for the ensemble.
            not_selected_models (list): List of classifiers not yet selected for evaluation.

        Returns:
            list: A list of tuples containing classifier indices and their average scores (if batched) or individual scores (if non-batched).
        """
        
        #the batched version of the algorithm is executed
        if proba_target_path is not None:
            batch_scores = []

            
            #iterate through the whole batch, accumulating scores
            for proba, target in tqdm(ProbaTargetLoader(proba_target_path),total=len(ProbaTargetLoader(proba_target_path)), desc="Batched Metric Calculation",unit='batch', leave=False):
                scores = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self._metric) ( i, proba, selected_models, target, self.metric_options) for i in not_selected_models
                )

                #sort scores by i in ascending order (altough in theory joblib keeps ordering)
                scores.sort(key=lambda t: t[0])

            
                batch_scores.append(scores)
            

            #average the accumulated scores
            averaged_scores = {}

            for i in not_selected_models:
                averaged_scores[i] = []

            for score in batch_scores:
                for (i,metric) in score:
                    averaged_scores[i].append(metric)

            final_scores = []
            for i in averaged_scores:
                final_scores.append((i,np.mean(averaged_scores[i])))

            return final_scores

        else:
            scores = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self._metric) ( i, proba, selected_models, target, self.metric_options) for i in not_selected_models
            )
            return scores



    def prune_(self, proba, target, proba_target_path=None):
        """
        Prune the ensemble by greedily selecting the best classifiers, supporting both batched and non-batched computations.

    

        Args:
            proba (numpy.ndarray): The predicted probabilities for the ensemble (for non-batched computation)
            target (numpy.ndarray): The true target values for evaluation (for non-batched computation)
            proba_target_path (str or None, optional): Path to the batch directory, which stores proba and target arrays (if batch processing is used, check the ProbaTargetLoader for more details)

        Returns:
            tuple: A tuple containing the list of selected classifier indices and their corresponding weights.
        """


        if(proba_target_path is None):
            n_received = proba.shape[0]
        else:
            proba_target_loader = ProbaTargetLoader(proba_target_path)
            first_proba,_ = next(proba_target_loader)
            n_received = first_proba.shape[0]

        
       

        not_selected_models = list(range(n_received))
        selected_models = [ ]

        for _ in range(self.n_estimators):
            scores = self.batched_metric_greedy(proba, target,proba_target_path, selected_models,not_selected_models)
           
            best_model, _ = min(scores, key = lambda e: e[1])
            not_selected_models.remove(best_model)
            selected_models.append(best_model)

        return selected_models, [1.0 / len(selected_models) for _ in selected_models]
    


    
            
            
            


        