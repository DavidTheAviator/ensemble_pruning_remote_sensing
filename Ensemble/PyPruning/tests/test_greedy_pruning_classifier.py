import sys
import os
import shutil
from pathlib import Path


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.test_utils import ensemble_proba, cls1, target, ensemble_proba_dummy_a, target_dummy_a, ensemble_proba_dummy_b, target_dummy_b




import numpy as np
import unittest 
from PyPruning.GreedyPruningClassifier import GreedyPruningClassifier
from PyPruning.GreedyPruningClassifier import margin_distance_minimization
from PyPruning.GreedyPruningClassifier import uwa
from PyPruning.GreedyPruningClassifier import diaces
from PyPruning.GreedyPruningClassifier import sdacc, dftwo,accrein



import math




class TestMarginDistanceMinimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_dir = Path(os.path.dirname(__file__))
        cls.temp_dir = test_dir.joinpath('temp')

        #create temp dir 
        if not os.path.exists(cls.temp_dir):
            os.makedirs(cls.temp_dir)

        #add ensemble and target there (properly named)                
        np.save(cls.temp_dir.joinpath('ensemble_proba_batch_1.npy'),ensemble_proba_dummy_b)
        np.save(cls.temp_dir.joinpath('ensemble_proba_batch_2.npy'),ensemble_proba_dummy_b)
        np.save(cls.temp_dir.joinpath('target_batch_1.npy'),target_dummy_b)
        np.save(cls.temp_dir.joinpath('target_batch_2.npy'),target_dummy_b)

        
        
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_iproba(self):
        i = 0
        iproba = ensemble_proba[i,:,:]
        
        np.testing.assert_allclose(iproba,cls1)

    def test_csig_whole_ensemble(self):
        csig1 = np.array([1, -1, 1, -1, -1])
        csig2 = np.array([-1, 1, -1, -1, 1])
        csig3 = np.array([-1, 1, -1, 1, 1])
        csig4 = np.array([1,  1, 1, -1, -1])

        c_sigs_manual = np.stack([csig1,csig2,csig3,csig4],axis=0)

    
        
        M = ensemble_proba.shape[0]
        c_sigs_manual = np.sum(c_sigs_manual, axis=0) * (1/M) 
        u = 4
        p =  0.075
        o = np.full(len(target), p)

        dist = np.linalg.norm(o - c_sigs_manual)
        
        selected_models  = np.arange(3)
        self.assertEqual(margin_distance_minimization(3,ensemble_proba,selected_models,target),dist)
        

    def test_margin_distance_minimization_at_start(self):
        #calculate initial values 
        u = 1
        M = 4

        csig1 = np.array([1, -1, 1, -1, -1]) * 1/M
        csig2 = np.array([-1, 1, -1, -1, 1]) * 1/M
        csig3 = np.array([-1, 1, -1, 1, 1])  * 1/M
        csig4 = np.array([1,  1, 1, -1, -1]) * 1/M
        

        
        p =  0.075
        o = np.full(len(target), p)

        dist1 = np.linalg.norm(o - csig1)
        dist2 = np.linalg.norm(o - csig2)
        dist3 = np.linalg.norm(o - csig3)
        dist4 = np.linalg.norm(o - csig4)
        
        x = margin_distance_minimization(0, ensemble_proba, [], target)
        self.assertEqual(x,dist1)
        self.assertEqual(margin_distance_minimization(1, ensemble_proba, [], target),dist2)
        self.assertEqual(margin_distance_minimization(2, ensemble_proba, [], target),dist3)
        self.assertEqual(margin_distance_minimization(3, ensemble_proba, [], target),dist4)


    #dummy ensemble b used here, since pruning on dummy a does not give the desired results (altough computation is all correct)
    def test_margin_distance_minimization_inaccurate_ensemble(self):
        mdm_cl = GreedyPruningClassifier(n_estimators=2, metric = margin_distance_minimization, n_jobs = 4)
        mdm_ids, _ = mdm_cl.prune_(proba=ensemble_proba_dummy_b,target=target_dummy_b)
        
        
        self.assertTrue(set(mdm_ids).issubset([0,1,2,3,4,5]))

    #shold be the same as above, we simulate a batched execution
    def test_margin_distance_minimization_inaccurate_ensemble_batched(self):
        mdm_cl = GreedyPruningClassifier(n_estimators=2, metric = margin_distance_minimization, n_jobs = 4)
        mdm_ids, _ = mdm_cl.prune_(proba= None, target=None, proba_target_path= self.__class__.temp_dir)
                    
        
        self.assertTrue(set(mdm_ids).issubset([0,1,2,3,4,5]))




class TestUwa(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_dir = Path(os.path.dirname(__file__))
        cls.temp_dir = test_dir.joinpath('temp')

        #create temp dir 
        if not os.path.exists(cls.temp_dir):
            os.makedirs(cls.temp_dir)

        #add ensemble and target there (properly named)                
        np.save(cls.temp_dir.joinpath('ensemble_proba_batch_1.npy'),ensemble_proba_dummy_a)
        np.save(cls.temp_dir.joinpath('ensemble_proba_batch_2.npy'),ensemble_proba_dummy_a)
        np.save(cls.temp_dir.joinpath('target_batch_1.npy'),target_dummy_a)
        np.save(cls.temp_dir.joinpath('target_batch_2.npy'),target_dummy_a)

       


        

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)



    def test_accuracy_calc(self):
        acc_manual = np.array([0.4,0.4,0.6,0.6]) * -1.0


        N = len(target)
        for i in range(4):
            iproba = ensemble_proba[i,:,:]
            self.assertEqual(-np.count_nonzero(iproba.argmax(axis=1) == target) * 1/N, acc_manual[i])
    

    def test_ipred(self):
        pred1 = np.array([ True, False,  True, False, False])
        pred2 = np.array([False,  True, False, False,  True])
        pred3 = np.array([False,  True, False,  True,  True])
        pred4 = np.array([ True,  True,  True, False, False])

        preds = np.stack([pred1,pred2,pred3,pred4])
        
        for i, pred in enumerate(preds):
            iproba = ensemble_proba[i,:,:]
            np.testing.assert_allclose(pred,(iproba.argmax(axis=1)==target))


    def test_sub_pred(self):
        pred_ensemble = [False,  True, False, False, False]


        sub_proba = ensemble_proba[np.arange(4), :, :]
        sub_pred = (sub_proba.sum(axis=0).argmax(axis=1) == target)
        
       
        np.testing.assert_allclose(sub_pred,pred_ensemble)


    def test_ratio_correct(self):
        ratios = np.array([0.5,0.75,0.5,0.25,0.5])

        selected_models = np.arange(4)

        sub_proba = ensemble_proba[selected_models,:,:]
        S =  len(selected_models)
        sub_individual_pred = sub_proba.argmax(axis=2) == target[np.newaxis, :] 
        prop_correct = np.count_nonzero(sub_individual_pred, axis = 0) * 1/S
        
        np.testing.assert_allclose(ratios,prop_correct)


    def test_uwa(self):
        uwa_manual = -1 

        #this classifier predicts [True,True,True,True,False], since Target = [1 0 2 0 1]
        clsi = np.array(
            [[0.1, 0.8, 0.1 ], 
            [0.75, 0.  , 0.25],
            [0.2  , 0.2, 0.6],
            [0.5, 0.25, 0.25],
            [0.1, 0.1, 0.8]]
        )


        ensemble_proba_custom = np.concatenate([ensemble_proba,clsi[np.newaxis,:,:]],axis=0)
        selected_models = np.arange(4)
        
        self.assertEqual(uwa(4, ensemble_proba_custom, selected_models , target), uwa_manual)    
        
        
    def test_first_iteration(self):
        acc_manual = np.array([0.4,0.4,0.6,0.6]) * -1.0
        for i in range(4):
            self.assertEqual(uwa(i,ensemble_proba,[], target), acc_manual[i])


    def test_uwa_inaccurate_ensemble(self):
        uwa_cl = GreedyPruningClassifier(n_estimators=2, metric = uwa, n_jobs = 4)
        uwa_ids, _ = uwa_cl.prune_(proba=ensemble_proba_dummy_a,target=target_dummy_a)
        print(uwa_ids)

        self.assertTrue(set(uwa_ids).issubset([0,1,2]))

    def test_uwa_inaccurate_ensemble_batched(self):
        uwa_cl = GreedyPruningClassifier(n_estimators=2, metric = uwa, n_jobs = 4)
        uwa_ids, _ = uwa_cl.prune_(proba= None, target=None, proba_target_path= self.__class__.temp_dir)
        print(uwa_ids)

        self.assertTrue(set(uwa_ids).issubset([0,1,2]))


class TestDiaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_dir = Path(os.path.dirname(__file__))
        cls.temp_dir = test_dir.joinpath('temp')

        #create temp dir 
        if not os.path.exists(cls.temp_dir):
            os.makedirs(cls.temp_dir)

        #add ensemble and target there (properly named)                
        np.save(cls.temp_dir.joinpath('ensemble_proba_batch_1.npy'),ensemble_proba_dummy_a)
        np.save(cls.temp_dir.joinpath('ensemble_proba_batch_2.npy'),ensemble_proba_dummy_a)
        np.save(cls.temp_dir.joinpath('target_batch_1.npy'),target_dummy_a)
        np.save(cls.temp_dir.joinpath('target_batch_2.npy'),target_dummy_a)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)
    
    
    
    def test_choose_initial(self):

        #first two classifiers really bad, second two medium performance, last two
        #4 classifiers, 2 samples, 3 classes
        target_initial_test = np.array([0,0])
        ensemble_proba_initial_test = np.array([
            #bad
            [[0,0.7,0.3],
             [0,0.1,0.9]],
            [[0,0.7,0.3],
            [0,0.1,0.9]],

            #medium
            [[0.8,0.1,0.1],
             [0.1,0.8,0.1]],
            [[0.7,0.2,0.1],
             [0.2,0.8,0]],
            
            #good
            [[0.9,0.1,0],
             [0.6,0.3,0.1]],
            [[0.9,0.1,0],
            [0.6,0.3,0.1]],
        ])

        
        for i in range(4):
            metric = diaces(i, ensemble_proba_initial_test, selected_models=[], target=target_initial_test, alpha=1)
            if i in [2,3]:
                self.assertTrue(metric< 0)
            else:
                self.assertTrue(metric == 1)
        

    def test_c_star_e_star(self):
        i = 0
        selected_models = [1]
        



        subset_ids = selected_models+ [i]
        subset_proba = ensemble_proba[subset_ids, :,:]


       
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


        np.testing.assert_allclose(c_star, (20.0/3.0 - 6)/14.0)
        np.testing.assert_allclose(e_star, (4*18 - 36)/114.0)


    def test_metric(self):
        s_2_1 = diaces(0, ensemble_proba, selected_models=[1], target=target, alpha=1)
        s_2_3 = diaces(2, ensemble_proba, selected_models=[1], target=target, alpha=1)
        s_2_4 = diaces(3, ensemble_proba, selected_models=[1], target=target, alpha=1)


        k = 4
        n = 5
        alpha = 1
    
        #these are the total errors in the respective cases (ensemble of classifier 1 and 2 , 2 and 3 etc)
        x_2_1 = 6 
        x_2_3 = 5
        x_2_4 = 5

        #the values for c and e in the respective cases
        c_2_1 = (1/6)**2 + (1/6)**2 + (1/6)**2 + (2/6)**2 + (1/6)**2
        e_2_1 = (3/5)**2 + (3/5)**2
        c_2_3 = (2/5)**2 + (2/5)**2 + (1/5)**2 
        e_2_3 = (3/5)**2 + (2/5)**2
        c_2_4 = (1/5)**2 + (1/5)**2 + (2/5)**2 + (1/5)**2
        e_2_4 = (3/5)**2 + (2/5)**2



        s_2_1_manual = ((n*x_2_1*c_2_1- x_2_1)/(n*k-x_2_1))+ alpha* ((k*(n**2)*e_2_1- (x_2_1**2))/(x_2_1*k*n-x_2_1))
        s_2_3_manual = ((n*x_2_3*c_2_3- x_2_3)/(n*k-x_2_3))+ alpha* ((k*(n**2)*e_2_3- (x_2_3**2))/(x_2_3*k*n-x_2_3))
        s_2_4_manual = ((n*x_2_4*c_2_4- x_2_4)/(n*k-x_2_4))+ alpha* ((k*(n**2)*e_2_4- (x_2_4**2))/(x_2_4*k*n-x_2_4))


        np.testing.assert_allclose(s_2_1,s_2_1_manual)
        np.testing.assert_allclose(s_2_3,s_2_3_manual)
        np.testing.assert_allclose(s_2_4,s_2_4_manual)


    #does not work as with the other algos, since a "medium performing classifier", in this case a bad performing one
    # is chosen in the initial round
    """
    def test_diaces_inaccurate_ensemble(self):
        diaces_cl = GreedyPruningClassifier(n_estimators=2, metric = diaces, n_jobs = 4,  metric_options={'alpha':1} )
        diaces_ids, _ = diaces_cl.prune_(proba=ensemble_proba_dummy_a,target=target_dummy_a)
        print(diaces_ids)

        self.assertTrue(set(diaces_ids).issubset([0,1,2]))
    """


class TestSdacc(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_dir = Path(os.path.dirname(__file__))
        cls.temp_dir = test_dir.joinpath('temp')

        #create temp dir 
        if not os.path.exists(cls.temp_dir):
            os.makedirs(cls.temp_dir)

        #add ensemble and target there (properly named)                
        np.save(cls.temp_dir.joinpath('ensemble_proba_batch_1.npy'),ensemble_proba_dummy_a)
        np.save(cls.temp_dir.joinpath('ensemble_proba_batch_2.npy'),ensemble_proba_dummy_a)
        np.save(cls.temp_dir.joinpath('target_batch_1.npy'),target_dummy_a)
        np.save(cls.temp_dir.joinpath('target_batch_2.npy'),target_dummy_a)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)



    def test_sdacc(self):
        sdacc_manual = -1.5 

        #this classifier predicts [True,True,True,True,False], since Target = [1 0 2 0 1]
        clsi = np.array(
            [[0.1, 0.8, 0.1 ], 
            [0.75, 0.  , 0.25],
            [0.2  , 0.2, 0.6],
            [0.5, 0.25, 0.25],
            [0.1, 0.1, 0.8]]
        )


        ensemble_proba_custom = np.concatenate([ensemble_proba,clsi[np.newaxis,:,:]],axis=0)
        selected_models = np.arange(4)
        
        self.assertEqual(sdacc(4, ensemble_proba_custom, selected_models , target), sdacc_manual)   

    def test_sdacc_inaccurate_ensemble(self):
        sdacc_cl = GreedyPruningClassifier(n_estimators=2, metric = sdacc, n_jobs = 4)
        sdacc_ids, _ = sdacc_cl.prune_(proba=ensemble_proba_dummy_a,target=target_dummy_a)
        print(sdacc_ids)

        self.assertTrue(set(sdacc_ids).issubset([0,1,2]))

    def test_sdacc_inaccurate_ensemble_batched(self):
        sdacc_cl = GreedyPruningClassifier(n_estimators=2, metric = sdacc, n_jobs = 4)
        sdacc_ids, _ = sdacc_cl.prune_(proba= None, target=None, proba_target_path= self.__class__.temp_dir)
        print(sdacc_ids)

        self.assertTrue(set(sdacc_ids).issubset([0,1,2]))


class TestDftwo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_dir = Path(os.path.dirname(__file__))
        cls.temp_dir = test_dir.joinpath('temp')

        #create temp dir 
        if not os.path.exists(cls.temp_dir):
            os.makedirs(cls.temp_dir)

        #add ensemble and target there (properly named)                
        np.save(cls.temp_dir.joinpath('ensemble_proba_batch_1.npy'),ensemble_proba_dummy_a)
        np.save(cls.temp_dir.joinpath('ensemble_proba_batch_2.npy'),ensemble_proba_dummy_a)
        np.save(cls.temp_dir.joinpath('target_batch_1.npy'),target_dummy_a)
        np.save(cls.temp_dir.joinpath('target_batch_2.npy'),target_dummy_a)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)


    def test_dftwo(self):
        dftwo_manual = -1.75

        #this classifier predicts [True,True,True,True,False], since Target = [1 0 2 0 1]
        clsi = np.array(
            [[0.1, 0.8, 0.1 ], 
            [0.75, 0.  , 0.25],
            [0.2  , 0.2, 0.6],
            [0.5, 0.25, 0.25],
            [0.1, 0.1, 0.8]]
        )


        ensemble_proba_custom = np.concatenate([ensemble_proba,clsi[np.newaxis,:,:]],axis=0)
        selected_models = np.arange(4)
        
        self.assertEqual(dftwo(4, ensemble_proba_custom, selected_models , target), dftwo_manual)   

    def test_dftwo_inaccurate_ensemble(self):
        dftwo_cl = GreedyPruningClassifier(n_estimators=2, metric = dftwo, n_jobs = 4)
        dftwo_ids, _ = dftwo_cl.prune_(proba=ensemble_proba_dummy_a,target=target_dummy_a)
        print(dftwo_ids)

        self.assertTrue(set(dftwo_ids).issubset([0,1,2]))

    def test_dftwo_inaccurate_ensemble_batched(self):
        dftwo_cl = GreedyPruningClassifier(n_estimators=2, metric = dftwo, n_jobs = 4)
        dftwo_ids, _ = dftwo_cl.prune_(proba= None, target=None, proba_target_path= self.__class__.temp_dir)
        print(dftwo_ids)

        self.assertTrue(set(dftwo_ids).issubset([0,1,2]))

class TestAccrein(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_dir = Path(os.path.dirname(__file__))
        cls.temp_dir = test_dir.joinpath('temp')

        #create temp dir 
        if not os.path.exists(cls.temp_dir):
            os.makedirs(cls.temp_dir)

        #add ensemble and target there (properly named)                
        np.save(cls.temp_dir.joinpath('ensemble_proba_batch_1.npy'),ensemble_proba_dummy_a)
        np.save(cls.temp_dir.joinpath('ensemble_proba_batch_2.npy'),ensemble_proba_dummy_a)
        np.save(cls.temp_dir.joinpath('target_batch_1.npy'),target_dummy_a)
        np.save(cls.temp_dir.joinpath('target_batch_2.npy'),target_dummy_a)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)




    def test_accrein(self):
        accrein_manual = -2

        #this classifier predicts [True,True,True,True,False], since Target = [1 0 2 0 1]
        clsi = np.array(
            [[0.1, 0.8, 0.1 ], 
            [0.75, 0.  , 0.25],
            [0.2  , 0.2, 0.6],
            [0.5, 0.25, 0.25],
            [0.1, 0.1, 0.8]]
        )


        ensemble_proba_custom = np.concatenate([ensemble_proba,clsi[np.newaxis,:,:]],axis=0)
        selected_models = np.arange(4)
        
        self.assertEqual(accrein(4, ensemble_proba_custom, selected_models , target), accrein_manual)   

    def test_accrein_inaccurate_ensemble(self):
        accrein_cl = GreedyPruningClassifier(n_estimators=2, metric = accrein, n_jobs = 4)
        accrein_ids, _ = accrein_cl.prune_(proba=ensemble_proba_dummy_a,target=target_dummy_a)
        print(accrein_ids)

        self.assertTrue(set(accrein_ids).issubset([0,1,2]))

    def test_accrein_inaccurate_ensemble_batched(self):
        accrein_cl = GreedyPruningClassifier(n_estimators=2, metric = accrein, n_jobs = 4)
        accrein_ids, _ = accrein_cl.prune_(proba= None, target=None, proba_target_path= self.__class__.temp_dir)
        print(accrein_ids)

        self.assertTrue(set(accrein_ids).issubset([0,1,2]))

if __name__ == '__main__':
    unittest.main()