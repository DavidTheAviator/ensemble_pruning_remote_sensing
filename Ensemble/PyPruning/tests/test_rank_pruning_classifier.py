import sys
import os
import shutil 
from pathlib import Path

from PyPruning.RankPruningClassifier import RankPruningClassifier
from PyPruning.RankPruningClassifier import angle_between
from PyPruning.RankPruningClassifier import orientation_ordering
from PyPruning.RankPruningClassifier import individual_contribution_ordering
from PyPruning.RankPruningClassifier import individual_margin_diversity

import math
import numpy as np
import unittest 


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.test_utils import ensemble_proba, cls1, target, V, ensemble_proba_dummy_a, ensemble_proba_dummy_b, target_dummy_a, target_dummy_b









class TestOrientationOrdering(unittest.TestCase):
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


    def test_angle_between(self):
        a1 = np.array([1,0])
        b1 = np.array([1,0])

        a2 = np.array([1,0])
        b2 = np.array([0,1])

        a3 = np.array([1,0])
        b3 = np.array([-1,0])

        a4 = np.array([1,0])
        b4 = np.array([0,-1])

        self.assertEqual(angle_between(a1,b1),0)
        self.assertEqual(angle_between(a2,b2),math.pi/2)
        self.assertEqual(angle_between(a3,b3),math.pi)
        self.assertEqual(angle_between(a4,b4),math.pi/2)

    def test_cref(self):
        c_sig = 2.0*(ensemble_proba.argmax(axis=2) == target[np.newaxis,:]) -1.0
        #ensemble signature vector
        c_ens = c_sig.mean(axis=0)
        o = np.ones(len(c_ens), dtype=float)


        lamb = np.dot(-o,c_ens)/np.dot(c_ens,c_ens)
        c_ref = o + lamb * c_ens
        
        c_ref_manual = np.ones(5)

        #example from paper 
        c_ens_paper = np.array([1,0.5,-0.5])
        o_paper = np.ones(3, dtype=float)


        lamb_paper = np.dot(-o_paper,c_ens_paper)/np.dot(c_ens_paper,c_ens_paper)
        c_ref_paper = o_paper + lamb_paper * c_ens_paper
        c_ref_paper_manual = np.array([1/3,2/3,4/3])


        np.testing.assert_allclose(c_ref,c_ref_manual)
        np.testing.assert_allclose(c_ref_paper,c_ref_paper_manual)



    def test_orientation_ordering(self):
        csig1 = np.array([1, -1, 1, -1, -1])
        csig2 = np.array([-1, 1, -1, -1, 1])
        csig3 = np.array([-1, 1, -1, 1, 1]) 
        csig4 = np.array([1,  1, 1, -1, -1])

        oo_manual = np.array([
            angle_between(np.ones(5),csig1),
            angle_between(np.ones(5),csig2),
            angle_between(np.ones(5),csig3),
            angle_between(np.ones(5),csig4)
        ])


        for i in range(4):
            self.assertEqual(orientation_ordering(i, ensemble_proba, target),oo_manual[i])


    def test_orientation_ordering_c_ref_passed(self):
        csig1 = np.array([1, -1, 1, -1, -1])
        csig2 = np.array([-1, 1, -1, -1, 1])
        csig3 = np.array([-1, 1, -1, 1, 1]) 
        csig4 = np.array([1,  1, 1, -1, -1])

        oo_manual = np.array([
            angle_between(np.ones(5),csig1),
            angle_between(np.ones(5),csig2),
            angle_between(np.ones(5),csig3),
            angle_between(np.ones(5),csig4)
        ])

        c_ref = np.ones(5)

        for i in range(4):
            self.assertEqual(orientation_ordering(i, ensemble_proba, target, c_ref),oo_manual[i])

    

    def test_orientation_ordering_inaccurate_ensemble(self):
        oo_cl = RankPruningClassifier(n_estimators=2, metric = orientation_ordering, n_jobs = 4)
        oo_ids, _ = oo_cl.prune_(proba= ensemble_proba_dummy_a ,target = target_dummy_a)

        self.assertTrue(set(oo_ids).issubset([0,1,2]))

    def test_orientation_ordering_inaccurate_ensemble_batched(self):
        oo_cl = RankPruningClassifier(n_estimators=2, metric = orientation_ordering, n_jobs = 4)
        oo_ids, _ = oo_cl.prune_(proba= None, target=None, proba_target_path= self.__class__.temp_dir)

        self.assertTrue(set(oo_ids).issubset([0,1,2]))
        
    


    
        
class TestIndividualContributionOrdering(unittest.TestCase):
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




   

    def test_votes_array(self):
        V_calc = np.zeros(ensemble_proba.shape)
        idx = ensemble_proba.argmax(axis=2)
        V_calc[np.arange(ensemble_proba.shape[0])[:,None],np.arange(ensemble_proba.shape[1]),idx] = 1
        V_calc = V_calc.sum(axis=0)
        
        np.testing.assert_allclose(V, V_calc)


    
    def test_vote_vectors(self):
        ensemble_preds = ensemble_proba.argmax(axis=2) 
        n = ensemble_proba.shape[1]


        partition = np.partition(V, -2)

        #highest number of votes for a class (per sample)
        v_max = partition[:,-1]

        #second highest number of votes for a class (per sample)
        v_sec = partition[:,-2]




        #number of votes for the correct class
        v_correct = V[np.arange(n),target]


        v_correct_manual = np.array([2,3,2,1,2])
        v_max_manual = np.array([2,3,2,2,2])
        v_sec_manual = np.array([1,1,1,1,1])

        np.testing.assert_allclose(v_correct_manual, v_correct)
        np.testing.assert_allclose(v_max_manual,v_max)
        np.testing.assert_allclose(v_sec_manual,v_sec)


        for i in range(4):   
            iproba = ensemble_proba[i,:,:]
            pred = iproba.argmax(axis=1)
            #number of votes for the prediction of the classifier
            v_c = V[np.arange(n),pred]

            v_classifier_manual = V[np.arange(V.shape[0]),ensemble_preds[i]]
            np.testing.assert_allclose(v_classifier_manual, v_c)


    def test_individual_contribution_ordering(self):
        ensemble_preds = ensemble_proba.argmax(axis=2) 
        ensemble_pred_vote = np.array([1, 0, 2, 2, 1])

        def ic_manual_helper():
            l = []
            for i in range(4):
                 
                ensemble_pred_combined = ensemble_pred_vote


                alpha = np.logical_and(target==ensemble_preds[i],ensemble_pred_combined!= target)
                beta = np.logical_and(target==ensemble_preds[i],ensemble_pred_combined== target)
                theta = target!= ensemble_preds[i]

                v_correct = np.array([2,3,2,1,2])
                v_max = np.array([2,3,2,2,2])
                v_sec = np.array([1,1,1,1,1])
                v_classifier = V[np.arange(V.shape[0]),ensemble_preds[i]]

                


                alpha = alpha.astype(float) * (2*v_max - v_classifier)
                beta = beta.astype(float) * v_sec
                theta = theta.astype(float) * (v_correct- v_classifier - v_max)

                IC = alpha+beta+theta
                IC = IC.sum()
                
                l.append(IC)

            return np.array(l)



        ic_vote_manual = ic_manual_helper()  * -1.0


  

        for i in range(4):
            self.assertEqual(ic_vote_manual[i],individual_contribution_ordering(i, ensemble_proba, target))
   

            #V passed
            self.assertEqual(ic_vote_manual[i],individual_contribution_ordering(i, ensemble_proba, target,V=V))
 


    def test_individual_contribution_ordering_inaccurate_ensemble(self):
        epic_cl = RankPruningClassifier(n_estimators=2, metric = individual_contribution_ordering, n_jobs = 4)
        epic_ids, _ = epic_cl.prune_(proba= ensemble_proba_dummy_a ,target = target_dummy_a)

        self.assertTrue(set(epic_ids).issubset([0,1,2]))

    def test_individual_contribution_ordering_inaccurate_ensemble_batched(self):
        epic_cl = RankPruningClassifier(n_estimators=2, metric = individual_contribution_ordering, n_jobs = 4)
        epic_ids, _ = epic_cl.prune_(proba= None, target=None, proba_target_path= self.__class__.temp_dir)

        self.assertTrue(set(epic_ids).issubset([0,1,2]))
        
    
    




class TestIndividualMarginDiversity(unittest.TestCase):
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
    
    
    
    
    def test_margin_diversity_array(self):
        margin_manual = [0.25, 0.5, 0.25, -0.25, 0.25]
        one_minus_cond_div_manual = [0.5,0.75,0.5,0.25,0.5]
        
        
        n = ensemble_proba.shape[1]
        m = ensemble_proba.shape[0]


        #votes for the correct class on each sample
        v_correct = V[np.arange(n),target]

        #max votes on each sample (excluding the correct class)
        V_correct_masked = np.ma.array(V,mask=False)
        V_correct_masked.mask[np.arange(n), target] = True
        v_pseudo_max = V_correct_masked.max(axis=1)

        margin_calc = (v_correct - v_pseudo_max) * 1/m
        con_div_calc = v_correct*1/m

        np.testing.assert_allclose(margin_calc, margin_manual)
        np.testing.assert_allclose(con_div_calc, one_minus_cond_div_manual)


 

    def test_individual_margin_diversity(self):
        margin_manual = [0.25, 0.5, 0.25, -0.25, 0.25]
        one_minus_cond_div_manual = [0.5,0.75,0.5,0.25,0.5]

        for i in range(4):
            calc_mask = (ensemble_proba[i,:,:].argmax(axis=1) == target).astype(float)
            mdm =  0.2 * np.log(np.abs(margin_manual)) + 0.8* np.log(one_minus_cond_div_manual)
            mdm = (mdm * calc_mask).sum()
            mdm = mdm
            self.assertEqual(mdm, individual_margin_diversity(i, ensemble_proba, target, alpha = 0.2, V=None))
            self.assertEqual(mdm, individual_margin_diversity(i, ensemble_proba, target, alpha = 0.2, V=V))



    def test_individual_margin_diversity_inaccurate_ensemble(self):
        mdep_cl = RankPruningClassifier(n_estimators=2, metric = individual_margin_diversity, n_jobs = 4)
        mdep_ids, _ = mdep_cl.prune_(proba= ensemble_proba_dummy_a ,target = target_dummy_a)
        print(mdep_ids)

        self.assertTrue(set(mdep_ids).issubset([0,1,2]))


    def test_individual_margin_diversity_inaccurate_ensemble_batched(self):
        mdep_cl = RankPruningClassifier(n_estimators=2, metric = individual_margin_diversity, n_jobs = 4)
        mdep_ids, _ = mdep_cl.prune_(proba= None, target=None, proba_target_path= self.__class__.temp_dir)
        print(mdep_ids)

        self.assertTrue(set(mdep_ids).issubset([0,1,2]))
            
        






if __name__ == '__main__':
    unittest.main()