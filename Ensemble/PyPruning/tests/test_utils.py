import numpy as np

def predict_crips(ensemble_pred,n_classes):
    
    N = ensemble_pred.shape[1]
    C = n_classes
    class_count =  np.zeros((N,C))

    for pred in ensemble_pred:
        class_count[np.arange(N),pred] += 1


    

    return class_count.argmax(axis=1)


cls1 = np.array(
    [[0.37, 0.45, 0.18],
    [0.23, 0.15, 0.62],
    [0.24, 0.33, 0.43],
    [0.  , 0.47, 0.53],
    [0.06, 0.44, 0.5 ]]
)


cls2 = np.array(
    [[0.56, 0.11, 0.33],
    [0.55, 0.36, 0.09],
    [0.56, 0.22, 0.22],
    [0.  , 0.27, 0.73],
    [0.12, 0.44, 0.44]]
)

cls3 = np.array(
    [[0.07, 0.33, 0.6 ],
    [0.75, 0.  , 0.25],
    [0.  , 0.82, 0.18],
    [0.41, 0.18, 0.41],
    [0.28, 0.44, 0.28]]
)

cls4 = np.array(
    [[0.2  , 0.4  , 0.4  ],
    [0.64, 0.36, 0.  ],
    [0.32, 0.32, 0.36],
    [0.12, 0.88, 0.  ],
    [0.5 , 0.12, 0.38]]
)




target =    np.array([1,0,2,0,1])
cls1_pred = np.array([1, 2, 2, 2, 2])
cls2_pred = np.array([0, 0, 0, 2, 1])
cls3_pred = np.array([2, 0, 1, 0, 1])
cls4_pred = np.array([1, 0, 2, 1, 0])



#ensemble_proba:  4 classfifiers x 5 samples x 3 classes (MxNxC)
ensemble_proba = np.stack([cls1,cls2,cls3,cls4])


#ensemble_pred: 4 classifiers x 5 samples (MxN)
ensemble_preds = ensemble_proba.argmax(axis=2)


#(N)
final_pred = predict_crips(ensemble_preds,3)


V = np.array([
[1., 2., 1.],
 [3., 0., 1.],
 [1., 1., 2.],
 [1., 1., 2.],
 [1., 2., 1.]])




########DUMMY ENSEMBLES 

### Only 3 out of 10 accurate 
# 10 Classifiers, 2 Samples, 3 Classes
#Sample 1, class 0
#Sample 2, class 1
#Sample 3, class 1

#accurate
dummy_a_1 = np.array([[0.8,0.1,0.1],[0.1,0.7,0.2]])
dummy_a_2 = np.array([[0.6,0.3,0.1],[0.05,0.9,0.05]])
dummy_a_3 = np.array([[0.9,0.06,0.04],[0.2,0.6,0.2]])


#bad
dummy_a_4 = np.array([[0.2,0.7,0.1],   [0.1,0.15,0.75]])
dummy_a_5 = np.array([[0.4,0,0.6],     [0.6,0.1,0.3]])
dummy_a_6 = np.array([[0.3,0.7,0],     [0.6,0.4,0]])
dummy_a_7 = np.array([[0,0.5,0.5],     [0.99,0.1,0]])
dummy_a_8 = np.array([[0.05,0.4,0.55], [0,0.01,0.99]])
dummy_a_9 = np.array([[0.25,0.25,0.5], [0.75,0.2,0.05]])
dummy_a_10 = np.array([[0.15,0.15,0.7],[0.25,0.15,0.6]])

target_dummy_a = np.array([0,1])

ensemble_proba_dummy_a = np.stack([dummy_a_1, dummy_a_2,dummy_a_3, dummy_a_4, dummy_a_5, dummy_a_6, dummy_a_7, dummy_a_8, dummy_a_9, dummy_a_10])


### Classifier 1,2,3 (first right); 4,5,6 (second right); and 7,8,9,10  (none right) make the same prediction
# 10 Classifiers, 2 Samples, 3 Classes
#Sample 1, class 0
#Sample 2, class 1

#get first correct
dummy_b_1 = np.array([[0.8,0.1,0.1],[0.7,0.15,0.15]])
dummy_b_2 = np.array([[0.8,0.1,0.1],[0.7,0.15,0.15]])
dummy_b_3 = np.array([[0.8,0.1,0.1],[0.7,0.15,0.15]])


#get second correct
dummy_b_4 = np.array([[0.25,0.3,0.45],[0.05,0.9,0.05]])
dummy_b_5 = np.array([[0.25,0.3,0.45],[0.05,0.9,0.05]])
dummy_b_6 = np.array([[0.25,0.3,0.45],[0.05,0.9,0.05]])

#get none correct
dummy_b_7 = np.array([[0.15,0.15,0.7],[0.25,0.15,0.6]])
dummy_b_8 = np.array([[0.15,0.15,0.7],[0.25,0.15,0.6]])
dummy_b_9 = np.array([[0.15,0.15,0.7],[0.25,0.15,0.6]])
dummy_b_10 = np.array([[0.15,0.15,0.7],[0.25,0.15,0.6]])


target_dummy_b = np.array([0,1])

ensemble_proba_dummy_b = np.stack([dummy_b_1, dummy_b_2, dummy_b_3, dummy_b_4, dummy_b_5, dummy_b_6, dummy_b_7, dummy_b_8, dummy_b_9, dummy_b_10])
