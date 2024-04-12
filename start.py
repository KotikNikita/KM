import matplotlib.pyplot as plt
import numpy as np
from src.methods.multiclass_methods import MulticlassSVC
from src.util.kernels import RBF, Polynomial, Intersection, Linear, Chi2
import src.util.utils as ut
from src.methods.kernel_pca import *
from src.methods.feature_extractor import *

data_path = 'data/'
seed = 42

np.random.seed(seed)

Xtr,Ytr,Xte = ut.read_data(data_path)
write_test_res = True
experiment_name = 'final_model'

if not write_test_res:
    Xtr, Xte, Ytr, Yte = ut.train_test_split(Xtr, Ytr, test_size=0.2, shuffle=True)

# Transform into multilevel energy features
Xtr, Ytr = ut.augment_data(Xtr, Ytr)
#print('Xtr shape:', Xtr.shape)
#print('Ytr shape:', Ytr.shape)

nb_filters = 8

filters = create_filters(nb_filters, 3, 1, lambda x,y : f1(x,y,1,3,1))
filters2 = create_filters(nb_filters, 3, 1, lambda x,y : f2(x,y,1,3,1))
both_filters = np.concatenate([filters, filters2])

mlef = multi_level_energy_features(8, both_filters, non_max=False)
Xtr = mlef.transform_all(Xtr) + 1e-6
Xte = mlef.transform_all(Xte) + 1e-6

kernel = Chi2(gamma=1.5).kernel
C = 1
classifier = MulticlassSVC(10, kernel, C, tol = 1e-2)
print('Fitting classifier C = ', C)
classifier.fit(Xtr, Ytr, verbose = True, use_weights=True, solver='cvxopt')
predictions, scores = classifier.predict(Xtr)
#print((scores > 0).any())
#print("Training accuracy ;", ut.accuracy(Ytr, predictions))

#if not write_test_res :
#    predictions, scores = classifier.predict(Xte)
#    print((scores > 0).any())
#    print("Validation accuracy ;",ut.accuracy(Yte, predictions))
#    print('Confusion matrix')
#    print(ut.compute_confusion_matrix(Yte,predictions,10))

#else :
    
#    predictions, scores = classifier.predict(Xte)
#    print((scores > 0).any())
#    ut.save_results(predictions,results_name='Yte_pred_'+ experiment_name+'.csv' , results_path=data_path)

if write_test_res:
    predictions, scores = classifier.predict(Xte)
    print((scores > 0).any())
    ut.save_results(predictions,results_name='Yte_pred_'+ experiment_name+'.csv' , results_path=data_path)
