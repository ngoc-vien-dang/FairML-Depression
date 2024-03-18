"""
File name: sup.py
Author: ngocviendang
Date created: March 13, 2024

This file contains functions for addressing algorithmic bias (sup).
"""
import numpy as np
from FairML.utils import helper
from FairML.utils import dataset
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from joblib import dump, load
import copy
def sup(X_train,y_train,X_test,y_test,cls_mdls,unprivileged_groups,privileged_groups,\
              conditions,label_name,protected_attribute_name,protected_attribute_values,i,model_func,sup):
    train_ds = BinaryLabelDataset(df=X_train.join(y_train),\
                    label_names=[label_name],\
                    protected_attribute_names=[protected_attribute_name],
                    favorable_label=1, unfavorable_label=0)
    test_ds = BinaryLabelDataset(df=X_test.join(y_test),\
                    label_names=[label_name],\
                    protected_attribute_names=[protected_attribute_name],
                    favorable_label=1, unfavorable_label=0)
    protected_index = [train_ds.feature_names.index(m) for m in sup]
    X_train_sup = np.delete(train_ds.features, protected_index, axis=1)
    X_test_sup = np.delete(test_ds.features, protected_index, axis=1)    
    model_opt = model_func(X_train_sup, y_train,None) 
    cls_mdls['sup'+'_'+str(i)] = helper.evaluate_class_mdl(model_opt,\
            X_train_sup, X_test_sup,  y_train, y_test)
    test_pred_ds = test_ds.copy(deepcopy=True)
    test_pred_ds.labels = cls_mdls['sup'+'_'+str(i)]['preds_test'].reshape(-1,1)
    test_pred_ds.scores = cls_mdls['sup'+'_'+str(i)]['probs_test'].reshape(-1,1)
    y_pred = cls_mdls['sup'+'_'+str(i)]['preds_test'].reshape(-1,1)
    metrics_test_dict, metrics_test_cls = helper.compute_aif_metrics(test_ds, test_pred_ds,\
                          unprivileged_groups=unprivileged_groups,\
                          privileged_groups=privileged_groups)
    cls_mdls['sup'+'_'+str(i)].update(metrics_test_dict)
    for j,k in enumerate(protected_attribute_values):
        globals()[f'per_{k}'] = helper.performance_measures(X_test,y_test,y_pred,1,0,conditions[j],k)
        cls_mdls['sup'+'_'+str(i)].update(globals()[f'per_{k}'])
    return cls_mdls
