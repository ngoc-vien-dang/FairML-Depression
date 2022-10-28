"""
File name: cpp.py
Author: ngocviendang
Date created: October 26, 2022

This file contains functions for addressing algorithmic bias (cpp).
"""
import numpy as np
from FairML.utils import helper
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
from joblib import dump, load

def cpp(X_train,y_train,X_test,y_test,cls_mdls,cls_cpp,model_path,unprivileged_groups,privileged_groups,\
              conditions,label_name,protected_attribute_name,protected_attribute_values,i,sup=[]):
    train_ds = BinaryLabelDataset(df=X_train.join(y_train),\
                    label_names=[label_name],\
                    protected_attribute_names=[protected_attribute_name],
                    favorable_label=1, unfavorable_label=0)
    test_ds = BinaryLabelDataset(df=X_test.join(y_test),\
                    label_names=[label_name],\
                    protected_attribute_names=[protected_attribute_name],
                    favorable_label=1, unfavorable_label=0)
    # Load model
    model_opt = load(model_path)
    if sup:
        protected_index = [train_ds.feature_names.index(m) for m in sup]
        X_train_sup = np.delete(train_ds.features, protected_index, axis=1)
        X_test_sup = np.delete(test_ds.features, protected_index, axis=1)
        cls_mdls['benchmark'+'_'+str(i)] = helper.evaluate_class_mdl(model_opt,\
            X_train_sup, X_test_sup,  y_train, y_test)
    else:
        cls_mdls['benchmark'+'_'+str(i)] = helper.evaluate_class_mdl(model_opt,\
            X_train, X_test, y_train, y_test)
    test_pred_ds = test_ds.copy(deepcopy=True)
    test_pred_ds.labels = cls_mdls['benchmark'+'_'+str(i)]['preds_test'].reshape(-1,1)
    test_pred_ds.scores = cls_mdls['benchmark'+'_'+str(i)]['probs_test'].reshape(-1,1)
    ###
    cpp = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups,\
                                     unprivileged_groups=unprivileged_groups,\
                                     cost_constraint="fnr",
                                     seed=9)
    cpp = cpp.fit(test_ds, test_pred_ds)
    test_pred_cpp_ds = cpp.predict(test_pred_ds)
    cls_cpp['cpp'+'_'+str(i)] =\
    helper.evaluate_class_metrics_mdl(test_pred_cpp_ds.scores, test_pred_cpp_ds.labels, y_test)

    y_pred = cls_cpp['cpp'+'_'+str(i)]['preds_test'].reshape(-1,1)

    metrics_test_cpp_dict, _ = helper.compute_aif_metrics(test_ds, test_pred_cpp_ds,\
                          unprivileged_groups=unprivileged_groups,\
                          privileged_groups=privileged_groups)
    cls_cpp['cpp'+'_'+str(i)].update(metrics_test_cpp_dict)

    for j,k in enumerate(protected_attribute_values):
        globals()[f'per_{k}'] = helper.performance_measures(X_test,y_test,y_pred,1,0,conditions[j],k)
        cls_cpp['cpp'+'_'+str(i)].update(globals()[f'per_{k}'])
    return cls_mdls,cls_cpp
