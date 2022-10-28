"""
File name: dir.py
Author: ngocviendang
Date created: October 26, 2022

This file contains functions for addressing algorithmic bias (dir).
"""
import numpy as np
from tqdm.notebook import tqdm
from FairML.utils import helper
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import DisparateImpactRemover
from joblib import dump, load
from FairML.utils import dataset

def dir_(X_train,y_train,X_test,y_test,cls_mdls,unprivileged_groups,privileged_groups,\
              conditions,label_name,protected_attribute_name,protected_attribute_values,i,model_arch,sup):
    train_ds = BinaryLabelDataset(df=X_train.join(y_train),\
                    label_names=[label_name],\
                    protected_attribute_names=[protected_attribute_name],
                    favorable_label=1, unfavorable_label=0)
    test_ds = BinaryLabelDataset(df=X_test.join(y_test),\
                    label_names=[label_name],\
                    protected_attribute_names=[protected_attribute_name],
                    favorable_label=1, unfavorable_label=0)
    model_dict = {'ls-lr': dataset.longscan_lr,\
              'ls-xgb': dataset.longscan_xgb,\
              'fu-lr': dataset.fuus_lr,\
              'fu-xgb': dataset.fuus_xgb,\
              'nh-lr': dataset.nhanes_lr,\
              'nh-xgb': dataset.nhanes_xgb,\
              'uk-lr': dataset.ukb_lr,\
              'uk-xgb': dataset.ukb_xgb}
    di = np.array([])
    train_dir_ds = None
    test_dir_ds = None
    lr_dir_mdl = None
    X_train_dir = None
    X_test_dir = None
    levels = np.hstack([np.linspace(0., 0.1, 9), np.linspace(0.2, 1, 9)])
    for level in tqdm(levels):
        di_remover = DisparateImpactRemover(repair_level=level)
        train_dir_ds_i = di_remover.fit_transform(train_ds)
        test_dir_ds_i = di_remover.fit_transform(test_ds)
        
        protected_index = [train_ds.feature_names.index(m) for m in sup]
        X_train_dir_i = np.delete(train_dir_ds_i.features, protected_index, axis=1)
        X_test_dir_i = np.delete(test_dir_ds_i.features, protected_index, axis=1)
        if 'uk' in model_arch:
            lr_dir_mdl_i = model_dict[model_arch](X_train_dir_i, y_train,None) 
        else:
            lr_dir_mdl_i = model_dict[model_arch](X_train_dir_i, y_train,None,i)
        test_dir_ds_pred_i = test_dir_ds_i.copy()
        test_dir_ds_pred_i.labels = lr_dir_mdl_i.predict(X_test_dir_i)
        metrics_test_dir_ds = BinaryLabelDatasetMetric(test_dir_ds_pred_i,\
                                   unprivileged_groups=unprivileged_groups,\
                                   privileged_groups=privileged_groups)
        di_i = metrics_test_dir_ds.disparate_impact()
        if (di.shape[0]==0) or (np.min(np.abs(di-1)) >= abs(di_i-1)):
            print(abs(di_i-1))
            train_dir_ds = train_dir_ds_i
            test_dir_ds = test_dir_ds_i
            X_train_dir = X_train_dir_i
            X_test_dir = X_test_dir_i
            lr_dir_mdl = lr_dir_mdl_i
        di = np.append(np.array(di), di_i)
    cls_mdls['dir'+'_'+str(i)] = helper.evaluate_class_mdl(lr_dir_mdl,\
            X_train_dir, X_test_dir, train_dir_ds.labels, test_dir_ds.labels)    
  
    test_pred_dir_ds = test_ds.copy(deepcopy=True)
    test_pred_dir_ds.labels = cls_mdls['dir'+'_'+str(i)]['preds_test'].reshape(-1,1)
    metrics_test_dir_dict, _ =\
        helper.compute_aif_metrics(test_ds, test_pred_dir_ds,\
                          unprivileged_groups=unprivileged_groups,\
                          privileged_groups=privileged_groups)
    cls_mdls['dir'+'_'+str(i)].update(metrics_test_dir_dict)
    #
    y_pred = cls_mdls['dir'+'_'+str(i)]['preds_test'].reshape(-1,1)
    for j,k in enumerate(protected_attribute_values):
        globals()[f'per_{k}'] = helper.performance_measures(X_test,y_test,y_pred,1,0,conditions[j],k)
        cls_mdls['dir'+'_'+str(i)].update(globals()[f'per_{k}'])
    return cls_mdls
