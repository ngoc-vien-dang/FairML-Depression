"""
File name: st.py
Author: ngocviendang
Date created: October 26, 2022

This file contains functions for addressing algorithmic bias (psta).
"""
import numpy as np
import pandas as pd
from FairML.utils import helper
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from joblib import dump, load

def psta(X_train,y_train,X_test,y_test,cls_mdls,cls_3st,model_path,unprivileged_groups,privileged_groups,\
              conditions,label_name,protected_attribute_name,protected_attribute_values,i,\
              unprivileged_value_3st,sup=[]):
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
    ####
    if sup:
        data = {unprivileged_value_3st[0]: X_test[unprivileged_value_3st[0]],\
                'threshold': 0.5*np.ones(X_test.shape[0]),'prob': model_opt.predict_proba(X_test_sup)[:,1]}
        df_y_test = pd.DataFrame(data)
        for l in unprivileged_value_3st[1:]:
            df_y_test.loc[X_test[unprivileged_value_3st[0]] == l,'threshold'] = helper.get_threshold_sen(model_opt,\
            X_train_sup,y_train,X_train_sup[X_train[unprivileged_value_3st[0]]==l],\
            y_train[X_train[unprivileged_value_3st[0]]==l])
    else:
        data = {unprivileged_value_3st[0]: X_test[unprivileged_value_3st[0]],\
                'threshold': 0.5*np.ones(X_test.shape[0]),'prob': model_opt.predict_proba(X_test)[:,1]}
        df_y_test = pd.DataFrame(data)
        for l in unprivileged_value_3st[1:]:
            df_y_test.loc[X_test[unprivileged_value_3st[0]] == l,'threshold'] = helper.get_threshold_sen(model_opt,\
            X_train,y_train,X_train[X_train[unprivileged_value_3st[0]]==l],\
            y_train[X_train[unprivileged_value_3st[0]]==l])
    df_y_test['pred'] = (df_y_test['prob'] >= df_y_test['threshold']).astype('int')
    cls_3st['3st'+'_'+str(i)] =\
    helper.evaluate_class_metrics_mdl(test_pred_ds.scores, df_y_test['pred'], y_test)
    #
    test_pred_3st_ds = test_ds.copy(deepcopy=True)
    test_pred_3st_ds.labels = cls_3st['3st'+'_'+str(i)]['preds_test'].values.reshape(-1,1)
    test_pred_3st_ds.scores = cls_3st['3st'+'_'+str(i)]['probs_test'].reshape(-1,1)
    y_pred = cls_3st['3st'+'_'+str(i)]['preds_test'].values.reshape(-1,1)
    #
    metrics_test_3st_dict, _ =\
        helper.compute_aif_metrics(test_ds, test_pred_3st_ds,\
                          unprivileged_groups=unprivileged_groups,\
                          privileged_groups=privileged_groups)
    cls_3st['3st'+'_'+str(i)].update(metrics_test_3st_dict)
    for j,k in enumerate(protected_attribute_values):
        globals()[f'per_{k}'] = helper.performance_measures(X_test,y_test,y_pred,1,0,conditions[j],k)
        cls_3st['3st'+'_'+str(i)].update(globals()[f'per_{k}'])
    return cls_mdls,cls_3st
