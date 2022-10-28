"""
File name: helper.py
Author: ngocviendang
Date created: October 26, 2022

This file contains helper functions for other scripts.
"""
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from sklearn.preprocessing import QuantileTransformer, StandardScaler, OneHotEncoder

def get_threshold_sen(fitted_model,X_train,y_train,X_g,y_g):
    y_train_pred = fitted_model.predict(X_train)
    sen = metrics.recall_score(y_train,y_train_pred,zero_division=0)
    yhat = fitted_model.predict_proba(X_g)
    probs = yhat[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_g,probs)
    sen_list = np.asarray([sen]*recall.shape[0]) 
    dis = np.abs(recall - sen_list)
    ix_list = np.where(dis == dis.min())
    return thresholds[ix_list].mean()

def evaluate_class_mdl(fitted_model, X_train, X_test, y_train, y_test,predopts={},threshold = 0.5):
    eval_dict = {}
    y_train_pred = fitted_model.predict(X_train, **predopts).squeeze()
    if len(np.unique(y_train_pred)) > 2:
        y_test_prob = fitted_model.predict(X_test, **predopts).squeeze()
        y_test_pred = np.where(y_test_prob > threshold, 1, 0)
    else:   
        y_test_prob = fitted_model.predict_proba(X_test, **predopts)[:,1]
        y_test_pred = np.where(y_test_prob > threshold, 1, 0)
    return evaluate_class_metrics_mdl(y_test_prob, y_test_pred, y_test)

def evaluate_class_metrics_mdl(y_test_prob, y_test_pred, y_test):      
    eval_dict = {}
    eval_dict['probs_test'] = y_test_prob
    eval_dict['preds_test'] = y_test_pred
    eval_dict['roc-auc_test'] = metrics.roc_auc_score(y_test, y_test_prob)
    eval_dict['bacc_test'] = metrics.balanced_accuracy_score(y_test, y_test_pred)
    return eval_dict
def compute_aif_metrics(dataset_true, dataset_pred, unprivileged_groups, privileged_groups):

    metrics_cls = ClassificationMetric(dataset_true, dataset_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    metrics_dict = {}
    metrics_dict["EOD"] = metrics_cls.equal_opportunity_difference()
    return metrics_dict, metrics_cls

def compute_boolean_conditioning_vector(X,condition=None):
    feature_names = X.columns.tolist()
    X = X.to_numpy()
    overall_cond = np.zeros(X.shape[0], dtype=bool)
    for group in condition:
        group_cond = np.ones(X.shape[0], dtype=bool)
        for name, val in group.items():
            index = feature_names.index(name)
            group_cond = np.logical_and(group_cond, X[:, index] == val)
        overall_cond = np.logical_or(overall_cond, group_cond)
    return overall_cond

def performance_measures(X,y_true,y_pred,positive_label,negative_label,condition,group):
    class_rate_dict = {}
    w = np.ones(X.shape[0], dtype=int)
    cond_vec = compute_boolean_conditioning_vector(X,condition)
    # to prevent broadcasts
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    y_true_pos = (y_true == positive_label)
    y_true_neg = (y_true == negative_label)
    y_pred_pos = np.logical_and(y_pred == positive_label, cond_vec)
    y_pred_neg = np.logical_and(y_pred == negative_label, cond_vec)

    # True/false positives/negatives
    TP = np.sum(w[np.logical_and(y_true_pos, y_pred_pos)], dtype=np.float64)
    FP = np.sum(w[np.logical_and(y_true_neg, y_pred_pos)], dtype=np.float64)
    TN = np.sum(w[np.logical_and(y_true_neg, y_pred_neg)], dtype=np.float64)
    FN = np.sum(w[np.logical_and(y_true_pos, y_pred_neg)], dtype=np.float64)
    P=TP + FN; N=TN + FP
    class_rate_dict['tpr'+'_'+ group]= TP / P
    class_rate_dict['tnr'+'_'+ group]= TN / N
    class_rate_dict['fpr'+'_'+ group]= FP / N
    class_rate_dict['fnr'+'_'+ group]= FN / P
    return class_rate_dict