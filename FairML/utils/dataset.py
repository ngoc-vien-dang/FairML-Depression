"""
File name: dataset.py
Author: ngocviendang
Date created: October 26, 2022

This file contains functions for data transforms and training models.
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn.metrics import fbeta_score, make_scorer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

FOLDS = 10
f2_scorer = make_scorer(fbeta_score, beta=2)
######LONGSCAN#######
def model_preparation_longscan(X_train,X_test):
    imp_median = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    X_train_imputed = imp_median.fit_transform(X_train)
    X_test_imputed = imp_median.transform(X_test)
    X_train_imputed = pd.DataFrame(X_train_imputed)
    X_test_imputed = pd.DataFrame(X_test_imputed)
    X_train_imputed.columns = X_train.columns
    X_train_imputed.index = X_train.index
    X_test_imputed.columns = X_test.columns
    X_test_imputed.index = X_test.index
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # Selecting the columns to be one hot encoded
    ohe_cols = ['gender','race']
    # One hot encoding the categorical columns of the dataframes
    ohe_train = encoder.fit_transform(X_train_imputed[ohe_cols])
    ohe_test = encoder.transform(X_test_imputed[ohe_cols])
    # Getting the new names of the columns
    col_names = encoder.get_feature_names(ohe_cols)
    # Turning the encoded columns into dataframes
    ohe_train_df = pd.DataFrame(ohe_train, columns=col_names, index=X_train.index)
    ohe_test_df = pd.DataFrame(ohe_test, columns=col_names, index=X_test.index)
    # Listing the nummerical columns
    scale_cols = ['caregiver_depressive_symptoms_1','caregiver_depressive_symptoms_2',\
                'caregiver_depressive_symptoms_3']
    transformer = StandardScaler()
    # Fitting the transformer
    trans_train = transformer.fit_transform(X_train_imputed[scale_cols])
    trans_test = transformer.transform(X_test_imputed[scale_cols])
    # Turning the scaled data into dataframes
    trans_train_df = pd.DataFrame(trans_train, columns=X_train[scale_cols].columns,index=X_train_imputed.index)
    trans_test_df = pd.DataFrame(trans_test, columns=X_train[scale_cols].columns,index=X_test_imputed.index)
    ## Drop
    X_train_imputed.drop(['gender','race'], axis=1, inplace=True)
    X_test_imputed.drop(['gender','race'], axis=1, inplace=True)
    X_train_tf = pd.concat([ohe_train_df, trans_train_df,X_train_imputed], axis=1)
    X_test_tf = pd.concat([ohe_test_df, trans_test_df,X_test_imputed], axis=1)
    return X_train_tf, X_test_tf

# Optimal list of hyperparameters by performing nested cross-validation
longscan_lr_list = [{'C': [0.01], 'penalty': ['l2'], 'solver': ['liblinear']},\
{'C': [0.01], 'penalty': ['l2'], 'solver': ['liblinear']},\
{'C': [0.01], 'penalty': ['l2'], 'solver': ['liblinear']},\
{'C': [0.01], 'penalty': ['l2'], 'solver': ['liblinear']},\
{'C': [0.01], 'penalty': ['l2'], 'solver': ['liblinear']},\
{'C': [0.01], 'penalty': ['l2'], 'solver': ['liblinear']},\
{'C': [0.1], 'penalty': ['l2'], 'solver': ['liblinear']},\
{'C': [0.01], 'penalty': ['l2'], 'solver': ['liblinear']},\
{'C': [0.01], 'penalty': ['l2'], 'solver': ['liblinear']},\
{'C': [0.01], 'penalty': ['l2'], 'solver': ['liblinear']}]

def longscan_lr(X_train,y_train,sample_weight,i):
    model = LogisticRegression(class_weight='balanced',max_iter=1000)
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    # define grid search
    grid = longscan_lr_list[i]
    kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=9)
    grid_search = GridSearchCV(model, grid,  scoring=f2_scorer, n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # fitting to the training data,
    lr_opt = LogisticRegression(**grid_result.best_params_,random_state=9, max_iter=1000,class_weight='balanced')
    lr_opt = lr_opt.fit(X_train, y_train,sample_weight=sample_weight)
    return lr_opt
longscan_xgb_list = [{'subsample': [0.7], 'min_child_weight': [5], 'max_depth': [1], 'learning_rate': [0.3]},\
            {'subsample': [0.3], 'min_child_weight': [5], 'max_depth': [1], 'learning_rate': [0.1]},\
            {'subsample': [0.7], 'min_child_weight': [1], 'max_depth': [1], 'learning_rate': [0.3]},\
            {'subsample': [0.7], 'min_child_weight': [3], 'max_depth': [1], 'learning_rate': [0.2]},\
            {'subsample': [0.5], 'min_child_weight': [3], 'max_depth': [1], 'learning_rate': [0.2]},\
            {'subsample': [0.7], 'min_child_weight': [3], 'max_depth': [1], 'learning_rate': [0.1]},\
            {'subsample': [0.7], 'min_child_weight': [3], 'max_depth': [1], 'learning_rate': [0.1]},\
            {'subsample': [0.3], 'min_child_weight': [5], 'max_depth': [1], 'learning_rate': [0.1]},\
            {'subsample': [0.3], 'min_child_weight': [5], 'max_depth': [1], 'learning_rate': [0.2]},\
            {'subsample': [0.5], 'min_child_weight': [1], 'max_depth': [1], 'learning_rate': [0.1]}]

def longscan_xgb(X_train,y_train,sample_weight,i):
    counter = Counter(y_train)
    estimate = counter[0] / counter[1]
    #xgb_params = {'max_depth': [1,3,5], 
              #'learning_rate': [0.1,0.2,0.3], 
              #'min_child_weight': [1,3,5], 
              #'subsample': [.3,0.5,0.7]}
    xgb_params = longscan_xgb_list[i]
    # Creating the classifier
    xgb_clf = XGBClassifier(n_jobs=-1, random_state=123,scale_pos_weight=estimate)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=9)
    # Feeding the parameters into the grid for testing
    grid = RandomizedSearchCV(xgb_clf, xgb_params, scoring=f2_scorer, n_jobs=-1, cv=kfold)
    # Fitting to the training data
    xgb_grid = grid.fit(X_train, y_train)
    print("Best: %f using %s" % (xgb_grid.best_score_, xgb_grid.best_params_))
    xgb_opt = XGBClassifier(random_seed=9,**xgb_grid.best_params_,scale_pos_weight=estimate)
    xgb_opt  = xgb_opt.fit(X_train, y_train,sample_weight=sample_weight)
    return xgb_opt

######FUUS########
def cmToM(row):
    m = row / 100
    return m

def bmiCalc(height, weight):
    bmi = weight / (height ** 2)
    return bmi

def mapCalc(sys, dias):
    map = (1 / 3) * sys + (2 / 3) * dias
    return map

def model_preparation_fuus(X_train, X_test):
    imp_median = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp_median = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    X_train_imputed = imp_median.fit_transform(X_train)
    X_test_imputed = imp_median.transform(X_test)
    X_train_imputed = pd.DataFrame(X_train_imputed)
    X_test_imputed = pd.DataFrame(X_test_imputed)
    X_train_imputed.columns = X_train.columns
    X_train_imputed.index = X_train.index
    X_test_imputed.columns = X_test.columns
    X_test_imputed.index = X_test.index
    X_train_imputed['Height (cm)'] = cmToM(X_train_imputed['Height (cm)'])
    X_train_imputed.rename(columns={'Height (cm)': 'Height (m)'}, inplace=True)
    X_train_imputed['BMI'] = bmiCalc(X_train_imputed['Height (m)'], X_train_imputed['Weight (kg)'])
    X_train_imputed['pulse_pressure'] = X_train_imputed['Systolic blood pressure (mmHg)'] - X_train_imputed['Diastolic blood pressure (mmHg)']
    X_train_imputed['MAP'] = bmiCalc(X_train_imputed['Systolic blood pressure (mmHg)'],X_train_imputed['Diastolic blood pressure (mmHg)'])
  
    X_test_imputed['Height (cm)'] = cmToM(X_test_imputed['Height (cm)'])
    X_test_imputed.rename(columns={'Height (cm)': 'Height (m)'}, inplace=True)
    X_test_imputed['BMI'] = bmiCalc(X_test_imputed['Height (m)'], X_test_imputed['Weight (kg)'])
    X_test_imputed['pulse_pressure'] = X_test_imputed['Systolic blood pressure (mmHg)'] - X_test_imputed['Diastolic blood pressure (mmHg)']
    X_test_imputed['MAP'] = bmiCalc(X_test_imputed['Systolic blood pressure (mmHg)'],X_test_imputed['Diastolic blood pressure (mmHg)'])
    x_train = X_train_imputed
    x_test = X_test_imputed
    min_max_scaler = QuantileTransformer(random_state=123)
    x_train_scaled = min_max_scaler.fit_transform(x_train)
    x_test_scaled = min_max_scaler.transform(x_test)
    X_train_imputed_norm = pd.DataFrame(x_train_scaled)
    X_test_imputed_norm = pd.DataFrame(x_test_scaled)
    X_train_imputed_norm.columns = x_train.columns
    X_train_imputed_norm.index = x_train.index
    X_test_imputed_norm.columns = x_test.columns
    X_test_imputed_norm.index = x_test.index
    return X_train_imputed_norm, X_test_imputed_norm
fuus_lr_list = [{'C': [0.1], 'penalty': ['l2'], 'solver': ['newton-cg']},\
{'C': [0.1], 'penalty': ['l2'], 'solver': ['liblinear']},\
{'C': [100], 'penalty': ['l2'], 'solver': ['newton-cg']},\
{'C': [0.1], 'penalty': ['l2'], 'solver': ['newton-cg']},\
{'C': [0.1], 'penalty': ['l2'], 'solver': ['liblinear']},\
{'C': [0.01], 'penalty': ['l2'], 'solver': ['liblinear']},\
{'C': [0.01], 'penalty': ['l2'], 'solver': ['newton-cg']},\
{'C': [0.1], 'penalty': ['l2'], 'solver': ['lbfgs']},\
{'C': [100], 'penalty': ['l2'], 'solver': ['newton-cg']},\
{'C': [10], 'penalty': ['l2'], 'solver': ['newton-cg']}]
def fuus_lr(X_train, y_train,sample_weight,i):
    # define models and parameters
    model = LogisticRegression(class_weight='balanced',max_iter=1000)
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    # define grid search
    grid = fuus_lr_list[i]
    kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=9)
    grid_search = GridSearchCV(model, grid,  scoring=f2_scorer, n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # fitting to the training data,
    lr_opt = LogisticRegression(**grid_result.best_params_,random_state=9, max_iter=1000,class_weight='balanced')
    lr_opt = lr_opt.fit(X_train, y_train,sample_weight=sample_weight)
    return lr_opt
fuus_xgb_list = [{'learning_rate': [0.2], 'max_depth': [1], 'min_child_weight': [1], 'subsample': [0.3]},
{'learning_rate': [0.3], 'max_depth': [1], 'min_child_weight': [5], 'subsample': [0.5]},
{'learning_rate': [0.3], 'max_depth': [1], 'min_child_weight': [5], 'subsample': [0.3]},
{'learning_rate': [0.3], 'max_depth': [1], 'min_child_weight': [3], 'subsample': [0.3]},
{'learning_rate': [0.3], 'max_depth': [1], 'min_child_weight': [1], 'subsample': [0.5]},
{'learning_rate': [0.3], 'max_depth': [1], 'min_child_weight': [5], 'subsample': [0.5]},
{'learning_rate': [0.3], 'max_depth': [1], 'min_child_weight': [1], 'subsample': [0.3]},
{'learning_rate': [0.3], 'max_depth': [1], 'min_child_weight': [3], 'subsample': [0.7]},
{'learning_rate': [0.3], 'max_depth': [1], 'min_child_weight': [3], 'subsample': [0.3]},
{'learning_rate': [0.3], 'max_depth': [1], 'min_child_weight': [1], 'subsample': [0.3]}]
def fuus_xgb(X_train, y_train,sample_weight,i):
    counter = Counter(y_train)
    estimate = counter[0] / counter[1]
    #xgb_params = {'max_depth': [1,3,5], 
              #'learning_rate': [0.1,0.2,0.3], 
              #'min_child_weight': [1,3,5], 
              #'subsample': [.3,0.5,0.7]}
    xgb_params = fuus_xgb_list[i]
    # Creating the classifier
    xgb_clf = XGBClassifier(n_jobs=-1, random_state=123,scale_pos_weight=estimate)
    # Feeding the parameters into the grid for testing
    grid = GridSearchCV(xgb_clf, xgb_params, scoring=f2_scorer, n_jobs=-1, cv=3)
    # Fitting to the training data
    xgb_grid = grid.fit(X_train, y_train)
    print("Best: %f using %s" % (xgb_grid.best_score_, xgb_grid.best_params_))
    xgb_opt = XGBClassifier(random_seed=9,**xgb_grid.best_params_,scale_pos_weight=estimate)
    xgb_opt = xgb_opt.fit(X_train, y_train,sample_weight=sample_weight)
    return xgb_opt

#######CDC-NHANES########
def model_preparation_nhanes(X_train,X_test):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # Selecting the columns to be one hot encoded
    ohe_cols = X_train.select_dtypes('O').columns
    # One hot encoding the categorical columns of the dataframes
    ohe_train = encoder.fit_transform(X_train[ohe_cols])
    ohe_test = encoder.transform(X_test[ohe_cols])
    # Getting the new names of the columns
    col_names = encoder.get_feature_names(ohe_cols)
    # Turning the encoded columns into dataframes
    ohe_train_df = pd.DataFrame(ohe_train, columns=col_names, index=X_train.index)
    ohe_test_df = pd.DataFrame(ohe_test, columns=col_names, index=X_test.index)
    # Listing the nummerical columns
    scale_cols = X_train.select_dtypes('number').columns
    transformer = QuantileTransformer(random_state=123)
    #transformer = StandardScaler()
    # Fitting the transformer
    trans_train = transformer.fit_transform(X_train[scale_cols])
    trans_test = transformer.transform(X_test[scale_cols])
    # Turning the scaled data into dataframes
    trans_train_df = pd.DataFrame(trans_train, columns=X_train[scale_cols].columns,index=X_train.index)
    trans_test_df = pd.DataFrame(trans_test, columns=X_train[scale_cols].columns,index=X_test.index)
    # Combining the one hot encoded and scaled data back together
    X_train_tf = pd.concat([ohe_train_df, trans_train_df], axis=1)
    X_test_tf = pd.concat([ohe_test_df, trans_test_df], axis=1)
    return X_train_tf, X_test_tf

nhanes_lr_list = [{'C': [0.01], 'penalty': ['l2'], 'solver': ['newton-cg']},\
            {'C': [0.01], 'penalty': ['l2'], 'solver': ['lbfgs']},\
            {'C': [0.01], 'penalty': ['l2'], 'solver': ['liblinear']},\
            {'C': [0.01], 'penalty': ['l2'], 'solver': ['liblinear']},\
            {'C': [0.01], 'penalty': ['l2'], 'solver': ['liblinear']},\
            {'C': [0.01], 'penalty': ['l2'], 'solver': ['lbfgs']},\
            {'C': [0.01], 'penalty': ['l2'], 'solver': ['newton-cg']},\
            {'C': [0.01], 'penalty': ['l2'], 'solver': ['liblinear']},\
            {'C': [0.01], 'penalty': ['l2'], 'solver': ['newton-cg']},\
            {'C': [0.01], 'penalty': ['l2'], 'solver': ['liblinear']}]

def nhanes_lr(X_train,y_train,sample_weight,i):
    model = LogisticRegression(class_weight='balanced',max_iter=1000)
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    # define grid search
    grid = nhanes_lr_list[i]
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=9)
    grid_search = GridSearchCV(model, grid,  scoring=f2_scorer, n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # fitting to the training data,
    lr_opt = LogisticRegression(**grid_result.best_params_,random_state=9, max_iter=1000,class_weight='balanced')
    lr_opt = lr_opt.fit(X_train, y_train,sample_weight=sample_weight)
    return lr_opt

nhanes_xgb_list = [{'learning_rate': [0.15], 'max_depth': [3], 'min_child_weight': [8], 'subsample': [0.7]},\
            {'learning_rate': [0.25], 'max_depth': [1], 'min_child_weight': [8], 'subsample': [0.7]},\
            {'learning_rate': [0.05], 'max_depth': [5], 'min_child_weight': [12], 'subsample': [0.5]},\
            {'learning_rate': [0.25], 'max_depth': [1], 'min_child_weight': [8], 'subsample': [0.7]},\
            {'learning_rate': [0.05], 'max_depth': [3], 'min_child_weight': [8], 'subsample': [0.7]},\
            {'learning_rate': [0.25], 'max_depth': [1], 'min_child_weight': [10], 'subsample': [0.5]},\
            {'learning_rate': [0.15], 'max_depth': [3], 'min_child_weight': [12], 'subsample': [0.5]},\
            {'learning_rate': [0.25], 'max_depth': [1], 'min_child_weight': [10], 'subsample': [0.5]},\
            {'learning_rate': [0.25], 'max_depth': [1], 'min_child_weight': [12], 'subsample': [0.3]},\
            {'learning_rate': [0.15], 'max_depth': [3], 'min_child_weight': [12], 'subsample': [0.7]}]

def nhanes_xgb(X_train,y_train,sample_weight,i):
    counter = Counter(y_train)
    estimate = counter[0] / counter[1]
    xgb_params = nhanes_xgb_list[i]
    # Creating the classifier
    xgb_clf = XGBClassifier(n_jobs=-1, random_state=123,scale_pos_weight=estimate)
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=9)
    # Feeding the parameters into the grid for testing
    grid = GridSearchCV(xgb_clf, xgb_params, scoring=f2_scorer, n_jobs=-1, cv=kfold)
    # Fitting to the training data
    xgb_grid = grid.fit(X_train, y_train)
    print("Best: %f using %s" % (xgb_grid.best_score_, xgb_grid.best_params_))
    xgb_opt = XGBClassifier(random_seed=9,**xgb_grid.best_params_,scale_pos_weight=estimate)
    xgb_opt = xgb_opt.fit(X_train, y_train,sample_weight=sample_weight)
    return xgb_opt
    
#######UKBB########
def model_preparation_ukbb(X_train,X_test,var_encode,continous_var,keep_var):
    # Selecting the columns to be one hot encoded
    ohe_cols = var_encode
    imputer_encode = SimpleImputer(strategy= 'most_frequent')
    X_train_encode = imputer_encode.fit_transform(X_train[ohe_cols])
    X_test_encode = imputer_encode.transform(X_test[ohe_cols])
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # One hot encoding the categorical columns of the dataframes
    ohe_train = encoder.fit_transform(X_train_encode)
    ohe_test = encoder.transform(X_test_encode)
    # Getting the new names of the columns
    col_names = encoder.get_feature_names(ohe_cols)
    # Turning the encoded columns into dataframes
    ohe_train_df = pd.DataFrame(ohe_train, columns=col_names, index=X_train.index)
    ohe_test_df = pd.DataFrame(ohe_test, columns=col_names, index=X_test.index)
    # Listing the nummerical columns
    scale_cols = continous_var
    imputer_continous = SimpleImputer(strategy= 'most_frequent')
    X_train_continous = imputer_encode.fit_transform(X_train[scale_cols])
    X_test_continous = imputer_encode.transform(X_test[scale_cols])
    transformer = StandardScaler()
    # Fitting the transformer
    trans_train = transformer.fit_transform(X_train_continous)
    trans_test = transformer.transform(X_test_continous)
    # Turning the scaled data into dataframes
    trans_train_df = pd.DataFrame(trans_train, columns=continous_var,index=X_train.index)
    trans_test_df = pd.DataFrame(trans_test, columns=continous_var,index=X_test.index)
    # Combining the one hot encoded and scaled data back together
    ###
    imputer_keep = SimpleImputer(strategy= 'most_frequent')
    X_train_keep = imputer_encode.fit_transform(X_train[keep_var])
    X_test_keep = imputer_encode.transform(X_test[keep_var])
    keep_train_df = pd.DataFrame(X_train_keep, columns=keep_var, index=X_train.index)
    keep_test_df = pd.DataFrame(X_test_keep, columns=keep_var, index=X_test.index)
    X_train_tf = pd.concat([ohe_train_df, trans_train_df,keep_train_df], axis=1)
    X_test_tf = pd.concat([ohe_test_df, trans_test_df,keep_test_df], axis=1)
    return X_train_tf, X_test_tf

def ukb_lr(X_train,y_train,sample_weight):
    lr_opt = LogisticRegression(random_state=123,class_weight='balanced',max_iter=5000)
    lr_opt = lr_opt.fit(X_train, y_train,sample_weight=sample_weight)
    return lr_opt

def ukb_xgb(X_train,y_train,sample_weight):
    counter = Counter(y_train)
    estimate = counter[0] / counter[1]
    xgb_opt = XGBClassifier(n_jobs=-1, random_state=123,scale_pos_weight=estimate)
    xgb_opt = xgb_opt.fit(X_train, y_train,sample_weight=sample_weight)
    return xgb_opt