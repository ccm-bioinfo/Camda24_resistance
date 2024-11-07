#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:52:14 2023

@author: victor
"""

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.model_selection import KFold

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder
# Hacer compatible la red con sckitlearn
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.optimizers import SGD

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras import activations
import random 
from keras.utils import to_categorical
'''
Representacion de los datos basado en conteos de genes
'''

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD, NMF
import pickle

def get_tfidf(data):
    vectorizer = TfidfTransformer()
    tfidf_transformer = vectorizer.fit(data)
    X = tfidf_transformer.transform(data)
    tfidf = X.toarray()
    
    return  tfidf_transformer, tfidf

# realiza factorizacion SVD o NMF
def get_factorization(data, n_comp=100, nmf=True):
    if nmf==False:
        fact_model = TruncatedSVD(n_components=n_comp)
        fact_model.fit(data)
    else:
        fact_model = NMF(n_components = n_comp, init=None, max_iter=12000)
        fact_model.fit(data)
    
    return fact_model

def save_pickle_model(model_obj, file_path, model_name):    
    pkl_name = os.path.join(file_path, model_name)
    with open(pkl_name,'wb') as file:
        pickle.dump(model_obj,file)

def load_from_pickle(file_path, model_name):
    pkl_name = os.path.join(file_path, model_name)
    with open(pkl_name,'rb') as file:
        fact_model = pickle.load(file)
    
    return fact_model


'''
Reducción y selección de características.
Se probarán diferentes métodos para seleccionar características relevantes. 
Los criterios de relevancia son diferentes para cada método.
Se usarán todos los datos como entrenamiento.
'''
def get_reduced_df(df, varnames, importances_df):
    var_to_drop = [elem for idx, elem in enumerate(varnames) if elem not in importances_df.index]
    reduced_df = df.drop(var_to_drop, axis=1)
    return reduced_df

# selección de características con random forests
# El criterio para la selección de características, es la reducción del índice de impureza de Gini
def rf_features(X_df, y, num_features = 100, n_trees=500, min_samples_split=3):
    rf = RandomForestClassifier(n_estimators=n_trees, min_samples_split=3, n_jobs=-1, random_state=42)
    rf.fit(X_df,y)
    varnames = X_df.columns.tolist()
    importances = rf.feature_importances_
    indices = np.flip(np.argsort(importances))[:num_features]
    fnames = [varnames[i] for i in indices]
    rf_importances = pd.Series(importances[indices], index=fnames)
    
    return rf_importances
    
    
    
# Recursive feature elimination
# En RFE, las variables "importantes" son las que quedan después de elimnar las "menos importantes". 
# El criterio para eliminar un conjunto de variables en cada recursión, 
# son los pesos del modelo base (estimator). En éste caso, se usa una SVM con un kernel lineal, 
# para mayor facilidad y porque es en éste caso, la solución está definida en el espacio original 
# de las variables. No estoy seguro si RFE sea válido (o cómo se haga) al usar un kernel no lineal. 
# Es necesario checarlo.... También puede usarse algún otro estimator.

def rfe_features(X_df, y, num_features = 100, steps = 50):
    svc = SVC(kernel="linear", C=1)
    #clf = LogisticRegression()
    rfe = RFE(estimator=svc, step=steps, n_features_to_select=num_features, verbose=False)
    rfe.fit(X_df, y)
    varnames = X_df.columns.tolist()
    rfe_indices = rfe.get_support(indices=True)
    fnames = [varnames[i] for i in rfe_indices]
    rfe_importances = pd.Series(rfe.ranking_[rfe_indices], index=fnames)
    
    return rfe_importances


def split_stratified_into_train_val_test(X, y, frac_train=0.6, frac_val=0.15, frac_test=0.25, std = True, 
                                         two_subsets=False, random_state=None):
    '''
    Splits a dataset into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in y (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    X : numpy dataframe of covariates
    y : numpy array of responses
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''
    
    if round(frac_train + frac_val + frac_test,10) != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    # Split original dataframe into temp and test dataframes.
    #x_train, x_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=(1.0 - frac_train), random_state=random_state)
    x_temp, x_test, y_temp, y_test = train_test_split(X, y, stratify=y, test_size=(1.0 - (frac_train+frac_val)), random_state=random_state)
    scaler = None
    if std:
        # standardize train_val (temp) and test data
        scaler = StandardScaler()
        x_temp = scaler.fit_transform(x_temp)
        x_test = scaler.transform(x_test)
        
    # weights for class imbalance (https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)
    class_w = compute_class_weight('balanced',classes=np.unique(y_temp),y=y_temp)
    # the latter is equivalent to:
    # unique, class_counts = np.unique(y_temp, return_counts=True)
    # class_w = sum(class_counts)/(len(unique)*class_counts)    
    if two_subsets:        
        x_train = x_temp
        y_train = y_temp
        x_val = None
        y_val = None
        #return x_train, y_train, x_test, y_test, class_w, scaler
    else:
        # Split the temp dataframe into train and val dataframes.
        relative_frac_val = frac_val / (frac_train + frac_val)
        x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, stratify=y_temp, 
                                                          test_size=relative_frac_val, random_state=random_state)
        #assert len(df_input) == len(df_train) + len(df_val) + len(df_test)
    
    return x_train, y_train, x_val, y_val, x_test, y_test, class_w, scaler


def get_X_y_ordinal(data_path, X_file, Y_file, data_label, only_counts = False, exist_fact = False, fact_path = os.getcwd(), n_comp = 100):
    ruta_archivo = data_path + X_file
    temp_data = pd.read_table(ruta_archivo, delimiter="\t", index_col=0, compression='infer')
    ruta_archivo = data_path + Y_file
    Y = pd.read_csv(ruta_archivo)

    # metadatos y variable ordinal y
    yy = Y[data_label].loc[Y[data_label].notnull()]

    # datos de train y test
    tt = ['test','train']
    tr_te = temp_data['phenotype'].notnull().astype(int)
    lab_trte = [tt[i] for i in tr_te]

    # metadatos
    metadata = temp_data[["genus","species","phenotype","mic"]]
    # MIC recategorizado, es decir, se juntan algunos valores, sobre todo de concentraciones pequeñas
    metadata['mic_recat'] = metadata['mic']
    metadata['mic_recat'].loc[metadata['mic_recat'].notnull()] = list(yy)
    metadata['str_mic'] = metadata['mic']
    metadata['str_mic'].loc[metadata['mic_recat'].notnull()] = [str(val) for val in yy]
    metadata['type'] = lab_trte
    metadata['gen_spec'] = metadata[['genus','species']].agg('_'.join,axis=1)
    metadata_tr = metadata[metadata['type']=='train']
    metadata_te = metadata[metadata['type']=='test']

    # categorización de la respuesta
    y = metadata_tr['str_mic']
    #y = metadata_tr['mic_recat'].astype('category')
    le = LabelEncoder()
    le.fit(y)
    #le.classes_
    y_cat = le.transform(y)

    # X
    names_to_drop = ["genus","species","phenotype","mic"]
    amr_count_train = temp_data[temp_data['phenotype'].notnull()].drop(names_to_drop, axis=1)
    amr_count_test = temp_data[temp_data['phenotype'].isnull()].drop(names_to_drop, axis=1)
    # realiza TF-IDF y factorizacion
    if only_counts is False:
        tfidf_vect, tfidf = get_tfidf(amr_count_train)
        tfidf_train = pd.DataFrame(tfidf,columns=tfidf_vect.get_feature_names_out(),index=amr_count_train.index)
        # si no hay un modelo de factorización guardado, obtiene uno y lo guarda
        if exist_fact is False:
            svd_fact = get_factorization(data=tfidf_train, n_comp=n_comp, nmf=False)
            nmf_fact = get_factorization(data=tfidf_train, n_comp=n_comp, nmf=True)
            pkl_file = 'nmf_'+ data_label + '.pkl'
            save_pickle_model(nmf_fact, fact_path, pkl_file)
            pkl_file = 'svd_' + data_label + '.pkl'
            save_pickle_model(svd_fact, fact_path, pkl_file)
        else: 
            # lsa
            pkl_file = 'svd_' + data_label + '.pkl'
            svd_fact = load_from_pickle(fact_path, pkl_file)
            # nmf
            pkl_file = 'nmf_' + data_label + '.pkl'
            nmf_fact = load_from_pickle(fact_path, pkl_file)

        amr_train_lsa = svd_fact.transform(tfidf_train)
        amr_train_nmf = nmf_fact.transform(tfidf_train)
        amr_train_lsa = pd.DataFrame(amr_train_lsa,index=amr_count_train.index)
        amr_train_nmf = pd.DataFrame(amr_train_nmf,index=amr_count_train.index)

        tfidf_test  = tfidf_vect.transform(amr_count_test).toarray()
        tfidf_test = pd.DataFrame(tfidf_test,columns=tfidf_vect.get_feature_names_out(),index=amr_count_test.index)
        amr_test_lsa = svd_fact.transform(tfidf_test)
        amr_test_nmf = nmf_fact.transform(tfidf_test)
        amr_test_lsa = pd.DataFrame(amr_test_lsa,index=amr_count_test.index)
        amr_test_nmf = pd.DataFrame(amr_test_nmf,index=amr_count_test.index)
        
        return metadata, metadata_tr, metadata_te, tfidf_train, tfidf_test, tfidf_vect, amr_train_lsa, amr_train_nmf, amr_test_lsa, amr_test_nmf, y, y_cat, le
    else:
        return metadata, metadata_tr, metadata_te, amr_count_train, amr_count_test, y, y_cat, le


# obtiene parámetros óptimos del clasificador ordinal con grid search y CV
def search_params_ordinal(ordinal_cl, X, y, param_grid, cv=5, scoring = 'accuracy'):
    # Realizar Grid Search y Cross Validation
    grid_search = GridSearchCV(ordinal_cl,
                            param_grid = param_grid,
                            cv = cv,
                            scoring = 'accuracy',
                            n_jobs = -1
                            )

    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_params, best_score

# obtiene estadísticas del desempeño del clasificador ordinal con validación cruzada
def ordinal_clas_stat(ordinal_cl, X, y, score, rand_state = None, n_repeats=5, n_split = 5):
    # Realizar la validación cruzada repetida
    cv_scores = []
    for _ in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=rand_state)
        scores = cross_val_score(ordinal_cl, X, y, cv=cv, scoring=score)
        cv_scores.extend(-scores)

    # Calcular media y desviación estándar de los puntajes
    mean_score = sum(cv_scores) / len(cv_scores)
    std_score = np.std(cv_scores)
    return mean_score, std_score, cv_scores

def ordinal_clas_stat2(ordinal_cl, X, y, rand_state = None, n_repeats=5, n_split = 5):
    # Realizar la validación cruzada repetida
    mae_cv_scores = []
    rmse_cv_scores = []
    for _ in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=rand_state)
        scs = ('neg_mean_absolute_error', 'neg_root_mean_squared_error') # MAE y RMSE
        scores = cross_validate(ordinal_cl, X, y, cv=cv, scoring=scs)
        mae_cv_scores.extend(-scores['test_neg_mean_absolute_error'])
        rmse_cv_scores.extend(-scores['test_neg_root_mean_squared_error'])

    # Calcular media y desviación estándar de los puntajes
    mean_mae = np.mean(mae_cv_scores)
    mean_rmse = np.mean(rmse_cv_scores)
    std_mae = np.std(mae_cv_scores)
    std_rmse = np.std(rmse_cv_scores)
    return mean_mae, mean_rmse, std_mae, std_rmse

def ordinal_encoding_nn(label, num_clases):
    ord_enc_y = []
    for k in label:
        shape_vector = np.zeros(num_clases)  
        shape_vector[:k+1] = 1 
        ord_enc_y.append(shape_vector)
    ord_enc_y = np.array(ord_enc_y)
    return ord_enc_y

def prediction2label_nn(pred: np.ndarray):
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1


def cross_val_metrics_nn(model, X, y, cv):
    kf = KFold(n_splits=cv)
    mae_scores = []
    rmse_scores = []

    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=16, verbose=0)

        y_pred = prediction2label_nn(model.predict(X_val_fold))
        y_val_lab = prediction2label_nn(y_val_fold)
        mae = mean_absolute_error(y_val_lab, y_pred)
        rmse = root_mean_squared_error(y_val_lab, y_pred)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        # Calcular los valores promedio y desviación estándar
        mae_mean = np.mean(mae_scores)
        mae_std = np.std(mae_scores)
        rmse_mean = np.mean(rmse_scores)
        rmse_std = np.std(rmse_scores)

    return mae_mean, mae_std, rmse_mean, rmse_std

def results_report(y_test, y_hat, label_enc):
    mae = mean_absolute_error(y_test, y_hat)
    rmse = root_mean_squared_error(y_test, y_hat)
    print('MAE: ', np.round(mae,3), '\nRMSE: ',np.round(rmse,3))

    # Calcular la matriz de confusión
    print(metrics.classification_report(y_test, y_hat))

    cm = metrics.confusion_matrix(y_test, y_hat, normalize = 'true')
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=.8)
    ax = sns.heatmap(cm, annot=True, fmt='.2f', cmap='bwr', vmin=0, vmax=1,
                xticklabels=label_enc.inverse_transform(label_enc.transform(label_enc.classes_)), 
                yticklabels=label_enc.inverse_transform(label_enc.transform(label_enc.classes_)))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    #plt.title('Confusion matrix')
    plt.xlabel('Predicted MIC', fontsize=10)
    plt.ylabel('True MIC', fontsize=10)
    plt.show()
