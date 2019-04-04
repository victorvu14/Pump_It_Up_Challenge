#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 09:19:35 2019

@author: federicologuercio
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score as metric_scorer
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split

def score_model(model, x, y):
    scores = cross_val_score(model, x, y, cv=5)
    return scores

def cv_evaluate(df, splits = 5, model = make_pipeline(LogisticRegression(multi_class = 'ovr', solver = 'lbfgs', max_iter = 400)), transformers = None, grid = None, confusion = False):
    TARGET_VARIABLE = 'status_group'
    METRIC = 'accuracy'
    X = df.loc[:, df.columns != TARGET_VARIABLE]
    y = df.loc[:, TARGET_VARIABLE]
    #train_size = int(len(df) * 0.85)
    #X_train, X_validate, y_train, y_validate = X[0:train_size], X[train_size:len(df)], y[0:train_size], y[train_size:len(df)]
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size = 0.15)

    if transformers:
        model = make_pipeline(model)
        for ind,i in enumerate(transformers):
            model.steps.insert(ind,[str(ind+1),i])

    if grid:
        model = RandomizedSearchCV(model, grid, scoring = METRIC, cv=splits, n_iter = 20, refit=True, return_train_score = False, error_score=0.0)
        model.fit(X_train, y_train)
        scores = model.cv_results_['mean_test_score']
    else:
        scores = cross_val_score(model, X_train, y_train, scoring = METRIC, cv = splits)
        model.fit(X_train, y_train)

    pred = model.predict(X_validate)
    final_score = metric_scorer(y_validate, pred)
    
    if confusion:
        print('Classification report \n', classification_report(y_validate,pred))
    
    return final_score, scores, model

def feature_engineering_pipeline(df, models, transformers, splits = 5):
    all_scores  = pd.DataFrame(columns = ['Model', 'Function', 'CV Score', 'Holdout Score', 'Difference', 'Outcome'])

    for model in models:
        best_score = 0
        top_score, scores, cv_model = cv_evaluate(df, model = model['model'], splits = splits)
        model['score'] = top_score
        model['transformers'] = []
        all_scores = all_scores.append({'Model': model['name'], 'Function':'base_score','CV Score': '{:.2f} +/- {:.02}'.format(np.mean(scores[scores > 0.0]),np.std(scores[scores > 0.0])),'Holdout Score': top_score, 'Difference': 0, 'Outcome': 'Base ' + model['name']}, ignore_index=True)
        
        difference = (best_score - top_score)
        if difference > 0:
            best_score = top_score
            model['score'] = top_score
        
        for transformer in transformers:
            engineered_data = df.copy()
            outcome = 'Rejected'
            
            try:
                transformer_score, scores, cv_model = cv_evaluate(engineered_data, model = model['model'], transformers = [transformer['transformer']], splits = splits)
                difference = (transformer_score - top_score)
                
                if difference > 0:
                    model['transformers'] = [i for i in model['transformers'] if i['name'] != transformer['name']]
                    model['transformers'].append(transformer['transformer'])
                    best_score = transformer_score
                    outcome = 'Accepted'
                
                mean = np.mean(scores[scores > 0.0])
                std = np.std(scores[scores > 0.0])
                if np.isnan(mean) or np.isnan(std):
                    mean = 0.00
                    std = 0.00

                score = {'Model': model['name'], 'Function':transformer['name'],'CV Score': '{:.2f} +/- {:.02}'.format(mean,std),'Holdout Score': transformer_score, 'Difference': difference, 'Outcome': outcome}

            except: 
                score = {'Model': model['name'], 'Function':transformer['name'],'CV Score': '0.00 +/- 0.00','Holdout Score': 0, 'Difference': 0, 'Outcome': 'Error'}
        
            all_scores = all_scores.append(score, ignore_index=True)

    return create_pipelines(models), all_scores

def create_pipelines(pipes):
    for item in pipes:
        item['pipeline'] = make_pipeline(*item['transformers'], item['model'])
    
    return sorted(pipes, key=lambda k: k['score'], reverse = True) 