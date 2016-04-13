# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:51:30 2016

@author: GILLES Armand
"""

import numpy as np
from scipy import interp
from scipy.stats import sem
import operator

from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns


def get_metric(y_test, y_pred, plot=False):
    """
    Calcul metrics.
    In : y_test, y_pred
    Return : 
    If plot == True, then plot CM normalize
    """
    # Metrics
    metrics_classification = metrics.classification_report(y_test, y_pred)
    
    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    
    roc_auc_score = metrics.roc_auc_score(y_test, y_pred)
    
    recall_score = metrics.recall_score(y_test, y_pred)
    
    
    print "Metrics classification : " 
    print metrics_classification
    print "Accuracy score : "
    print accuracy_score 
    print "Roc auc score : "
    print roc_auc_score
    print "Recall score : "
    print recall_score
    
    # Confusion Matrix
    
    cm = metrics.confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    print "Confusion Matrix : "
    print cm
    print " % ------------------------ "
    print cm_normalized
    
    if plot == True:
        
        label_unique = y_test.unique()
    
        # Graph Confusion Matrix
        tick_marks = np.arange(len(label_unique))
        plt.figure(figsize=(8,6))
        sns.heatmap(cm_normalized, cmap='Greens',annot=True,linewidths=.5)
        plt.title('confusion matrix')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.xticks(tick_marks + 0.5, list(label_unique))
        plt.yticks(tick_marks + 0.5, list(reversed(list(label_unique))), rotation=0)

def get_roc_curve_cv(model, X, y):
    """
    Create graph with ROC curve with cross validation cv : 5
    If model is from Sklearn, the model is fit at each cv
    else if xgboost, we only predict proba at erach cv
    resutl : 6 ROC curve (5 cv + mean) 
    Exemple :
    graph_roc(gbm, X_test, y_test)
    
    """    
    
    cv = StratifiedKFold(y, n_folds=5)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    for i, (train, test) in enumerate(cv):
        try: #Sklearn API
            probas_ = model.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
            fpr, tpr, thresholds = metrics.roc_curve(y.iloc[test], probas_[:, 1])
        except: #XGboost API
             probas_ = model.predict(xgb.DMatrix(X.iloc[test]), ntree_limit=model.best_ntree_limit)
             fpr, tpr, thresholds = metrics.roc_curve(y.iloc[test], probas_)
        # Compute ROC curve and area the curve

        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves with 5 CV')
    plt.legend(loc="lower right")
    plt.show()
    
def display_scores(params, scores, append_star=False):
    """Format the mean score +/- std error for params"""
    params = ", ".join("{0}={1}".format(k, v)
                      for k, v in params.items())
    line = "{0}:\t{1:.3f} (+/-{2:.3f})".format(
        params, np.mean(scores), sem(scores))
    if append_star:
        line += " *"
    return line

def display_grid_scores(grid_scores, top=None):
    """Helper function to format a report on a grid of scores"""
    
    grid_scores = sorted(grid_scores, key=lambda x: x[1], reverse=True)
    if top is not None:
        grid_scores = grid_scores[:top]
        
    # Compute a threshold for staring models with overlapping
    # stderr:
    _, best_mean, best_scores = grid_scores[0]
    threshold = best_mean - 2 * sem(best_scores)
    
    for params, mean_score, scores in grid_scores:
        append_star = mean_score + 2 * sem(scores) > threshold
        print(display_scores(params, scores, append_star=append_star))
        
        
def get_grid(model, X, y):
    parameters = [{'C': [0.1, 1, 10,], 
              'solver' : ['sag']}, #,
              {'C': [0.1, 1, 10,], 
              'solver' : ['lbfgs']},
              {'C': [0.1, 1, 10,], 
              'solver' : ['liblinear']}]

    clf = GridSearchCV(model, parameters,
                       cv=5,
                       verbose=2, refit=True,scoring='roc_auc')
    
    clf.fit(X, y)
    
    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    print('Raw AUC score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))      
    
    display_grid_scores(clf.grid_scores_, top=20)