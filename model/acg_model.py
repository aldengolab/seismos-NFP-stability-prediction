# CAPP 30254: Machine Learning for Public Policy
# ALDEN GOLAB
# ML Pipeline
# 
# Model loop. 

## CODE STRUCTURE LIBERALLY BORROWED FROM RAYID GHANI, WITH EXTENSIVE EDITS ##
## https://github.com/rayidghani/magicloops/blob/master/magicloops.py ##
## Accessed: 5/5/2016 ##

from __future__ import division
import sys
import acg_read
import matplotlib.cm as cm
import copy
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
import acg_process
import random

THRESHOLD = .75

def define_project_params():
    '''
    Parameters specific to the project being run.
    '''
    models_to_run = ['KNN', 'RF','LR','AB','NB','DT','SGD']
    y_variable = '2014_YOY_revenue_fell'
    imp_cols = []
    robustscale_cols = ['2013_rev_change', '2012_grsincmembers', '2013_grsincmembers', '2012_inventory_sale_perofrev', '2012_investments_perofrev', '2012_programs_perofrev', '2013_inventory_sale_perofrev', '2013_investments_perofrev', '2013_programs_perofrev', '2013_inventory_sale_perofrev', '2013_investments_perofrev', '2013_programs_perofrev', '2013_rental_perofrev', '2012_debtassetratio', '2013_debtassetratio', '2012_supportrevratio', '2013_supportrevratio', '2013_initiationfees_changepercent', '2013_grsrcptspublicuse_changepercent', '2013_grsincmembers_changepercent', '2013_grsincother_changepercent', '2013_totcntrbgfts_changepercent', '2013_totprgmrevnue_changepercent', '2013_invstmntinc_changepercent', '2013_txexmptbndsproceeds_changepercent', '2013_royaltsinc_changepercent', '2013_grsrntsreal_changepercent', '2013_grsrntsprsnl_changepercent', '2013_rntlexpnsreal_changepercent', '2013_rntlexpnsprsnl_changepercent', '2013_rntlincreal_changepercent', '2013_rntlincprsnl_changepercent', '2013_netrntlinc_changepercent', '2013_grsalesecur_changepercent', '2013_grsalesothr_changepercent', '2013_cstbasisecur_changepercent', '2013_cstbasisothr_changepercent', '2013_gnlsecur_changepercent', '2013_gnlsothr_changepercent', '2013_netgnls_changepercent', '2013_grsincfndrsng_changepercent', '2013_lessdirfndrsng_changepercent', '2013_netincfndrsng_changepercent', '2013_grsincgaming_changepercent', '2013_lessdirgaming_changepercent', '2013_netincgaming_changepercent', '2013_grsalesinvent_changepercent', '2013_lesscstofgoods_changepercent', '2013_netincsales_changepercent', '2013_miscrevtot11e_changepercent', '2013_totrevenue_changepercent', '2013_compnsatncurrofcr_changepercent', '2013_othrsalwages_changepercent', '2013_payrolltx_changepercent', '2013_profndraising_changepercent', '2013_totfuncexpns_changepercent', '2013_totassetsend_changepercent', '2013_secrdmrtgsend_changepercent', '2013_txexmptbndsend_changepercent', '2013_unsecurednotesend_changepercent', '2013_totliabend_changepercent', '2013_retainedearnend_changepercent', '2013_totnetassetend_changepercent', '2013_gftgrntsrcvd170_changepercent', '2013_txrevnuelevied170_changepercent', '2013_grsinc170_changepercent', '2013_grsrcptsadmissn509_changepercent', '2013_subtotsuppinc509_changepercent', '2013_totsupp509_changepercent', 'GDP2002', 'GDP2003', 'GDP2004', 'GDP2006', 'GDP2007', 'GDP2008', 'GDP2009', 'GDP2010', 'GDP2011', 'GDP2012', 'GDP2013', 'GDP2014', '2013_noemplyeesw3cnt', '2012_totassetsend', '2013_totassetsend', '2012_totgftgrntrcvd509', '2013_totgftgrntrcvd509', '2014_totgftgrntrcvd509', '2012_totfuncexpns', '2013_totfuncexpns',  '2012_compnsatncurrofcr', '2013_compnsatncurrofcr', '2012_lessdirfndrsng', '2013_lessdirfndrsng', '2013_officexpns', '2013_interestamt']
    scale_columns = ['2013_rev_change', '2012_grsincmembers', '2013_grsincmembers', '2012_assets_sale_perofrev', '2012_inventory_sale_perofrev', '2012_investments_perofrev', '2012_fundraising_perofrev', '2012_programs_perofrev', '2012_rental_perofrev', '2013_assets_sale_perofrev', '2013_inventory_sale_perofrev', '2013_investments_perofrev', '2013_fundraising_perofrev', '2013_programs_perofrev', '2013_rental_perofrev', '2012_debtassetratio', '2013_debtassetratio', '2012_supportrevratio', '2013_supportrevratio', '2013_initiationfees_changepercent', '2013_grsrcptspublicuse_changepercent', '2013_grsincmembers_changepercent', '2013_grsincother_changepercent', '2013_totcntrbgfts_changepercent', '2013_totprgmrevnue_changepercent', '2013_invstmntinc_changepercent', '2013_txexmptbndsproceeds_changepercent', '2013_royaltsinc_changepercent', '2013_grsrntsreal_changepercent', '2013_grsrntsprsnl_changepercent', '2013_rntlexpnsreal_changepercent', '2013_rntlexpnsprsnl_changepercent',  '2013_rntlincreal_changepercent', '2013_rntlincprsnl_changepercent',  '2013_netrntlinc_changepercent', '2013_grsalesecur_changepercent', '2013_grsalesothr_changepercent', '2013_cstbasisecur_changepercent', '2013_cstbasisothr_changepercent', '2013_gnlsecur_changepercent', '2013_gnlsothr_changepercent', '2013_netgnls_changepercent', '2013_grsincfndrsng_changepercent', '2013_lessdirfndrsng_changepercent', '2013_netincfndrsng_changepercent', '2013_grsincgaming_changepercent', '2013_lessdirgaming_changepercent', '2013_netincgaming_changepercent', '2013_grsalesinvent_changepercent', '2013_lesscstofgoods_changepercent', '2013_netincsales_changepercent', '2013_miscrevtot11e_changepercent', '2013_totrevenue_changepercent', '2013_compnsatncurrofcr_changepercent', '2013_othrsalwages_changepercent', '2013_payrolltx_changepercent', '2013_profndraising_changepercent', '2013_totfuncexpns_changepercent', '2013_totassetsend_changepercent', '2013_txexmptbndsend_changepercent', '2013_secrdmrtgsend_changepercent', '2013_unsecurednotesend_changepercent', '2013_totliabend_changepercent', '2013_retainedearnend_changepercent', '2013_totnetassetend_changepercent', '2013_gftgrntsrcvd170_changepercent', '2013_txrevnuelevied170_changepercent', '2013_srvcsval170_changepercent', '2013_grsinc170_changepercent', '2013_grsrcptsrelated170_changepercent', '2013_totgftgrntrcvd509_changepercent', '2013_grsrcptsadmissn509_changepercent', '2013_txrevnuelevied509_changepercent', '2013_srvcsval509_changepercent', '2013_subtotsuppinc509_changepercent', '2013_totsupp509_changepercent', 'GDP2002', 'GDP2003', 'GDP2004', 'GDP2006', 'GDP2007', 'GDP2008', 'GDP2009', 'GDP2010', 'GDP2011', 'GDP2012', 'GDP2013', 'GDP2014', '2013_noemplyeesw3cnt', '2012_grsrcptspublicuse', '2013_grsrcptspublicuse', '2012_totassetsend', '2013_totassetsend', '2012_totgftgrntrcvd509', '2013_totgftgrntrcvd509', '2014_totgftgrntrcvd509', '2012_totfuncexpns', '2013_totfuncexpns',  '2012_compnsatncurrofcr', '2013_compnsatncurrofcr', '2012_lessdirfndrsng', '2013_lessdirfndrsng', '2013_officexpns', '2013_interestamt']
    X_variables = X_columns = ['2013_YOY_revenue_fell','2013_rev_change', '2013_payroll_change', '2013_totnetassetend_change', '2012_grsincmembers', '2013_grsincmembers', '2012_assets_sale_perofrev', '2012_assets_sale_perofrev_isnegative', '2012_inventory_sale_perofrev', '2012_inventory_sale_perofrev_isnegative', '2012_investments_perofrev', '2012_investments_perofrev_isnegative', '2012_fundraising_perofrev', '2012_fundraising_perofrev_isnegative', '2012_programs_perofrev', '2012_programs_perofrev_isnegative', '2012_rental_perofrev', '2012_rental_perofrev_isnegative', '2013_assets_sale_perofrev', '2013_assets_sale_perofrev_isnegative', '2013_inventory_sale_perofrev', '2013_inventory_sale_perofrev_isnegative', '2013_investments_perofrev', '2013_investments_perofrev_isnegative', '2013_fundraising_perofrev', '2013_fundraising_perofrev_isnegative', '2013_programs_perofrev', '2013_programs_perofrev_isnegative', '2013_rental_perofrev', '2013_rental_perofrev_isnegative', '2012_fundraisingROI', '2013_fundraisingROI', '2012_debtassetratio', '2013_debtassetratio', '2012_supportrevratio', '2013_supportrevratio', '2012__persupp_govtaxes', '2012__persupp_govservices', '2013__persupp_govtaxes', '2013__persupp_govservices', '2013_initiationfees_changepercent', '2013_grsrcptspublicuse_changepercent', '2013_grsincmembers_changepercent', '2013_grsincother_changepercent', '2013_totcntrbgfts_changepercent', '2013_totprgmrevnue_changepercent', '2013_invstmntinc_changepercent', '2013_txexmptbndsproceeds_changepercent', '2013_royaltsinc_changepercent', '2013_grsrntsreal_changepercent', '2013_grsrntsprsnl_changepercent', '2013_rntlexpnsreal_changepercent', '2013_rntlexpnsprsnl_changepercent', '2013_rntlincreal_changepercent', '2013_rntlincprsnl_changepercent', '2013_netrntlinc_changepercent', '2013_grsalesecur_changepercent', '2013_grsalesothr_changepercent', '2013_cstbasisecur_changepercent', '2013_cstbasisothr_changepercent', '2013_gnlsecur_changepercent', '2013_gnlsothr_changepercent', '2013_netgnls_changepercent', '2013_grsincfndrsng_changepercent', '2013_lessdirfndrsng_changepercent', '2013_netincfndrsng_changepercent', '2013_grsincgaming_changepercent', '2013_lessdirgaming_changepercent', '2013_netincgaming_changepercent', '2013_grsalesinvent_changepercent', '2013_lesscstofgoods_changepercent', '2013_netincsales_changepercent', '2013_miscrevtot11e_changepercent', '2013_totrevenue_changepercent', '2013_compnsatncurrofcr_changepercent', '2013_othrsalwages_changepercent', '2013_payrolltx_changepercent', '2013_profndraising_changepercent', '2013_totfuncexpns_changepercent', '2013_totassetsend_changepercent', '2013_txexmptbndsend_changepercent', '2013_secrdmrtgsend_changepercent', '2013_unsecurednotesend_changepercent', '2013_totliabend_changepercent', '2013_retainedearnend_changepercent', '2013_totnetassetend_changepercent', '2013_gftgrntsrcvd170_changepercent', '2013_txrevnuelevied170_changepercent', '2013_srvcsval170_changepercent', '2013_grsinc170_changepercent', '2013_grsrcptsrelated170_changepercent', '2013_totgftgrntrcvd509_changepercent', '2013_grsrcptsadmissn509_changepercent', '2013_txrevnuelevied509_changepercent', '2013_srvcsval509_changepercent', '2013_subtotsuppinc509_changepercent', '2013_totsupp509_changepercent', 'GDP2002', 'GDP2003', 'GDP2004', 'GDP2005', 'GDP2006', 'GDP2007', 'GDP2008', 'GDP2009', 'GDP2010', 'GDP2011', 'GDP2012', 'GDP2013', '2012_missing', '2013_missing', '2013_noemplyeesw3cnt', '2012_grsrcptspublicuse', '2013_grsrcptspublicuse', '2012_totassetsend', '2013_totassetsend', '2012_totgftgrntrcvd509', '2013_totgftgrntrcvd509', '2012_totfuncexpns', '2013_totfuncexpns', '2012_compnsatncurrofcr', '2013_compnsatncurrofcr', '2012_lessdirfndrsng', '2013_lessdirfndrsng', '2013_officexpns', '2013_interestamt']
        
    return (y_variable, imp_cols, models_to_run, robustscale_cols, 
        scale_columns, X_variables)

def define_clfs_params():
    '''
    Defines all relevant parameters and classes for classfier objects.
    '''
    clfs = {
        'RF': RandomForestClassifier(n_estimators = 50, n_jobs = -1),
        'ET': ExtraTreesClassifier(n_estimators = 10, n_jobs = -1, criterion = 'entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1), algorithm = "SAMME", n_estimators = 200),
        'LR': LogisticRegression(penalty = 'l1', C = 1e5),
        'SVM': svm.SVC(kernel = 'linear', probability = True, random_state = 0),
        'GB': GradientBoostingClassifier(learning_rate = 0.05, subsample = 0.5, max_depth = 6, n_estimators = 10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss = 'log', penalty = 'l2'),
        'KNN': KNeighborsClassifier(n_neighbors = 3) 
        }
    params = { 
        'RF':{'n_estimators': [1,10,100,1000], 'max_depth': [1,2,3], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'LR': {'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
        'SGD': {'loss': ['log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
        'ET': {'n_estimators': [1,10,100,1000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,3], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'AB': {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000]},
        'GB': {'n_estimators': [1,10,100,1000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
        'NB' : {},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,2,3], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
        'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
        }
    
    return clfs, params

def clf_loop(dataframe, clfs, models_to_run, params, y_variable, X_variables, 
 imp_cols = [], addl_runs = 0, evalution = ['AUC', 'precision', 'recall'], stat_k = .10, plot = False, 
 robustscale_cols = [], scale_columns = [], params_iter_max = 50):
    '''
    Runs through each model specified by models_to_run once with each possible
    setting in params.
    '''
    N = 0
    maximum = ('name', 0, 0)
    for n in range(1 + addl_runs):
        print('Sampling new test/train split...')
        # Drop NaNs on y variable, since this is neeeded for validation
        dataframe.dropna(subset = [y_variable], inplace = True)
        X_train, X_test, y_train, y_test = acg_process.test_train_split(dataframe, 
            y_variable, test_size=0.1)
        # Limit to X & Y variables that have been specified
        X_train = X_train[X_variables]
        X_test = X_test[X_variables]
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        print('Imputing data for new split...')
        for col in imp_cols:
            X_train, median = acg_process.impute_median(X_train, col)
            X_test = acg_process.impute_specific(X_test, col, median)
        print('Finished imputing, transforming data...')
        for col in robustscale_cols:
            if col in X_variables:
                X_train, scaler = acg_process.robust_transform(X_train, col, keep = True)
                X_test = acg_process.robust_transform(X_test, col, scaler = scaler)
        for col in scale_columns:
            if col in X_variables: 
                X_train, maxval, minval = acg_process.normalize_scale(X_train, col = col, keep = True)
                X_test = acg_process.normalize_scale(X_test, col = col, maxval = maxval, minval = minval)
        print('Training model...')
        for index, clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            # Iterate through all possible parameter combinations
            parameter_values = params[models_to_run[index]]
            grid = ParameterGrid(parameter_values)
            iteration = 0
            for p in grid:
                # If cut-off of parameter iterations expected, choose random
                if len(grid) > params_iter_max:
                    p = random.choice(list(grid))
                # Run until hitting max number of parameter iterations
                if iteration < params_iter_max:
                    try:
                        clf.set_params(**p)
                        print(clf)
                        y_pred_probs = clf.fit(X_train, 
                        y_train).predict_proba(
                            X_test)[:,1]
                        if 'precision' in evalution:
                            result = precision_at_k(y_test, y_pred_probs, 
                            stat_k)
                            print('Precision: ', result)
                            if result[0] > maximum[1]:
                                maximum = (clf, result[0], result[1])
                                print('Max Precision: {}'.format(maximum))
                                plot_precision_recall_n(y_test, 
                                y_pred_probs, clf, N)
                                N += 1
                        if 'AUC' in evalution:
                            result = auc_roc(y_test, y_pred_probs)
                            print('AUC: {}'.format(result))
                        if plot and result[0] <= maximum[1]:
                            plot_precision_recall_n(y_test, y_pred_probs, clf, N)
                            N += 1
                        if 'recall' in evalution:
                            print('Recall: ', recall_at_k(y_test, y_pred_probs, stat_k))
                        print('Accuracy: ', accuracy_at_k(y_test, y_pred_probs, stat_k))
                        iteration += 1
                    except IndexError as e:
                        print('Error: {0}'.format(e))
                        continue
                    except RuntimeError as e:
                        print('RuntimeError: {}'.format(e))
                        continue
                    except AttributeError as e:
                        print('AttributeError: {}'.format(e))
                        continue


def plot_precision_recall_n(y_true, y_prob, model_name, N):
    '''
    Plots the precision recall curve.
    '''
    from sklearn.metrics import precision_recall_curve

    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    
    model = str(model_name)[:20]
    name = '{}_{}.png'.format(model, N)
    print('File saved as {} for model above'.format(name))
    plt.title(name)
    plt.savefig(name)
    #plt.show()
    plt.close()
    
def accuracy_at_k(y_true, y_scores, k = None):
    '''
    Dyanamic k-threshold accuracy. Defines threshold for Positive at the 
    value that returns the k*n top values where k is within [0-1]. If k is not
    specified, threshold will default to THRESHOLD.
    '''
    if k != None:
        threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))] 
    else: 
        threshold = THRESHOLD
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return (metrics.accuracy_score(y_true, y_pred), threshold)

def auc_roc(y_true, y_scores):
    '''
    Computes the Area-Under-the-Curve for the ROC curve. 
    '''
    return metrics.roc_auc_score(y_true, y_scores)

def precision_at_k(y_true, y_scores, k = None):
    '''
    Dyanamic k-threshold precision. Defines threshold for Positive at the 
    value that returns the k*n top values where k is within [0-1]. If k is not
    specified, threshold will default to THRESHOLD.
    '''
    if k != None:
        threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))] 
    else: 
        threshold = THRESHOLD
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return (metrics.precision_score(y_true, y_pred), threshold)

def recall_at_k(y_true, y_scores, k = None):
    '''
    Dyanamic k-threshold recall. Defines threshold for Positive at the 
    value that returns the k*n top values where k is within [0-1]. If k is not
    specified, threshold will default to THRESHOLD.
    '''
    if k != None:
        threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))] 
    else: 
        threshold = THRESHOLD
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return (metrics.recall_score(y_true, y_pred), threshold)

def main(filename): 
    '''
    Runs the loop.
    '''
    dataframe = acg_read.load_file(filename, index = 0)
    # Get all the necessary parameters
    clfs, params = define_clfs_params()
    y_variable, imp_cols, models_to_run, robustscale_cols, scale_columns, \
     X_variables = define_project_params()
    # Remove any infinities, replace with missing
    dataframe.replace([np.inf, -np.inf], np.nan)
    # Find any columns with missing values, set to impute
    for x in X_variables: 
        if len(dataframe[dataframe[x].isnull()]) > 0:
            imp_cols.append(x)
    # If a column has more than 40% missing, don't use
    X_drop = []
    for x in X_variables: 
        if len(dataframe[dataframe[x].isnull()]) / len(dataframe) > 0.4: 
            X_drop.append(x)
    for x in X_drop: 
        X_variables.remove(x)
        imp_cols.remove(x)
    # Run the loop
    clf_loop(dataframe, clfs, models_to_run, params, y_variable, X_variables, 
        imp_cols = imp_cols, scale_columns = scale_columns)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        data = acg_read.load_file(sys.argv[1])
        main(sys.argv[1])
    else:
        print('Usage: -u model.py <datafilename> > <results_file>')
