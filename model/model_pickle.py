## ML Pickling  ##
## June 4, 2016 ##
## 
## CODE STRUCTURE LIBERALLY BORROWED FROM RAYID GHANI, WITH EXTENSIVE EDITS ##
## https://github.com/rayidghani/magicloops/blob/master/magicloops.py ##
## Accessed: 5/5/2016 ##

'''
HOW TO USE THIS FILE: 

Take the successful model & parameters from acg_model.py run and put them in 
define_clfs_params; use models_to_run to specify which modeling technique. 

This file will iterate through 20 runs of the data with different 90/10 splits
in order to get best precision model with these params. Every time it improves
the Precision, it will pickle a model. Final pickle is the best performing model.
'''

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
from sklearn.tree import DecisionTreeClassifier, export_graphviz
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
from sklearn.externals import joblib

# Our data is processed using year labels, however these need to be 
# generalized for the model preservation and use

TEST_LABELS = {'2004':'10YP', '2009': '5YP', '2011': '3YP', '2012': '2YP', '2013': '1YP', '2014': 'Y00'}

def define_project_params():
    '''
    Parameters specific to the project being run.
    '''
    models_to_run = ['NB','DT','SGD','KNN'] # Use this to specify the model type
    y_variable = 'Y00_YOY_revenue_fell'
    imp_cols = []
    robustscale_cols = ['1YP_rev_change', '2YP_grsincmembers', '1YP_grsincmembers', '2YP_inventory_sale_perofrev', '2YP_investments_perofrev', '2YP_programs_perofrev', '1YP_inventory_sale_perofrev', '1YP_investments_perofrev', '1YP_programs_perofrev', '1YP_inventory_sale_perofrev', '1YP_investments_perofrev', '1YP_programs_perofrev', '1YP_rental_perofrev', '2YP_debtassetratio', '1YP_debtassetratio', '2YP_supportrevratio', '1YP_supportrevratio', '1YP_initiationfees_changepercent', '1YP_grsrcptspublicuse_changepercent', '1YP_grsincmembers_changepercent', '1YP_grsincother_changepercent', '1YP_totcntrbgfts_changepercent', '1YP_totprgmrevnue_changepercent', '1YP_invstmntinc_changepercent', '1YP_txexmptbndsproceeds_changepercent', '1YP_royaltsinc_changepercent', '1YP_grsrntsreal_changepercent', '1YP_grsrntsprsnl_changepercent', '1YP_rntlexpnsreal_changepercent', '1YP_rntlexpnsprsnl_changepercent', '1YP_rntlincreal_changepercent', '1YP_rntlincprsnl_changepercent', '1YP_netrntlinc_changepercent', '1YP_grsalesecur_changepercent', '1YP_grsalesothr_changepercent', '1YP_cstbasisecur_changepercent', '1YP_cstbasisothr_changepercent', '1YP_gnlsecur_changepercent', '1YP_gnlsothr_changepercent', '1YP_netgnls_changepercent', '1YP_grsincfndrsng_changepercent', '1YP_lessdirfndrsng_changepercent', '1YP_netincfndrsng_changepercent', '1YP_grsincgaming_changepercent', '1YP_lessdirgaming_changepercent', '1YP_netincgaming_changepercent', '1YP_grsalesinvent_changepercent', '1YP_lesscstofgoods_changepercent', '1YP_netincsales_changepercent', '1YP_miscrevtot11e_changepercent', '1YP_totrevenue_changepercent', '1YP_compnsatncurrofcr_changepercent', '1YP_othrsalwages_changepercent', '1YP_payrolltx_changepercent', '1YP_profndraising_changepercent', '1YP_totfuncexpns_changepercent', '1YP_totassetsend_changepercent', '1YP_secrdmrtgsend_changepercent', '1YP_txexmptbndsend_changepercent', '1YP_unsecurednotesend_changepercent', '1YP_totliabend_changepercent', '1YP_retainedearnend_changepercent', '1YP_totnetassetend_changepercent', '1YP_gftgrntsrcvd170_changepercent', '1YP_txrevnuelevied170_changepercent', '1YP_grsinc170_changepercent', '1YP_grsrcptsadmissn509_changepercent', '1YP_subtotsuppinc509_changepercent', '1YP_totsupp509_changepercent', 'GDP2002', 'GDP2003', 'GDP2004', 'GDP2006', 'GDP2007', 'GDP2008', 'GDP2009', 'GDP2010', 'GDP2011', 'GDP2YP', 'GDP1YP', 'GDP2014', '1YP_noemplyeesw3cnt', '2YP_totassetsend', '1YP_totassetsend', '2YP_totgftgrntrcvd509', '1YP_totgftgrntrcvd509', '2014_totgftgrntrcvd509', '2YP_totfuncexpns', '1YP_totfuncexpns',  '2YP_compnsatncurrofcr', '1YP_compnsatncurrofcr', '2YP_lessdirfndrsng', '1YP_lessdirfndrsng', '1YP_officexpns', '1YP_interestamt']
    scale_columns = ['1YP_rev_change', '2YP_grsincmembers', '1YP_grsincmembers', '2YP_assets_sale_perofrev', '2YP_inventory_sale_perofrev', '2YP_investments_perofrev', '2YP_fundraising_perofrev', '2YP_programs_perofrev', '2YP_rental_perofrev', '1YP_assets_sale_perofrev', '1YP_inventory_sale_perofrev', '1YP_investments_perofrev', '1YP_fundraising_perofrev', '1YP_programs_perofrev', '1YP_rental_perofrev', '2YP_debtassetratio', '1YP_debtassetratio', '2YP_supportrevratio', '1YP_supportrevratio', '1YP_initiationfees_changepercent', '1YP_grsrcptspublicuse_changepercent', '1YP_grsincmembers_changepercent', '1YP_grsincother_changepercent', '1YP_totcntrbgfts_changepercent', '1YP_totprgmrevnue_changepercent', '1YP_invstmntinc_changepercent', '1YP_txexmptbndsproceeds_changepercent', '1YP_royaltsinc_changepercent', '1YP_grsrntsreal_changepercent', '1YP_grsrntsprsnl_changepercent', '1YP_rntlexpnsreal_changepercent', '1YP_rntlexpnsprsnl_changepercent',  '1YP_rntlincreal_changepercent', '1YP_rntlincprsnl_changepercent',  '1YP_netrntlinc_changepercent', '1YP_grsalesecur_changepercent', '1YP_grsalesothr_changepercent', '1YP_cstbasisecur_changepercent', '1YP_cstbasisothr_changepercent', '1YP_gnlsecur_changepercent', '1YP_gnlsothr_changepercent', '1YP_netgnls_changepercent', '1YP_grsincfndrsng_changepercent', '1YP_lessdirfndrsng_changepercent', '1YP_netincfndrsng_changepercent', '1YP_grsincgaming_changepercent', '1YP_lessdirgaming_changepercent', '1YP_netincgaming_changepercent', '1YP_grsalesinvent_changepercent', '1YP_lesscstofgoods_changepercent', '1YP_netincsales_changepercent', '1YP_miscrevtot11e_changepercent', '1YP_totrevenue_changepercent', '1YP_compnsatncurrofcr_changepercent', '1YP_othrsalwages_changepercent', '1YP_payrolltx_changepercent', '1YP_profndraising_changepercent', '1YP_totfuncexpns_changepercent', '1YP_totassetsend_changepercent', '1YP_txexmptbndsend_changepercent', '1YP_secrdmrtgsend_changepercent', '1YP_unsecurednotesend_changepercent', '1YP_totliabend_changepercent', '1YP_retainedearnend_changepercent', '1YP_totnetassetend_changepercent', '1YP_gftgrntsrcvd170_changepercent', '1YP_txrevnuelevied170_changepercent', '1YP_srvcsval170_changepercent', '1YP_grsinc170_changepercent', '1YP_grsrcptsrelated170_changepercent', '1YP_totgftgrntrcvd509_changepercent', '1YP_grsrcptsadmissn509_changepercent', '1YP_txrevnuelevied509_changepercent', '1YP_srvcsval509_changepercent', '1YP_subtotsuppinc509_changepercent', '1YP_totsupp509_changepercent', 'GDP2002', 'GDP2003', 'GDP2004', 'GDP2006', 'GDP2007', 'GDP2008', 'GDP2009', 'GDP2010', 'GDP2011', 'GDP2YP', 'GDP1YP', 'GDP2014', '1YP_noemplyeesw3cnt', '2YP_grsrcptspublicuse', '1YP_grsrcptspublicuse', '2YP_totassetsend', '1YP_totassetsend', '2YP_totgftgrntrcvd509', '1YP_totgftgrntrcvd509', '2014_totgftgrntrcvd509', '2YP_totfuncexpns', '1YP_totfuncexpns',  '2YP_compnsatncurrofcr', '1YP_compnsatncurrofcr', '2YP_lessdirfndrsng', '1YP_lessdirfndrsng', '1YP_officexpns', '1YP_interestamt']

    return (y_variable, imp_cols, models_to_run, robustscale_cols, scale_columns)

def define_clfs_params():
    '''
    Defines all relevant parameters and classes for classfier objects.
    '''
    clfs = {
        'RF': RandomForestClassifier(n_estimators = 50, n_jobs = -1),
        'ET': ExtraTreesClassifier(n_estimators = 10, n_jobs = -1, criterion = 'entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth = [1, 5, 10, 15]), algorithm = "SAMME", n_estimators = 200),
        'LR': LogisticRegression(penalty = 'l1', C = 1e5),
        'SVM': svm.SVC(kernel = 'linear', probability = True, random_state = 0),
        'GB': GradientBoostingClassifier(learning_rate = 0.05, subsample = 0.5, max_depth = 6, n_estimators = 10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss = 'log', penalty = 'l2'),
        'KNN': KNeighborsClassifier(n_neighbors = 3)
        }
    params = {
        'RF': {'n_estimators': [10], 'max_depth': [10], 'max_features': ['sqrt'], 'min_samples_split': [5], 'random_state': [1]},
        'LR': {'penalty': [], 'C': [],'random_state': []},
        'SGD': {'loss': [], 'penalty': [], 'random_state': []},
        'ET': {'n_estimators': [], 'criterion' : [] ,'max_depth': [], 'max_features': [],'min_samples_split': [], 'random_state': []},
        'AB': {'algorithm': [], 'n_estimators': [], 'random_state': []},
        'GB': {'n_estimators': [], 'learning_rate' : [],'subsample' : [], 'max_depth': [], 'random_state': []},
        'NB' : {},
        'DT': {'criterion': [], 'max_depth': [], 'max_features': [],'min_samples_split': [], 'random_state': []},
        'SVM' :{'C' :[],'kernel':[], 'random_state': []},
        'KNN' :{'n_neighbors': [],'weights': [],'algorithm': [], 'random_state': []}
        }

    return clfs, params

def clf_loop(dataframe, clfs, models_to_run, params, y_variable, X_variables,
 imp_cols = [], addl_runs = 0, evalution = ['AUC', 'precision', 'recall'], stat_k = .10, plot = False,
 robustscale_cols = [], scale_columns = [], params_iter_max = 50, randomize_features = None):
    '''
    Runs through each model specified by models_to_run once with each possible
    setting in params. For boosting don't include randomize_features.
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
        if randomize_features:
            size = len(X_variables) * randomize_features
            rand_X = random.sample(X_variables, int(size))
            print("New X Variables: {}".format(X_variables))
        else:
            rand_X = X_variables
        X_train = X_train[rand_X]
        X_test = X_test[rand_X]
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        print('Imputing data for new split...')
        for col in imp_cols:
            if col in rand_X:
                X_train, median = acg_process.impute_median(X_train, col)
                X_test = acg_process.impute_specific(X_test, col, median)
        print('Finished imputing, transforming data...')
        for col in robustscale_cols:
            if col in rand_X:
                X_train, scaler = acg_process.robust_transform(X_train, col, keep = True)
                X_test = acg_process.robust_transform(X_test, col, scaler = scaler)
        for col in scale_columns:
            if col in rand_X:
                X_train, maxval, minval = acg_process.normalize_scale(X_train, col = col, keep = True)
                X_test = acg_process.normalize_scale(X_test, col = col, maxval = maxval, minval = minval)
        print('Finished transfroming. The final training set has the shape',X_train.shape)
        for index, clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            sys.stderr.write('Running {}.'.format(models_to_run[index]))
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
                        y_pred_probs = clf.fit(X_train, y_train).predict_proba(
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
                                path = 'maxPrecisionModel{}.pkl'.format(N)
                                joblib.dump(clf, 'path')
                                N += 1
                                if models_to_run[index] == 'RF':
                                    importances = clf.feature_importances_
                                    sortedidx = np.argsort(importances)
                                    best_features = X_train.columns[sortedidx]
                                    print('Best Features: {}'.format(best_features[::-1]))
                                if models_to_run[index] == 'DT':
                                    export_graphviz(clf, 'DT_graph_' + str(N) + '.dot')
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

def main(filename, start_year):
    '''
    Runs the loop.
    '''
    dataframe = acg_read.load_file(filename, index = 0)
    new_col_names = []
    for x in dataframe.columns: 
        year = re.search('[2][0-9]*', x)
        if year != None and year.group(0) in TEST_LABELS:
            gen_year = TEST_LABELS[year.group(0)]
            new_col_names.append(x[:year.start()] + gen_year + x[year.end():])
        else:
            new_col_names.append(x)
    dataframe.columns = new_col_names
    # Get all the necessary parameters
    clfs, params = define_clfs_params()
    y_variable, imp_cols, models_to_run, robustscale_cols, scale_columns = define_project_params()
    X_variables = [i for i in dataframe.columns if i != y_variable]
    # Remove any infinities, replace with missing
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    # Find any columns with missing values, set to impute
    for x in X_variables:
        if len(dataframe[dataframe[x].isnull()]) > 0:
            imp_cols.append(x)
    # Drop row if missing y-variable
    dataframe = dataframe[dataframe[y_variable].notnull()]
    # If a column has more than 40% missing, don't use
    X_drop = []
    for x in X_variables:
        if len(dataframe[dataframe[x].isnull()]) / len(dataframe) > 0.4:
            X_drop.append(x)
    for x in X_drop:
        if x in X_variables:
            X_variables.remove(x)
        if x in imp_cols:
            imp_cols.remove(x)
    # Run the loop
    clf_loop(dataframe, clfs, models_to_run, params, y_variable, X_variables,
        imp_cols = imp_cols, scale_columns = scale_columns, 
        robustscale_cols = robustscale_cols, addl_runs = 19)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1], sys.argv[2])
    else:
        print('Usage: -u model.py <featuresfilename> <start year>')
