## ML Pickling  ##
## June 4, 2016 ##
## 
## CODE STRUCTURE LIBERALLY BORROWED FROM RAYID GHANI, WITH EXTENSIVE EDITS ##
## https://github.com/rayidghani/magicloops/blob/master/magicloops.py ##
## Accessed: 5/5/2016 ##

'''
HOW TO USE THIS FILE: 

Specify a pickled classification model and run against validation year.
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

VERIFICATION_LABELS = {'2005':'10YP', '2010': '5YP', '2012': '3YP', '2013': '2YP', '2014': '1YP', '2015': 'Y00'}

def define_project_params():
    '''
    Parameters specific to the project being run.
    '''
    y_variable = 'Y00_YOY_revenue_fell'
    imp_cols = []
    robustscale_cols = ['1YP_rev_change', '2YP_grsincmembers', '1YP_grsincmembers', '2YP_inventory_sale_perofrev', '2YP_investments_perofrev', '2YP_programs_perofrev', '1YP_inventory_sale_perofrev', '1YP_investments_perofrev', '1YP_programs_perofrev', '1YP_inventory_sale_perofrev', '1YP_investments_perofrev', '1YP_programs_perofrev', '1YP_rental_perofrev', '2YP_debtassetratio', '1YP_debtassetratio', '2YP_supportrevratio', '1YP_supportrevratio', '1YP_initiationfees_changepercent', '1YP_grsrcptspublicuse_changepercent', '1YP_grsincmembers_changepercent', '1YP_grsincother_changepercent', '1YP_totcntrbgfts_changepercent', '1YP_totprgmrevnue_changepercent', '1YP_invstmntinc_changepercent', '1YP_txexmptbndsproceeds_changepercent', '1YP_royaltsinc_changepercent', '1YP_grsrntsreal_changepercent', '1YP_grsrntsprsnl_changepercent', '1YP_rntlexpnsreal_changepercent', '1YP_rntlexpnsprsnl_changepercent', '1YP_rntlincreal_changepercent', '1YP_rntlincprsnl_changepercent', '1YP_netrntlinc_changepercent', '1YP_grsalesecur_changepercent', '1YP_grsalesothr_changepercent', '1YP_cstbasisecur_changepercent', '1YP_cstbasisothr_changepercent', '1YP_gnlsecur_changepercent', '1YP_gnlsothr_changepercent', '1YP_netgnls_changepercent', '1YP_grsincfndrsng_changepercent', '1YP_lessdirfndrsng_changepercent', '1YP_netincfndrsng_changepercent', '1YP_grsincgaming_changepercent', '1YP_lessdirgaming_changepercent', '1YP_netincgaming_changepercent', '1YP_grsalesinvent_changepercent', '1YP_lesscstofgoods_changepercent', '1YP_netincsales_changepercent', '1YP_miscrevtot11e_changepercent', '1YP_totrevenue_changepercent', '1YP_compnsatncurrofcr_changepercent', '1YP_othrsalwages_changepercent', '1YP_payrolltx_changepercent', '1YP_profndraising_changepercent', '1YP_totfuncexpns_changepercent', '1YP_totassetsend_changepercent', '1YP_secrdmrtgsend_changepercent', '1YP_txexmptbndsend_changepercent', '1YP_unsecurednotesend_changepercent', '1YP_totliabend_changepercent', '1YP_retainedearnend_changepercent', '1YP_totnetassetend_changepercent', '1YP_gftgrntsrcvd170_changepercent', '1YP_txrevnuelevied170_changepercent', '1YP_grsinc170_changepercent', '1YP_grsrcptsadmissn509_changepercent', '1YP_subtotsuppinc509_changepercent', '1YP_totsupp509_changepercent', 'GDP2002', 'GDP2003', 'GDP2004', 'GDP2006', 'GDP2007', 'GDP2008', 'GDP2009', 'GDP2010', 'GDP2011', 'GDP2YP', 'GDP1YP', 'GDP2014', '1YP_noemplyeesw3cnt', '2YP_totassetsend', '1YP_totassetsend', '2YP_totgftgrntrcvd509', '1YP_totgftgrntrcvd509', '2014_totgftgrntrcvd509', '2YP_totfuncexpns', '1YP_totfuncexpns',  '2YP_compnsatncurrofcr', '1YP_compnsatncurrofcr', '2YP_lessdirfndrsng', '1YP_lessdirfndrsng', '1YP_officexpns', '1YP_interestamt']
    scale_columns = ['1YP_rev_change', '2YP_grsincmembers', '1YP_grsincmembers', '2YP_assets_sale_perofrev', '2YP_inventory_sale_perofrev', '2YP_investments_perofrev', '2YP_fundraising_perofrev', '2YP_programs_perofrev', '2YP_rental_perofrev', '1YP_assets_sale_perofrev', '1YP_inventory_sale_perofrev', '1YP_investments_perofrev', '1YP_fundraising_perofrev', '1YP_programs_perofrev', '1YP_rental_perofrev', '2YP_debtassetratio', '1YP_debtassetratio', '2YP_supportrevratio', '1YP_supportrevratio', '1YP_initiationfees_changepercent', '1YP_grsrcptspublicuse_changepercent', '1YP_grsincmembers_changepercent', '1YP_grsincother_changepercent', '1YP_totcntrbgfts_changepercent', '1YP_totprgmrevnue_changepercent', '1YP_invstmntinc_changepercent', '1YP_txexmptbndsproceeds_changepercent', '1YP_royaltsinc_changepercent', '1YP_grsrntsreal_changepercent', '1YP_grsrntsprsnl_changepercent', '1YP_rntlexpnsreal_changepercent', '1YP_rntlexpnsprsnl_changepercent',  '1YP_rntlincreal_changepercent', '1YP_rntlincprsnl_changepercent',  '1YP_netrntlinc_changepercent', '1YP_grsalesecur_changepercent', '1YP_grsalesothr_changepercent', '1YP_cstbasisecur_changepercent', '1YP_cstbasisothr_changepercent', '1YP_gnlsecur_changepercent', '1YP_gnlsothr_changepercent', '1YP_netgnls_changepercent', '1YP_grsincfndrsng_changepercent', '1YP_lessdirfndrsng_changepercent', '1YP_netincfndrsng_changepercent', '1YP_grsincgaming_changepercent', '1YP_lessdirgaming_changepercent', '1YP_netincgaming_changepercent', '1YP_grsalesinvent_changepercent', '1YP_lesscstofgoods_changepercent', '1YP_netincsales_changepercent', '1YP_miscrevtot11e_changepercent', '1YP_totrevenue_changepercent', '1YP_compnsatncurrofcr_changepercent', '1YP_othrsalwages_changepercent', '1YP_payrolltx_changepercent', '1YP_profndraising_changepercent', '1YP_totfuncexpns_changepercent', '1YP_totassetsend_changepercent', '1YP_txexmptbndsend_changepercent', '1YP_secrdmrtgsend_changepercent', '1YP_unsecurednotesend_changepercent', '1YP_totliabend_changepercent', '1YP_retainedearnend_changepercent', '1YP_totnetassetend_changepercent', '1YP_gftgrntsrcvd170_changepercent', '1YP_txrevnuelevied170_changepercent', '1YP_srvcsval170_changepercent', '1YP_grsinc170_changepercent', '1YP_grsrcptsrelated170_changepercent', '1YP_totgftgrntrcvd509_changepercent', '1YP_grsrcptsadmissn509_changepercent', '1YP_txrevnuelevied509_changepercent', '1YP_srvcsval509_changepercent', '1YP_subtotsuppinc509_changepercent', '1YP_totsupp509_changepercent', 'GDP2002', 'GDP2003', 'GDP2004', 'GDP2006', 'GDP2007', 'GDP2008', 'GDP2009', 'GDP2010', 'GDP2011', 'GDP2YP', 'GDP1YP', 'GDP2014', '1YP_noemplyeesw3cnt', '2YP_grsrcptspublicuse', '1YP_grsrcptspublicuse', '2YP_totassetsend', '1YP_totassetsend', '2YP_totgftgrntrcvd509', '1YP_totgftgrntrcvd509', '2014_totgftgrntrcvd509', '2YP_totfuncexpns', '1YP_totfuncexpns',  '2YP_compnsatncurrofcr', '1YP_compnsatncurrofcr', '2YP_lessdirfndrsng', '1YP_lessdirfndrsng', '1YP_officexpns', '1YP_interestamt']

    return (y_variable, imp_cols, robustscale_cols, scale_columns)

def clf_execute(dataframe, clf, y_variable, X_variables, stat_k = 0.1, 
     export_values = True):
    '''
    '''
    validation_data = dataframe[X_variables]
    validation_set = dataframe[y_variable]
    
    print(clf)
    y_pred_probs = predict_proba(validation_data)[:,1]
    precision = precision_at_k(y_test, y_pred_probs, stat_k)
    auc = auc_roc(y_test, y_pred_probs)
    recall = recall_at_k(y_test, y_pred_probs, stat_k)
    accuracy = accuracy_at_k(y_test, y_pred_probs, stat_k)
    print('RESULTS AT K = {}'.format(stat_k))
    print('    THRESHOLD: {}'.format(precision[1]))
    print('    PRECISION: {}'.format(precision[0])'
    print('    AUC - ROC: {}'.format(auc[0]))
    print('    RECALL: {}'.format(recall[0]))
    print('    ACCURACY: {}'.format(accuracy[0]))
    if export_values:
        y_pred_probs.to_csv('Validation_Result.csv')

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

def main(filename, clf_fp):
    '''
    Runs the validation.
    '''
    dataframe = acg_read.load_file(filename, index = 0)
    clf = joblib.load(clf_fp)
    # Generalize dataset column names
    new_col_names = []
    for x in dataframe.columns: 
        year = re.search('[2][0-9]*', x)
        if year != None and year.group(0) in VERIFICATION_LABELS:
            gen_year = VERIFICATION_LABELS[year.group(0)]
            new_col_names.append(x[:year.start()] + gen_year + x[year.end():])
        else:
            new_col_names.append(x)
    dataframe.columns = new_col_names
    # Drop row if missing y-variable
    dataframe = dataframe[dataframe[y_variable].notnull()]
    # Get all the necessary parameters
    y_variable, imp_cols, robustscale_cols, scale_columns = define_project_params()
    X_variables = [i for i in dataframe.columns if i != y_variable]
    # Remove any infinities, replace with missing
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    # Find any columns with missing values and impute, transform scales
    for x in X_variables:
        if len(dataframe[dataframe[x].isnull()]) > 0:
            dataframe = acg_process.impute_median(dataframe, x, keep = False)
    for col in robustscale_cols:
        if col in X_variables:
            dataframe = acg_process.robust_transform(dataframe, col, keep = False)
    for col in scale_columns:
        if col in X_variables:
            dataframe = acg_process.normalize_scale(dataframe, col = col, keep = False)
    # Execute validation
    clf_execute(dataframe, clf, y_variable, X_variables)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print('Usage: -u model.py <featuresfilename> <pickled model>')
