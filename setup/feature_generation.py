# Feature generator
# Creates all features for seismos model

import acg_read
import acg_process
import numpy as np
import pandas as pd
import sys
import math

def read_file(filename, convert_types = False, drop_duplicates = True):
    '''
    Reads csv file.
    '''
    try:
        data = acg_read.load_file(filename, index = 'EIN_x')
        ind = 'EIN_x'
    except ValueError:
        # For our implementation, some of our data had been uncleanly merged
        data = acg_read.load_file(filename, index = 'EIN')
        ind = 'EIN'
    
    # If you're getting a mixed type error for columns, it's because the 
    # dichotomous variables are incorrectly labeled 'N' and 'Y'. Fix by adding
    # convert_types = True to this function.
            
    return data
    
def generate_features(data, year1, year2, year3=None):
    '''
    Runs all feature generation code. For our purposes, we train the model
    using 2012 and 2013 data and test on 2014. In order to make the feature
    creation as generalized as possible, we allow for years to be added, so
    long as the data following the following convention:

    - All years are merged into the same dataset.
    - Columns are named: "YYYY_variablename" where YYYY is the year and
      variablename is the column label from the IRS.
    '''
    features = pd.DataFrame(index = data.index)
    features = generate_rev_fall(data, features, year1, year2)
    features = generate_YOY_rev_change(data, features, year1, year2)
    features = generate_YOY_change_payroll_taxes(data, features, year1, year2)
    features = generate_YOY_change_net_assets(data, features, year1, year2)
    #features = gen_one_year_prior_neg_revenue(data, features, year2)
    
    if year3:
        features = generate_rev_fall(data, features, year2, year3)
        features = generate_YOY_rev_change(data, features, year2, year3)
        features = generate_YOY_change_payroll_taxes(data, features, year2, year3)
        features = generate_YOY_change_net_assets(data, features, year2, year3)
        features = gen_one_year_prior_neg_revenue(data, features, year3)
        
    features = generate_missing_for_year(data, features)
    features = generate_NTEE_dummies(data, features)
    features = generate_GDP(data, features)
    features = generate_employee_number(data, features, ignore_year='2014')
    
    return features

def generate_YOY_rev_change(data, features, year1, year2, add_to_features=True):
    '''
    Generates YOY revenue change as a percentage of the year prior.
    '''
    base_year = str(year1)
    base_variable = base_year + '_totrevenue'
    second_year = str(year2)
    second_variable = second_year + '_totrevenue'

    base = pd.DataFrame(data[base_variable].dropna(axis=0))
    second = pd.DataFrame(data[second_variable].dropna(axis=0))
    # Eliminate orgs that don't have values for both years
    calc = base.join(second, how = 'inner')
    # Calculate YOY change
    calc[second_year + '_rev_change'] = (calc[second_variable] - 
     calc[base_variable]) / calc[base_variable]
    if add_to_features == False:
        return calc
    else: 
        return features.join(calc[second_year + '_rev_change'])
    
def generate_rev_fall(data, features, year1, year2, threshold = -0.2):
    '''
    Generates 0/1 variable for YOY Gross Revenue negative change from year1
    to year2. Uses threshold to determine what negative change to mark as 1 
    (e.g. if threshold is -20, will give values less than -20 a 1).
    '''
    calc = generate_YOY_rev_change(data, features, year1, year2, False)
    # Assign True to values that are below threshold
    second_year = str(year2)
    column_name = second_year + '_YOY_revenue_fell'
    calc.dropna(inplace = True)
    calc[column_name] = calc[second_year + '_rev_change'] < threshold
    # Returns features dataframe with new column
    return features.join(calc[column_name])
    
def generate_missing_for_year(data, features):
    '''
    Generates a 0/1 variable for an EIN if tot_revenue is missing from one 
    year but present in others.
    '''
    cols = []
    for x in data.columns: 
        if '_totrevenue' in x: 
            cols.append(x)
    for col in cols: 
        new = pd.notnull(data[col])
        features[col[:4] + '_missing'] = new
    
    return features
    
def generate_NTEE_dummies(data, features):
    '''
    Takes the NTEE column and converts to dummies, including one for missing 
    values.
    '''
    calc = pd.DataFrame(index=data.index)
    new = data[data['NTEE_CD'].notnull()]['NTEE_CD']
    calc = calc.join(new, how = 'left') 
    calc['NTEE_CD'] = 'NTEE_' + calc['NTEE_CD'].str.get(0) 
    calc.fillna(value = 'NTEE_Missing', inplace = True)
    rv = pd.get_dummies(calc['NTEE_CD'])
    return features.join(rv)
    
def generate_GDP(data, features):
    '''
    Transfers the GDP column from data to features.
    '''
    cols = []
    for col in data.columns: 
        if 'GDP' in col: 
            cols.append(col)
    for col in cols: 
        features[col] = data[col]
    return features
    
def generate_employee_number(data, features, ignore_year):
    '''
    Adds any employee counts that are not the test year.
    '''
    for col in data.columns: 
        if 'noemployees' in col and ignore_year not in col:
            features = features.join(calc[col])
    return features

def generate_YOY_change_payroll_taxes(data, features, year1, year2):
    '''
    Returns a feature set with yoy percent change in payroll taxes paid by org
    between year1 and year2.
    '''
    base_year = str(year1)
    base_variable = base_year + '_payrolltx'
    second_year = str(year2)
    second_variable = second_year + '_payrolltx'

    base = pd.DataFrame(data[base_variable].dropna(axis=0))
    second = pd.DataFrame(data[second_variable].dropna(axis=0))
    
    # Eliminate orgs that don't have values for both years
    calc = base.join(second, how = 'inner')
    # Calculate YOY change
    calc[second_year + '_payroll_change'] = (calc[second_variable] - 
     calc[base_variable]) / calc[base_variable]

    return features.join(calc[second_year + '_payroll_change'])
    
def generate_YOY_change_net_assets(data, features, year1, year2):
    '''
    Returns a feature set with yoy percent change in net assets 
    and log change between year1 and year2.
    '''
    base_year = str(year1)
    base_variable = base_year + '_totnetassetend'
    second_year = str(year2)
    second_variable = second_year + '_totnetassetend'

    base = pd.DataFrame(data[base_variable].dropna(axis=0))
    second = pd.DataFrame(data[second_variable].dropna(axis=0))
    
    # Eliminate orgs that don't have values for both years
    calc = base.join(second, how = 'inner')
    # Calculate YOY change
    calc[second_year + '_totnetassetend_change'] = (calc[second_variable] - 
     calc[base_variable]) / calc[base_variable]

    return features.join(calc[second_year + '_totnetassetend_change'])
    
def gen_one_year_prior_neg_revenue(data, features, year): 
    '''
    0/1 for whether year has negative revenue.
    '''
    column_name = str(year) + '_negative_revenue'
    features[column_name] = data[data[str(year) + '_totrevenue'] < 0]
    return features
    
def run(filename, new_filename, year1, num_years):
    '''
    Runs all feature generation.
    '''
    data = read_file(filename, convert_types = True)
    if num_years == 2: 
        features = generate_features(data, year1, year1 + 1)
    if num_years == 3:
        features = generate_features(data, year1, year1 + 1, year1 + 2)
    features.to_csv(new_filename)
    print "Wrote file to {}".format(new_filename)
            
if __name__ == "__main__":
    if len(sys.argv) == 5: 
        assert '.csv' in sys.argv[1]
        assert '.csv' in sys.argv[2]
        year1 = int(sys.argv[3])
        num_years = int(sys.argv[4])
        run(sys.argv[1], sys.argv[2], year1, num_years)
    else:  
        print "This program produces all features for model generation."
        print "It will write the file features data to a separate file."
        print "Number of years must be 2 or 3; all options required.\n"
        print "Usage: python feature_generation.py <DATA FILE> <WRITE FILE> <FIRST YEAR> <NUMBER OF YEARS>"
