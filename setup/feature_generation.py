# Feature generator
# Creates all features for seismos model

import acg_read
import acg_process
import numpy as np
import pandas as pd
import sys
import math

# For our dataset, we found that these columns needed extra attention. See 
# convert_types in read_file below.
MIXED_COLS = [4,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111, 112,113,114,115,116,119,121,122,123,124,125,126,127,128,129,130,131,132,134,135,136,137,138,139,140,145,147,150,151]

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
    if convert_types:
        for i in MIXED_COLS:
            data[data.columns[i]].replace(to_replace = 'N', value=0, inplace = True)
            data[data.columns[i]].replace(to_replace = 'Y', value=1, inplace = True)
    if drop_duplicates: 
        data = data.reset_index().drop_duplicates(subset = ind, keep = 
         False).set_index(ind)
    else:
        data = data.reset_index().drop_duplicates(subset = ind).set_index(ind)
            
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
    
    if year3:
        features = generate_rev_fall(data, features, year2, year3)
        features = generate_YOY_rev_change(data, features, year2, year3)
        
    features = generate_missing_for_year(data, features)
    features = generate_NTEE_dummies(data, features)
    features = generate_GDP(data, features)
    return features

def generate_YOY_rev_change(data, features, year1, year2, add_to_features=True):
    '''
    Generates raw YOY revenue change as a percentage of the year prior.
    '''
    base_year = str(year1)
    base_variable = base_year + '_totrevenue'
    second_year = str(year2)
    second_variable = second_year + '_totrevenue'

    base = pd.DataFrame(data[base_variable])
    # Remove zero values, as these are suspicious
    base = base[base[base_variable] != 0].dropna(axis=0)
    # Get second year
    second = pd.DataFrame(data[second_variable])
    second = second[second[second_variable] != 0].dropna(axis=0)
    # Eliminate orgs that don't have values for both years
    calc = base.join(second, how = 'inner')
    # Calculate YOY change
    calc[second_year + '_rev_change'] = (calc[second_variable] - 
     calc[base_variable]) / calc[base_variable]
    if add_to_features == False:
        return calc
    else: 
        calc[second_year + '_log_rev_change'] = np.log(calc[second_year + '_rev_change'])
        return features.join(calc[second_year + '_log_rev_change'])
    
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
