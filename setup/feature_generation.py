# Feature generator
# Creates all features for seismos model

import acg_read
import acg_process
import numpy as np
import pandas as pd

MIXED_COLS = [4,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111, 112,113,114,115,116,119,121,122,123,124,125,126,127,128,129,130,131,132,134,135,136,137,138,139,140,145,147,150,151]

def read_file(filename, convert_types = False):
    '''
    Reads csv file.
    '''
    try:
        data = acg_read.load_file(filename, index = 'EIN')
    except ValueError:
        data = acg_read.load_file(filename, index = 'EIN_y')
    
    # If you're getting a mixed type error for columns, it's because the 
    # dichotomous variables are incorrectly labeled 'N' and 'Y'. Fix by adding
    # convert_types = True to this function.
    if convert_types:
        for i in MIXED_COLS:
            data[data.columns[i]].replace(to_replace = 'N', value=0, inplace = True)
            data[data.columns[i]].replace(to_replace = 'Y', value=1, inplace = True)
            
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
    if year3:
        features = generate_rev_fall(data, features, year2, year3)
    return features

def generate_rev_fall(data, features, year1, year2, threshold = -0.2):
    '''
    Generates 0/1 variable for YOY Gross Revenue negative change from year1
    to year2. Uses threshold to determine what negative change to mark as 1 
    (e.g. if threshold is -20, will give values less than -20 a 1).
    '''
    base_year = str(year1)
    base_variable = base_year + '_totrevenue'
    second_year = str(year2)
    second_variable = second_year + '_totrevenue'

    base = pd.DataFrame(data[base_variable])
    # Remove zero values, as these are suspicious
    base = base[base[base_variable] != 0]
    # Get second year
    second = pd.DataFrame(data[second_variable])
    second = second[second[second_variable] != 0]
    # Eliminate orgs that don't have values for both years
    calc = base.join(second, how = 'inner')
    # Calculate YOY change
    calc['change'] = (calc[second_variable] - calc[base_variable]) /    calc[base_variable]
    # Assign True to values that are below threshold
    calc[second_year + '_YOY_revenue_fell'] = calc['change'] < threshold
    # Returns features dataframe with new column
    return features.join(calc[second_year + '_YOY_revenue_fell'])

def generate_YOY_rev_change(data, features, year1, year2):
    pass

def generate_missing_for_year():
    '''
    Generates a 0/1 variable for an EIN if missing from one year but present
    in others.
    '''
    pass

if __name__ == "__main__":
    pass
