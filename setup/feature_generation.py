# Feature generator
# Creates all features for seismos model
# 5-17-16

import acg_read
import acg_process
import numpy as np
import pandas as pd


def read_file(filename):
    '''
    Reads csv file.
    '''
    return acg_read.load_file(filename, index = 'EIN')

def generate_features(data, year1, year2, year3=None, test_year=None):
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
    features = generate_label(data, features, 2012, 2013)
    pass

def generate_rev_fall(data, features, year1, year2, threshold = -20):
    '''
    Generates 0/1 variable for YOY Gross Revenue negative change from year1
    to year2 and year_2 to test year. Uses threshold to determine what
    negative change to mark as 1 (e.g. if threshold is -20, will give
    values less than -20 a 1).
    '''
    base_year = str(year1)
    base_variable = base_year + '_totrevenue'
    second_year = str(year2)
    second_variable = second_year + '_totrevenue'

    base = pd.DataFrame(data[base_variable])
    # Change zero values to a small number to avoid inf
    base.replace(to_replace = 0, value = .01)
    second = pd.DataFrame(data[second_variable])
    # Eliminate orgs that don't have values for both years
    calc = base.dropna().join(second.dropna(), how = 'inner')
    # Calculate YOY change
    calc['change'] = (calc[second_variable] - calc[base_variable]) / calc[base_variable]
    # Calculate index
    calc['change_index'] = (calc['change'] - calc['change'].mean()) / calc['change'].mean()
    # Assign True to values that are below threshold
    calc[second_year + '_YOY_revenue_fell'] = calc['change_index'] < threshold
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
