# CAPP 30254: Machine Learning for Public Policy
# ALDEN GOLAB
# ML Pipeline
# 
# File read functions

import pandas as pd

def load_file(filename, index = None):
    '''
    Reads file with column index  and returns a Pandas dataframe. If index
    name is missing for UID, use [0] to refer to the first column even if 
    it is unnamed. Returns pandas dataframe. Index option can only be used
    with .csv files.

    Currently only has options for csv, json, and dat. More to come.
    '''
    if 'csv' in filename:
        if index != None: 
            return pd.read_csv(filename, index_col = index)
        else: 
            return pd.read_csv(filename)
    if 'json' in filename: 
        return pd.read_json(filename)
    if 'dta' in filename: 
        return pd.read_stata(filename)
    else: 
        print ('Input currently not built for this filetype')

def write_csv(dataframe, filename):
    '''
    Writes dataframe to csv.
    '''
    dataframe.to_csv(filename)
    print('Wrote data to {}'.format(filename))