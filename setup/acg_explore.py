# CAPP 30254: Machine Learning for Public Policy
# ALDEN GOLAB
# ML Pipeline
# 
# Data exploration functions.

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy

def summarize(dataframe, column = None, plots = False, write = False):
    '''
    Takes dataframe and returns summary statistics for each numerical column 
    as a pandas dataframe. Optional print plots of all columns; to do so, 
    input 'Y' for plots. Optional write summaries to file, input 'Y'.
    '''
    summaries = []

    if column == None:
        for col in dataframe.columns:
            summaries.append(get_summary(dataframe, col))        
    else:
        summaries.append(get_summary(dataframe, column))

    if write == True:
        for summ in summaries:
            summ[2].to_csv('{}_summary.csv'.format(summ[1]))
        print('Wrote descriptive statistics to file.')
    if plots == True:
        plot(dataframe)

    return summaries

def get_summary(dataframe, col):
    '''
    Produces column summary.
    '''
    if dataframe[col].dtype != object and 'disc' not in col:
        summary = ('float64', col, get_cont_summary(dataframe, col))
    else:
        summary = ('object', col, get_cat_summary(dataframe, col)) 

    return summary

def get_cont_summary(dataframe, col):
    '''
    Produces summary of continuous data.
    '''
    summary = dataframe[col].describe()
    # Get mode and append
    mode = pd.Series(dataframe[col].mode())
    mode = mode.rename({i: 'mode' + str(i) for i in range(len(mode))})
    summary = summary.append(mode)
    # Get median and append
    median = pd.Series(dataframe[col].median())
    median = median.rename({median.index[0]: 'median'})
    summary = summary.append(median)
    # Count missing values and append
    missing = pd.Series(dataframe[col].isnull().sum())
    missing = missing.rename({missing.index[0]: 'missing'})
    summary = summary.append(missing)

    return summary

def get_cat_summary(dataframe, col):
    '''
    Produces summary of categorical data.
    '''
    summary = dataframe[col].describe()
    # Count missing values and append
    missing = pd.Series(dataframe[col].isnull().sum())
    missing = missing.rename({missing.index[0]: 'missing'})
    summary = summary.append(missing)

    return summary

def plot(dataframe, col = None):
    '''
    Plots histograms or bar charts for each column and saves to file within
    the same directory.
    '''
    histogram = ['object', 'category']
    if col: 
        if dataframe[col].dtype.name in histogram:
            dataframe[col].value_counts().plot(kind = 'bar')
            plt.suptitle('Bar Graph for {}'.format(col), fontsize = 14)
            plt.savefig('{}.png'.format(col))
            print('Saved figure as {}.png'.format(col))
            plt.close()
        else:
            dataframe[col].hist()
            plt.suptitle('Histogram for {}'.format(col), fontsize = 14)
            plt.savefig('{}.png'.format(col))
            print('Saved figure as {}.png'.format(col))
            plt.close()
    else:   
        for column in dataframe.columns:
            if dataframe[column].dtype.name in histogram:
                dataframe[column].value_counts().plot(kind = 'bar')
                plt.suptitle('Bar Graph for {}'.format(column), fontsize = 14)
                plt.savefig('{}.png'.format(column))
                print('Saved figure as {}.png'.format(column))
                plt.close()
            else:
                dataframe[column].hist()
                plt.suptitle('Histogram for {}'.format(column), fontsize = 14)
                plt.savefig('{}.png'.format(column))
                print('Saved figure as {}.png'.format(column))
                plt.close()

def histogram(dataframe, col, bins = 10, write = None):
    '''
    Plots a histogram for the given column.
    '''
    dataframe[col].hist(bins = bins)
    plt.suptitle('Histogram for {}'.format(col), fontsize = 14)
    if write != None:
        plt.savefig('{}.png'.format(col))
        print('Saved figure as {}.png'.format(col))
        plt.close()
    if write == None:
        plt.show()
    

def pairwise_correlation(dataframe, cols = []):
    '''
    Produces a pairwise correlation matrix for the columns provided. If no
    columns are provided, will execute on all columns.
    '''
    if len(cols) > 0:
        data = dataframe[cols]
    else:
        data = dataframe
    if len(cols) != 1:
        return data.corr() 

