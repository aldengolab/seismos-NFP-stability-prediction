# Feature generator
# Creates all features for seismos model
# 5-17-16

from acg_read import *
from acg_process import *

LABEL_THRESHOLD = .2

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
    pass

def generate_label(data, features, year1, year2, test_year):
    '''
    Generates 0/1 variable for YOY Gross Revenue negative change from year1
    to year2 greater than LABEL_THRESHOLD.
    '''
    base_year = str(year1)
    second_year = str(year2)

    base = numpy.array(data[base_year + 'totrevenue'])
    second = numpy.array(data[second_year + 'totrevenue'])
    change = (second - base) / base
    change_index = (change / np.average(change)) * 100

    YOY_revenue_fell = change_index < -20
    YOY_revenue_fell = pd.DataFrame(YOY_revenue_fell)
    pd.concat([features, YOY_revenue_fell])

def generate_missing_for_year():
    '''
    Generates a 0/1 variable for an EIN if missing from one year but present
    in others.
    '''
    pass

if __name__ == "__main__":
    pass
