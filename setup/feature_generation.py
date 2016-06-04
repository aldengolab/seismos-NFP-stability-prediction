# Feature generator
# Creates all features for seismos model

import acg_read
import numpy as np
import pandas as pd
import sys
import math

def read_file(filename):
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
    # Generate YOY features
    features = pd.DataFrame(index = data.index)
    features = generate_NTEE_dummies(data, features)
    features = generate_rev_fall(data, features, year1, year2)
    features = generate_rev_fall(data, features, year1+1, year2+1)
    features = generate_YOY_rev_change(data, features, year1, year2)
    features = generate_YOY_change_payroll_taxes(data, features, year1, year2)
    features = generate_YOY_change_net_assets(data, features, year1, year2)
    # Generate year specific features
    features = generate_member_income(data, features, year1)
    features = generate_member_income(data, features, year2)
    features = generate_all_percent_of_revenue(data, features, year1)
    features = generate_all_percent_of_revenue(data, features, year2)
    features = generate_fundraising_ROI(data, features, year1)
    features = generate_fundraising_ROI(data, features, year2)
    features = generate_ratio(data, features, '_totliabend', '_totnetassetend', '_debtassetratio', year1)
    features = generate_ratio(data, features, '_totliabend', '_totnetassetend', '_debtassetratio', year2)
    features = generate_ratio(data, features, '_totsupp509', '_totrevenue', '_supportrevratio', year1)
    features = generate_ratio(data, features, '_totsupp509', '_totrevenue', '_supportrevratio', year2)
    features = generate_gov_support(data, features, year1)
    features = generate_gov_support(data, features, year2)
    features = generate_YOY_changepercent(data, features, year1, year2)

    if year3:
        # YOY change features
        features = generate_rev_fall(data, features, year2, year3)
        features = generate_YOY_rev_change(data, features, year2, year3)
        features = generate_YOY_change_payroll_taxes(data, features, year2, year3)
        features = generate_YOY_change_net_assets(data, features, year2, year3)
        # Year specific features
        features = generate_member_income(data, features, year3)
        features = generate_all_percent_of_revenue(data, features, year3)
        features = generate_fundraising_ROI(data, features, year3)
        features = generate_ratio(data, features, '_totliabend', '_totnetassetend', '_debtassetratio', year3)
        features = generate_ratio(data, features, '_totsupp509', '_totrevenue', '_supportrevratio', year3)
        features = generate_gov_support(data, features, year3)
        features = generate_YOY_changepercent(data, features, year2, year3)
  
    if year3:
        features = generate_GDP(data, features,list(range(2002,year3+1)))
        features = generate_missing_for_year(data, features,[year1,year2,year3])
        features = copy_features(data, features, ['noemplyeesw3cnt','grsrcptspublicuse','grsincmembers', 'totassetsend',  'totgftgrntrcvd509', 'totfuncexpns', 'compnsatncurrofcr','totfuncexpns', 'lessdirfndrsng', 'officexpns', 'interestamt'], [year1,year2,year3])
    else:
        features = generate_GDP(data, features,list(range(2002,year2+1)))
        features = generate_missing_for_year(data, features,[year1,year2])
        features = copy_features(data, features, ['noemplyeesw3cnt','grsrcptspublicuse','grsincmembers', 'totassetsend',  'totgftgrntrcvd509', 'totfuncexpns', 'compnsatncurrofcr','totfuncexpns', 'lessdirfndrsng', 'officexpns', 'interestamt'], [year1, year2])
    print 'Finished feature generation. The result dataframe has the shape',features.shape
    return features

def generate_YOY_changepercent(data, features, year1, year2):
    '''
    Returns a feature set with yoy percent change in payroll taxes paid by org
    between year1 and year2.
    '''
    '''
    ['grsrcptspublicuse','totassetsend','totliabend','totfuncexpns','compnsatncurrofcr','grsincmembers'
    ,'totprgmrevnue','invstmntinc','netrntlinc','netgnls','gnlsecur','netincfndrsng','lessdirfndrsng'
    ,'netincsales','srvcsval170','txrevnuelevied170','unsecurednotesend']
    '''
    for v in features.columns:
        l=v[5:]
        try:
            base_year = str(year1)
            base_variable = base_year +'_'+l
            second_year = str(year2)
            second_variable = second_year +'_'+l

            base = pd.DataFrame(features[base_variable].dropna(axis=0))
            second = pd.DataFrame(features[second_variable].dropna(axis=0))
            
            # Eliminate orgs that don't have values for both years
            calc = base.join(second, how = 'inner')
            calc=calc[calc[base_variable] != 0]
            # Calculate YOY change
            calc[second_varialbe+'_changepercent'] = (calc[second_variable] - calc[base_variable]) / calc[base_variable]
            features=features.join(calc[second_variable+'_changepercent'])
            print second_variable," percent change was calculated."
        except:
            continue
    
    li=['elf', 's501c3or4947a1cd', 'schdbind', 'politicalactvtscd', 'lbbyingactvtscd', 'subjto6033cd', 'dnradvisedfundscd', 'prptyintrcvdcd', 'maintwrkofartcd', 'crcounselingqstncd', 'hldassetsintermpermcd', 'rptlndbldgeqptcd', 'rptinvstothsecd', 'rptinvstprgrelcd', 'rptothasstcd', 'rptothliabcd', 'sepcnsldtfinstmtcd', 'sepindaudfinstmtcd', 'inclinfinstmtcd', 'operateschools170cd', 'frgnofficecd', 'frgnrevexpnscd', 'frgngrntscd', 'frgnaggragrntscd', 'rptprofndrsngfeescd', 'rptincfnndrsngcd', 'rptincgamingcd', 'operatehosptlcd', 'hospaudfinstmtcd', 'rptgrntstogovtcd', 'rptgrntstoindvcd', 'rptyestocompnstncd', 'txexmptbndcd', 'invstproceedscd', 'maintescrwaccntcd', 'actonbehalfcd', 'engageexcessbnftcd', 'awarexcessbnftcd', 'loantofficercd', 'grantoofficercd', 'dirbusnreltdcd', 'fmlybusnreltdcd', 'servasofficercd', 'recvnoncashcd', 'recvartcd', 'ceaseoperationscd', 'sellorexchcd', 'ownsepentcd', 'reltdorgcd', 'intincntrlcd', 'orgtrnsfrcd', 'conduct5percentcd', 'compltschocd', 'wthldngrulescd', 'filerqrdrtnscd', 'unrelbusinccd', 'filedf990tcd', 'frgnacctcd', 'prohibtdtxshltrcd', 'prtynotifyorgcd', 'filedf8886tcd', 'solicitcntrbcd', 'exprstmntcd', 'providegoodscd', 'notfydnrvalcd', 'filedf8282cd', 'fndsrcvdcd', 'premiumspaidcd', 'filedf8899cd', 'filedf1098ccd', 'excbushldngscd', 's4966distribcd', 'distribtodonorcd', 'filedlieuf1041cd', 'qualhlthplncd', 'rcvdpdtngcd', 'filedf720cd', 'subseccd', 'nonpfrea', 'tax_pd', 'prgmservcode2acd', 'prgmservcode2bcd', 'prgmservcode2ccd', 'prgmservcode2dcd', 'prgmservcode2ecd', 'miscrev11acd', 'miscrev11bcd', 'miscrev11ccd']

    for v in data.columns:
        if v[5:] in li: 
            pass
        else:
            l=v[5:]
            try:
                base_year = str(year1)
                base_variable = base_year +'_'+l
                second_year = str(year2)
                second_variable = second_year +'_'+l

                base = pd.DataFrame(data[base_variable].dropna(axis=0))
                second = pd.DataFrame(data[second_variable].dropna(axis=0))
                
                # Eliminate orgs that don't have values for both years
                calc = base.join(second, how = 'inner')
                calc=calc[calc[base_variable] != 0]
                # Calculate YOY change
                calc[second_variable+'_changepercent'] = (calc[second_variable] - calc[base_variable]) / calc[base_variable]
                features=features.join(calc[second_variable+'_changepercent'])
                print second_variable," percent change was calculated."
            except:
                continue
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

def copy_features(data, features, variables, years):
    '''
    Copy selected variables from the selected years to the  features dataframe
    '''
    cols = []
    for x in variables:
        for year in years:
            cols.append(str(year) + "_" + x)
    for col in cols:
        try:
            features[col] = data[col]
            print col, ' successfully copied to features'
        except:
            print col ," was not copied. Check to see if  exists for given year"
    return features

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
    print column_name,"was calculated"
    return features.join(calc[column_name])

def generate_missing_for_year(data, features, years):
    '''
    Generates a 0/1 variable for an EIN if tot_revenue is missing from one
    year but present in others.
    '''
    cols = []
    for year in years:
        cols.append(str(year) + '_totrevenue')
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

def generate_GDP(data, features, years):
    '''
    Transfers the GDP column from data to features.
    '''
    cols = []
    for year in years:
        cols.append('GDP'+str(year))
    for col in cols:
        try:
            features[col] = data[col]
            print col, ' successfully copied to features'
        except:
            print col ," was not copied. Check to see if  exists for given year"
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
    
def generate_member_income(data, features, year):
    '''
    0/1 for whether organization reports having member receipts.
    '''
    column = str(year) + '_' + 'grsincmembers'
    features = features.join(data[column] > 0)
    return features
    
def generate_all_percent_of_revenue(data, features, year):
    '''
    Generates all features that use a percent of revenue for a year.
    '''
    columns = {'totprgmrevnue': 'programs', 'invstmntinc': 'investments', 'netrntlinc': 'rental', 'netgnls':'assets_sale', 'netincfndrsng': 'fundraising', 'grsalesinvent': 'inventory_sale'}
    
    for col in columns.keys():
        new, dummy = percent_of_x(data, col, year)
        new.name = str(year) + '_' + columns[col] + '_perofrev'
        dummy.name = new.name + '_isnegative'
        features = features.join(new)
        features = features.join(dummy)
    return features
    
def generate_fundraising_ROI(data, features, year):
    '''
    Generates ROI for fundraising for a given year.
    '''
    f_cost = str(year) + '_lessdirfndrsng'
    f_return = str(year) + '_netincfndrsng'
    rv = (data[f_return].dropna(axis=0) / data[f_cost]).dropna(axis=0)
    rv.name = str(year) + '_fundraisingROI'
    return features.join(rv)

def generate_ratio(data, features, numerator, denominator, name, year):
    '''
    Generates debt asset ratio for a given year.
    '''
    d_col = str(year) + numerator
    a_col = str(year) + denominator
    rv = (data[d_col].dropna() / data[data[a_col] > 0][a_col].dropna()).dropna()
    rv.name = str(year) + name
    return features.join(rv)
    
def percent_of_x(data, column, year, denom = '_totrevenue'):
    '''
    Calculates the percent of revenue for an organization from the given
    column for the given year.
    
    Returns both the calculation and a 0/1 column for those with percent 
    values less than 1 (meaning they had negative denom or column). 
    '''
    column_ref = str(year) + '_' + column
    column_rev = str(year) + denom
    rv = (data[column_ref].dropna(axis=0) / data[column_rev]).dropna(axis=0)
    return (rv[rv >= 0], rv < 0)

def generate_gov_support(data, features, year):
    '''
    Generates the % of government support coming from taxes & services. 
    '''
    columns = {'srvcsval509': '_persupp_govservices', 'txrevnuelevied509': '_persupp_govtaxes'}
    for col in columns.keys():
        new = percent_of_x(data, col, year, denom = '_totsupp509')[0]
        new.name = str(year) + '_' + columns[col]
        features = features.join(new)
    return features


def run(filename, new_filename, year1, num_years):
    '''
    Runs all feature generation.
    '''
    data = read_file(filename)
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
