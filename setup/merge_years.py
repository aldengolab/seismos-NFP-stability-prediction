import pandas as pd

NONZERO_COLUMNS = ['tot_revenue']

#### read in files
df2012 = pd.read_table("py12_990.dat", sep = ' ')
df2013 = pd.read_table("py13_990.dat", sep = ' ')
df2014 = pd.read_table("py14_990.dat", sep = ' ')
df2015 = pd.read_table("15eofinextract990.dat", sep = ' ')

#### drop duplicates and non-positive values for revenue
df2012.drop_duplicates(subset='EIN', keep=False, inplace=True)
df2013.drop_duplicates(subset='EIN', keep=False, inplace=True)
df2014.drop_duplicates(subset='EIN', keep=False, inplace=True)
df2015.drop_duplicates(subset='EIN', keep=False, inplace=True)
df2012 = df2012[df2012['totrevenue'] > 0]
df2013 = df2013[df2013['totrevenue'] > 0]
df2014 = df2014[df2014['totrevenue'] > 0]
df2015 = df2015[df2015['totrevenue'] > 0]

#### get list of columns
cols12 = df2012.columns
cols12 = list(cols12)
mergecols12 = ["2012_" + col for col in cols12 if col != "EIN"]
mergecols12.insert(0,"EIN")
df2012.columns = mergecols12

cols13 = df2013.columns
cols13 = list(cols13)
mergecols13 = ["2013_" + col for col in cols13 if col != "EIN"]
mergecols13.insert(0,"EIN")
df2013.columns = mergecols13

cols14 = df2014.columns
cols14 = list(cols14)
mergecols14 = ["2014_" + col for col in cols14 if col != "EIN"]
mergecols14.insert(0,"EIN")
df2014.columns = mergecols14

df2015 = df2015[['EIN', 'totrevenue']]
cols15 = df2015.columns
cols15 = list(cols15)
mergecols15 = ["2015_" + col for col in cols15 if col != "EIN"]
mergecols15.insert(0,"EIN")
df2015.columns = mergecols15

#####merge
df12_13 = pd.merge(df2012, df2013,  how = 'outer', on = 'EIN')
dfmerge = pd.merge(df12_13, df2014, how = 'outer', on = 'EIN')
dffullmerge = pd.merge(dfmerge,df2015, how = 'outer', on = 'EIN')

print('Merged file has {} rows, saved to merged_data.csv'.format(len(dffullmerge)))
dffullmerge.to_csv("merged_data.csv")

df = pd.read_csv('BMFData.csv')
dfm = pd.read_csv('merged_data.csv')

dff = dfm.merge(df[['NAME','EIN','ZIP','MSA No.','NTEE_CD', 'GDP2002',
 'GDP2003', 'GDP2004', 'GDP2005', 'GDP2006', 'GDP2007', 'GDP2008', 'GDP2009',
 'GDP2010', 'GDP2011', 'GDP2012', 'GDP2013', 'GDP2014', 'GDP2015']],
 on='EIN', how='left')
df_15 = pd.merge(df2015, dff, on ='EIN' ,how='left')

#### Replace Y and N with intelligible values
for i in dff.columns:
    try:
        dff[i].replace(to_replace = 'N', value=0, inplace = True)
        dff[i].replace(to_replace = 'Y', value=1, inplace = True)
    except:
        break

for i in df_15.columns:
    try:
        df_15[i].replace(to_replace = 'N', value=0, inplace = True)
        df_15[i].replace(to_replace = 'Y', value=1, inplace = True)
    except:
        break

### Write out to file
dff.to_csv('2014_full_merge_dataset.csv',index=False)
df_15.to_csv('2015_full_merge_dataset.csv', index=False)
print('File saved to 2014_full_merge_dataset.csv and 2015_full_merge_dataset.csv')
