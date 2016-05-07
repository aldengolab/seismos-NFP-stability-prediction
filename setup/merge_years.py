import pandas as pd

#### read in files
df2012 = pd.read_table("py12_990.dat", sep = ' ')
df2013 = pd.read_table("py13_990.dat", sep = ' ')
df2014 = pd.read_table("py14_990.dat", sep = ' ')
df2015 = pd.read_table("15eofinextract990.dat", sep = ' ')

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

#####merge
df12_13 = pd.merge(df2012, df2013,  how = 'outer', on = 'EIN')
dfmerge = pd.merge(df12_13, df2014, how = 'outer', on = 'EIN')
print(len(dfmerge))
dfmerge.to_csv("merged_data.csv")
