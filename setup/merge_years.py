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

#https://www.irs.gov/Charities-&-Non-Profits/Exempt-Organizations-Business-Master-File-Extract-EO-BMF
df1=pd.read_csv('eo1.csv')
df2=pd.read_csv('eo2.csv')
df3=pd.read_csv('eo3.csv')
df4=pd.read_csv('eo4.csv')
#download the zipmsa.csv from github
dfz=pd.read_csv('zipmsa.csv')

df5=df1.append(df2, ignore_index=True).append(df3, ignore_index=True).append(df4, ignore_index=True)
df5['ZIP']=pd.DataFrame(list(df5['ZIP'].str.split('-')))
dfz['ZIP CODE']=pd.Series(dfz['ZIP CODE']).astype(str).str.zfill(5)
df=df5.merge(dfz, how='left', left_on=['ZIP','STATE'],right_on=['ZIP CODE','STATE'])
df.to_csv('BMFData.csv')

dfm=pd.read_csv('merged_data.csv')
dff=dfm.merge(df[['NAME','EIN','ZIP','MSA1','NTEE_CD']],left_on=dfm['EIN'],right_on=df['EIN'],how='left')
dff.to_csv('990.csv',index=False)
