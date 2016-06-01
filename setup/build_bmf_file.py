### This file creates the BMFData.csv for use in the merge_years.py file

import pandas as pd

dfz=pd.read_csv('msazip.csv')
dfm=pd.read_csv('gmpPCGDP.csv')
dfz['GeoFIPS']=dfz['MSA No.'].astype(str)
dfz['GeoFIPS']=dfz['GeoFIPS'].apply(lambda x:x[:5])
df=dfz.merge(dfm,left_on='GeoFIPS',right_on='GeoFIPS',how='left')
df.to_csv('zipmsa.csv',index=False)

#https://www.irs.gov/Charities-&-Non-Profits/Exempt-Organizations-Business-Master-File-Extract-EO-BMF

df1=pd.read_csv('eo1.csv')
df2=pd.read_csv('eo2.csv')
df3=pd.read_csv('eo3.csv')
df4=pd.read_csv('eo4.csv')

#download the zipmsagdp.csv from github data folder

dfz=pd.read_csv('zipmsa.csv')

df5=df1.append(df2, ignore_index=True).append(df3, ignore_index=True).append(df4, ignore_index=True)
df5['ZIP']=pd.DataFrame(list(df5['ZIP'].str.split('-')))
dfz['ZIP CODE']=pd.Series(dfz['ZIP CODE']).astype(str).str.zfill(5)
df=df5.merge(dfz, how='left', left_on=['ZIP','STATE'],right_on=['ZIP CODE','STATE'])
df.to_csv('BMFData.csv')
