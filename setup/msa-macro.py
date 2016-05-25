import pandas as pd
dfz=pd.read_csv('msazip.csv')
dfm=pd.read_csv('gmpPCGDP.csv')
dfz['GeoFIPS']=dfz['MSA No.'].astype(str)
dfz['GeoFIPS']=dfz['GeoFIPS'].apply(lambda x:x[:5])
df=dfz.merge(dfm,left_on='GeoFIPS',right_on='GeoFIPS',how='left')
df.to_csv('zipmsa.csv',index=False)


