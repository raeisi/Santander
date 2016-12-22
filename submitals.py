import pandas as pd
import numpy as np
import os, sys
import glob

path='C:/Users/Shahrouz/Google Drive/Codes/Kaggle/Submitals'

def pr(df, l=20):
    if l<1000:
        print(df.head(l).to_string().replace(' 0 ', '   '))#print(df.head(l).iloc[:,0:47].to_string().replace(' 0 ', '   '))
        try:
            print(df.name+ 'The DF length is:',len(df))
        except:
            print('The DF length is:',len(df))
    else:
        print('so many lines of rows are not possible to print. Printing first 1000 rows.')
        print(df.head(1000).to_string().replace(' 0 ', '   '))#print(df.head(l).iloc[:,0:47].to_string().replace(' 0 ', '   '))

smbs=pd.DataFrame()

for filename in glob.glob(os.path.join(path, '*.csv')):
  print(filename)
  smb=pd.read_csv(filename, index_col=0)
  pr(smb[(smb.added_products!='')&(smb.added_products.notnull())])
  smb=smb[(smb.added_products!='')&(smb.added_products.notnull())]
  smbs=smbs.append(smb[smb.added_products!=''])

smbs=smbs.groupby(smbs.index).first()
smbs.sort_index(inplace=True)
pr(smbs)

df=pd.read_csv('C:/Users/Shahrouz/Google Drive/Codes/Kaggle/sub_xgb_new', index_col=0)
df['added_products']=''
# for this in smbs.index:
#   df.loc[this]=smbs.loc[this]

df.update(smbs)
pr(df,999)
df.to_csv('C:/Users/Shahrouz/Google Drive/Codes/Kaggle/Submital05-0-1.csv')
