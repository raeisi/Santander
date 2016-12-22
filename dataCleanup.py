# Data Clean up for Santander Product Recommendation on Kaggle
# Most of the code is based on  Alan (AJ) Pryor, Jr. post on below link with minor modification and some add ups.
# https://www.kaggle.com/apryor6/santander-product-recommendation/detailed-cleaning-visualization-python

# There is absolutly no guarantee what so ever that the code work for any other system.
# You may use it on your own risk  with no warranty from author what so ever
# Open Source MIT Licence applies https://opensource.org/licenses/MIT

# Python 3 Codes on Windows Machine
# Paths should be revised for Linux machines

# Pandas, Numpy and Seaborn require on Python 3 64 bit
# Download  them from http://www.lfd.uci.edu/~gohlke/pythonlibs/
# Then install it using $python -m install pandas-0.19.1-cp35-cp35m-win_amd64.whl
# And $python -m install  tables-3.3.0-cp35-cp35m-win_amd64

# Download  scipy and seaborn from  http://www.lfd.uci.edu/~gohlke/pythonlibs/#pytables
# Then install like  $ python -m pip install seaborn-0.7.1-py2.py3-none-any.whl
# and then    $ python -m pip install seaborn-0.7.1-py2.py3-none-any.whl
# if you still get an error for     ImportError: cannot import name 'NUMPY_MKL'
# like $python -m pip install "numpy-1.12.0b1+mkl-cp35-cp35m-win_amd64.whl"
# In General I suggest to imstall below files from same website.
# geopy, lxml, seaborn, matplotlib, scipy, pymssql, pyodbc

import numpy as np
import pandas as pd
import datetime
import time
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt

# Paths should be revised for Linux machines
inputPath           = "YOUR INPUT PATH"
outputPath          = "YOUR OUPUT PATH"


# Reading @name and @n from command line
name=(sys.argv)[1]
n=(sys.argv)[2:]

#Setting Name and n manually for running codes on console
name            = 'train_ver2'
n                   = 100000

# Rading input.csv file
print(50*'#')
print(datetime.datetime.now())
start = time.time()
print('Reading', name+'.csv')

# Reading CSV Data to DataFrame
# I have run this on a windows machine 38GB RAM and  64bit with Python and Pandas 64bit with no problem

df                   = pd.read_csv(inputPath+name+'.csv',dtype={"sexo":str,
                                                    "ind_nuevo":str,
                                                    "ult_fec_cli_1t":str,
                                                    "indext":str})

df.name='The Original dataFrame from Cleaned Train_Ver2.csv file.'

print(df.memory_usage().sum())
end = time.time()
print('time spent =', end-start)

# Printing Dataframe on screen
def PrDf(df, n=len(df)):
  if n<1000:
    print(df.head(n).iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46]].to_string().replace(' 0 ', '   '))
    try:
      print(df.name+ 'The DF length is:',len(df))
    except:
      print('The DF length is:',len(df))
  else:
    print('so many lines of rows are not possible to print. Please provide with smaller no max 1000 lines as a secons arg to PrDf function.')


#Usage example when you want to print your sample data only where @renta is null
#Example PrDf(sdf[sdf.renta.isnull()],100)
PrDf(df, 10)

### If you do not havve enough RAM on your machine to read the CSV file in one go try to use below codes. It might need some editting to suit you well.
# c=pd.read_csv(inputPath+name+'.csv', low_memory=False, chunksize=10000, dtype={"sexo":str,
#                                                     "ind_nuevo":str,
#                                                     "ult_fec_cli_1t":str,
#                                                     "indext":str})
# df=pd.concat(c, ignore_index=True)

print(50*'#')
print(datetime.datetime.now())
start = time.time()
print('Cleaning Data')

#Converting fecha_dato and fecha_alta to datetime
df["fecha_dato"] = pd.to_datetime(df["fecha_dato"],format="%Y-%m-%d")
df["fecha_alta"] = pd.to_datetime(df["fecha_alta"],format="%Y-%m-%d")
df["fecha_dato"].unique()

#Creating month column with number from 1 to 17 with respected month value in 1.5 years data recoring
df["month"] = (pd.DatetimeIndex(df["fecha_dato"]).year-2015)*12+pd.DatetimeIndex(df["fecha_dato"]).month
df["age"]   = pd.to_numeric(df["age"], errors="coerce")

# Here is some data cleaning technics copy paste from original article of Alan (AJ) Pryor, Jr. post on below link with minor modification and some add ups.
# https://www.kaggle.com/apryor6/santander-product-recommendation/detailed-cleaning-visualization-python
#####################################################################################
df=df.loc[df.age.notnull()]
df.age = df["age"].astype(int)
df.antiguedad = pd.to_numeric(df.antiguedad,errors="coerce")

df.loc[df.nomprov=="CORU\xc3\x91A, A","nomprov"] = "CORUNA, A"

df.loc[df.nomprov.isnull(),"nomprov"] = "UNKN"

df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
df.loc[df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0

string_data = df.select_dtypes(include=["object"])
missing_columns = [col for col in string_data if string_data[col].isnull().any()]
for col in missing_columns:
    print("Unique values for {0}:\n{1}\n".format(col,string_data[col].unique()))

del string_data

df.drop(["tipodom","cod_prov"],axis=1,inplace=True)

df.loc[df.indfall.isnull(),"indfall"] = "N"
df.loc[df.tiprel_1mes.isnull(),"tiprel_1mes"] = "A"
df.tiprel_1mes = df.tiprel_1mes.astype("category")

# As suggested by @StephenSmith
map_dict = { 1.0  : "1",
            "1.0" : "1",
            "1"   : "1",
            "3.0" : "3",
            "P"   : "5",
            3.0   : "3",
            2.0   : "2",
            "3"   : "3",
            "2.0" : "2",
            "4.0" : "4",
            "4"   : "4",
            "2"   : "2"}

df.indrel_1mes.fillna("P",inplace=True)
df.indrel_1mes = df.indrel_1mes.apply(lambda x: map_dict.get(x,x))
df.indrel_1mes = df.indrel_1mes.astype("category")

unknown_cols = [col for col in missing_columns if col not in ["indfall","tiprel_1mes","indrel_1mes"]]
for col in unknown_cols:
    df.loc[df[col].isnull(),col] = "UNKN"

feature_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values
for col in feature_cols:
    df[col] = df[col].astype(int)

df.indrel=df.indrel.astype(int)
df.ind_actividad_cliente =df.ind_actividad_cliente .astype(int)

#Renaming Columns to English Abbreviations to take less space when printing datafram usinf PrDf function

df.columns=['rep_date', 'cust_id', 'emp', 'res', 'sx', 'ag', 'join_date', 'new', 'csn', 'prm', 'lsdp', 'ct1', 'cr1', 'isr', 'isf', 'spi', 'jch', 'psd', 'provna', 'act', 'inco', 'segm', 'sav', 'gua', 'cur', 'der', 'pay', 'jon',  'mpa', 'par', 'pap', 'std', 'mts', 'ltd', 'e-a', 'fun', 'mor', 'pen', 'loa', 'tax', 'crc', 'sec', 'hom', 'pyr', 'npn', 'did', 'month']

#Changing each feature to a specific number to be more recognizable when printign using PrDf Function
for i in range(22,46):
    df.iloc[:,i][df.iloc[:,i]==1]=i-21

#changing segments to numeric integer values
df.loc[df.segm=='02 - PARTICULARES',"segm"] = '2'
df.loc[df.segm=='03 - UNIVERSITARIO',"segm"] = '3'
df.loc[df.segm=='01 - TOP',"segm"] = '1'
df.loc[df.segm=='UNKN',"segm"] = '4'
df.segm=df.segm.astype(int)

#Converting some 0,1 or S,N values to bool type
df.new=df.new.astype(int)
df.new=df.new.astype(bool)

d = {'S': True, 'N': False, 'Y': True, 'N': False, 1:True, 0: False, '1': True, '0': False}
df.isr=df.isr.map(d)
df.isf=df.isf.map(d)
df.act=df.act.map(d)

print(df.memory_usage().sum())
end = time.time()
print('time spent =', end-start)

#Printing DF usinf PrDf Function
PrDf(df, 100)

#Writing DataFrame to Pickle and CSV format for futre call up
print(50*'#')
print(datetime.datetime.now())
start = time.time()
print('Writing Pickle and CSV file of the Cleaned Data')
df.to_pickle(inputPath+'Output/'+'cleaned_train.pkl')
#df.to_csv(inputPath+'Output/'+'cleaned_train.csv')
end = time.time()
print('time spent =', end-start)

#Sampling DF for @n number of customers only
print(50*'#')
print(datetime.datetime.now())
start = time.time()
print('Sampling Data for n number of customers and write sample100 CSV and Pickle files')

unique_ids   = pd.Series(df["cust_id"].unique())
limit_people = 100000
unique_id    = unique_ids.sample(n=limit_people)
sdf           = df[df.cust_id.isin(unique_id)]
sdf.name='The Sampled DataFrame for '+str(n)+ ' Customers selected from Cleaned Train_Ver2.csv file.'
sdf.describe()

#Sorting sdf based on Customer ID and Month
sdf.sort_values(by=["cust_id", "month"],inplace=True)
sdf.to_pickle(inputPath+'Output/'+'sample100.pkl')
#sdf.to_csv(inputPath+'Output/'+'sample100.csv')

end = time.time()
print('time spent =', end-start)

##########################################################################
# Experimenting more data modelling. This is an ongoing activity...
print(50*'#')
print(datetime.datetime.now())
start = time.time()
print('Experimenting more data modelling. This is an ongoing activity.')

PrDf(df, 70)
PrDf(sdf, 70)

#Sort df dataframe by Customer ID and Month
dfs=df.sort_values(by=["cust_id", "month"])

# Create a filterred DataFrame called fd based on the records that showing any of the features taken up or dropped by customers
fd=dfs[dfs.iloc[:,22].diff()!=0]
for i in range(23,46):
  fd=fd.append(dfs[dfs.iloc[:,i].diff()!=0])

fd=fd.sort_values(by=["cust_id", "month"])
fd=fd.drop_duplicates()

fd.name='The DataFrame from Customers that have taken up or droped a products.'

end = time.time()
print('time spent =', end-start)

fd.to_pickle(inputPath+'Output/'+'featureChanged.pkl')
fd.to_csv(inputPath+'Output/'+'featureChanged.csv')

fd.cust_id.unique().count()
