# Final Codesfor Santander Product Recommendation on Kaggle
# Data Clean up is based on  Alan (AJ) Pryor, Jr. post on below link with minor modification and some add ups.
# https://www.kaggle.com/apryor6/santander-product-recommendation/detailed-cleaning-visualization-python

# There is absolutly no guarantee what so ever that the code work for any other system.
# You may use it on your own risk    with no warranty from author what so ever
# Open Source MIT Licence applies https://opensource.org/licenses/MIT

# Python 3 Codes on Windows Machine
# Paths should be revised for Linux machines

#########IMPORTANT#############
#Python 3 64 bit is required for working on files larger than 2GB.
###############################

# Pandas, Numpy and Seaborn require on Python 3 64 bit
# Download    them from http://www.lfd.uci.edu/~gohlke/pythonlibs/
# Then install it using $python -m install pandas-0.19.1-cp35-cp35m-win_amd64.whl
# And $python -m install tables-3.3.0-cp35-cp35m-win_amd64

# Download    scipy and seaborn from    http://www.lfd.uci.edu/~gohlke/pythonlibs/#pytables
# Then install like    $ python -m pip install seaborn-0.7.1-py2.py3-none-any.whl
# and then        $ python -m pip install seaborn-0.7.1-py2.py3-none-any.whl
# if you still get an error for         ImportError: cannot import name 'NUMPY_MKL'
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
global inputPath
global outputPath
inputPath                     = "YOUR INPUT PATH"
outputPath                    = "YOUR OUPUT PATH"


# Reading @name and @n from command line
nameArg=(sys.argv)[1]
nameArg='pkl'
if nameArg=='csv':
    readingCSV=True
else:
    readingCSV=False

name='train_ver2'

try:
  n=int((sys.argv)[2])
  sampling=True
except:
  sampling=False
  n=None

#Setting Name and n manually for running codes on console
sampling=True
n=100000

def ReadAndCleanData(name):
    try:
        # Rading input.csv file
        print(50*'#')
        print(datetime.datetime.now())
        start = time.time()
        print('Reading', name+'.csv')
        # Reading CSV Data to DataFrame
        # I have run this on a windows machine 38GB RAM and    64bit with Python and Pandas 64bit with no problem
        df = pd.read_csv(inputPath+name+'.csv', dtype={"sexo":str,
                         "ind_nuevo":str,
                         "ult_fec_cli_1t":str,
                         "indext":str})
        df.name='The Original dataFrame from Cleaned ' + name + ' file.'
        print(df.memory_usage().sum())
        end = time.time()
        print('time spent =', end-start)
        #Usage example when you want to print your sample data only where @renta is null
        #Example PrDf(sdf[sdf.renta.isnull()],100)
        pr(df, 10)
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
        df["age"]     = pd.to_numeric(df["age"], errors="coerce")
        # Here is some data cleaning technics copy paste from original article of Alan (AJ) Pryor, Jr. post on below link with minor modification and some add ups.
        # https://www.kaggle.com/apryor6/santander-product-recommendation/detailed-cleaning-visualization-python
        #####################################################################################
        df=df.loc[df.age.notnull()]
        df.age = df["age"].astype(int)
        df.antiguedad = pd.to_numeric(df.antiguedad,errors="coerce")
        df.loc[df.antiguedad==-999999,"antiguedad"] = 0
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
        #df.tiprel_1mes = df.tiprel_1mes.astype(str)
        # As suggested by @StephenSmith
        d0= { 1.0    : "1",
                                "1.0" : "1",
                                "1"     : "1",
                                "3.0" : "3",
                                "P"     : "5",
                                3.0     : "3",
                                2.0     : "2",
                                "3"     : "3",
                                "2.0" : "2",
                                "4.0" : "4",
                                "4"     : "4",
                                "2"     : "2"}
        df.indrel_1mes.fillna("P",inplace=True)
        df.indrel_1mes = df.indrel_1mes.apply(lambda x: d0.get(x,x))
        df.indrel_1mes = df.indrel_1mes.astype(int)
        unknown_cols = [col for col in missing_columns if col not in ["indfall","tiprel_1mes","indrel_1mes"]]
        for col in unknown_cols:
            df.loc[df[col].isnull(),col] = "U"
        feature_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values
        for col in feature_cols:
            df[col] = df[col].astype(int)
        df.indrel=df.indrel.astype(int)
        df.ind_actividad_cliente =df.ind_actividad_cliente .astype(int)
        #Renaming Columns to English Abbreviations to take less space when printing datafram usinf PrDf function
        colDic={
        'fecha_dato': 'rep_date', 'ncodpers': 'cust_id', 'ind_empleado': 'emp', 'pais_residencia': 'res', 'sexo': 'sx', 'age': 'ag', 'fecha_alta': 'join_date', 'ind_nuevo': 'new', 'antiguedad': 'csn', 'indrel': 'prm', 'ult_fec_cli_1t': 'lsd', 'indrel_1mes': 'ct1', 'tiprel_1mes': 'cr1', 'indresi': 'isr', 'indext': 'isf', 'conyuemp': 'spi', 'canal_entrada': 'jch', 'indfall': 'psd', 'nomprov': 'provna', 'ind_actividad_cliente': 'act', 'renta': 'inc', 'segmento': 'seg', 'ind_ahor_fin_ult1': 'sav', 'ind_aval_fin_ult1': 'gua', 'ind_cco_fin_ult1': 'cur', 'ind_cder_fin_ult1': 'der', 'ind_cno_fin_ult1': 'pay', 'ind_ctju_fin_ult1': 'jon', 'ind_ctma_fin_ult1': 'mpa', 'ind_ctop_fin_ult1': 'par', 'ind_ctpp_fin_ult1': 'pap', 'ind_deco_fin_ult1': 'std', 'ind_deme_fin_ult1': 'mts', 'ind_dela_fin_ult1': 'ltd', 'ind_ecue_fin_ult1': 'e_a', 'ind_fond_fin_ult1': 'fun', 'ind_hip_fin_ult1': 'mor', 'ind_plan_fin_ult1': 'pen', 'ind_pres_fin_ult1': 'loa', 'ind_reca_fin_ult1': 'tax', 'ind_tjcr_fin_ult1': 'crc', 'ind_valo_fin_ult1' : 'sec', 'ind_viv_fin_ult1': 'hom', 'ind_nomina_ult1': 'pyr', 'ind_nom_pens_ult1': 'npn', 'ind_recibo_ult1': 'did', 'month': 'month'
        }#tipodom cod_prov
        df.columns=['rep_date', 'cust_id', 'emp', 'res', 'sx', 'ag', 'join_date', 'new', 'csn', 'prm', 'lsd', 'ct1', 'cr1', 'isr', 'isf', 'spi', 'jch', 'psd', 'provna', 'act', 'inc', 'seg', 'sav', 'gua', 'cur', 'der', 'pay', 'jon', 'mpa', 'par', 'pap', 'std', 'mts', 'ltd', 'e_a', 'fun', 'mor', 'pen', 'loa', 'tax', 'crc', 'sec', 'hom', 'pyr', 'npn', 'did', 'month']
        pr(df)
        #Changing each feature to a specific number to be more recognizable when printign using PrDf Function
        df['products']=''
        for i in range(22,46):
            df['products'] = df['products'] + df.iloc[:,i].map(str)
            df.iloc[:,i][df.iloc[:,i]==1]=i-21+df.month*100
        #Converting some 0,1 or S,N values to bool type
        df.new=df.new.astype(int)
        # d = {'S': 1, 'N': 0, 'Y': 1, 'N': 0, 1:1, 0: 0, '1': 1, '0': 1}
        # df.isr=df.isr.map(d)
        # df.isf=df.isf.map(d)
        # df.act=df.act.map(d)
        # d1 = {'S': 1, 'N': 0, 'U': 2}
        # df.spi=df.spi.map(d1)
        # d2 = {'A': 1, 'I':2, 'P': 3, 'R': 4, 'N': 5}
        # df.cr1=df.cr1.map(d2)
        #changing segments to numeric integer values
        d3={'01 - TOP':1, '02 - PARTICULARES': 2, '03 - UNIVERSITARIO':3, 'U':4}
        df.seg=df.seg.map(d3)
        #Removing Passed Away Customers
        df=df[df.psd=='N']
        df.drop("psd",axis=1,inplace=True)
        df.loc[df.prm==99,"prm"] = '2'
        df.loc[df.inc.isnull(),"inc"] = 0.0
        df.loc[df.ag < 10,"ag"]  = 10
        df.loc[df.ag > 100,"ag"] = 99
        print('Saving Cleaned Data to Pickle format')
        #df.to_pickle(inputPath+'Output/'+name+'-CleanedData07.pkl')
        #Normalizing age between 1-9
        df['fac1']=(df.ag/10).map(int).map(str)+'-'+(df.new).map(str)+'-'+(df.csn/25.6).map(int).map(str)+'-'+(df.prm).map(str)+'-'+(df.ct1).map(str)+'-'+(df.act).map(str)+'-'+(df.inc/28895).map(int).map(str)+'-'+(df.seg).map(str)
        df['fac2']=df.emp+'-'+df.res+'-'+df.sx+'-'+df.cr1+'-'+df.isr+'-'+df.isf+'-'+df.spi+'-'+df.jch
        #df['fac3']=df.fac1+'-'+df.fac2
        df.drop(["emp","res", "sx", "ag", "new", "csn", "prm", "ct1", "cr1", "isr", "isf", "spi", "jch", "act", "inc", "seg"],axis=1,inplace=True)
        ##########################################
        print(df.memory_usage().sum())
        end = time.time()
        print('time spent =', end-start)
        print('Saving Shapped Data to Pickle format')
        df.to_pickle(inputPath+'Output/'+name+'-ShapedData07.pkl')
        return(df)
    except Exception as e1:
        print(50*'#'+' ERROR HAPPEND IN CLEANING DATA '+50*'#')
        print('Error Text: '+str(e1))
        return(df)

# Printing Dataframe on screen
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

def SampleData(df, n):
    unique_ids     = pd.Series(df["cust_id"].unique())
    limit_people = n
    unique_id        = unique_ids.sample(n=limit_people)
    df                     = df[df.cust_id.isin(unique_id)]
    df.name='The Sampled DataFrame for '+str(n)+ ' Customers selected from Cleaned Train_Ver2.csv file.'
    df.describe()
    return(df)

##################################################################################################################
#def AnalyseData(df)
    #Printing DF using PrDf Function
try:
    if readingCSV:
       df=ReadAndCleanData(name)
except:
    print('readingCSV is not defined')

###########################################################################################
try:
    pr(df, 10)
    print('df was already in  memory')
except:
    print('Reading df from Pickle file')
    df=pd.read_pickle(inputPath+'Output/'+name+'-ShapedData07.pkl')
    pr(df, 10)

if sampling:
    df=SampleData(df, n)

#Sorting DataFrame based on customer ID and Month
print(50*'#')
print(datetime.datetime.now())
start = time.time()

df=df.sort_values(by=['cust_id','month'])

print('Finding New Products purchased each month')

nc=pd.DataFrame()

for tm in range(1,17):
    nm=tm+1
    print(50*'#')
    print(datetime.datetime.now())
    start = time.time()
    exec("print('Finding New Customers in month %d')" %nm)
    exec("tdf=df[df.month.isin([%d,%d])]" % (tm, nm))
    vc=tdf.cust_id.value_counts()
    #Finding the lines that is only in month 5 or file 6 and not both
    tfd=tdf[(tdf.cust_id.isin(list(vc[vc==1].index))) & (tdf.month==nm)]
    pr(tfd)
    nc=nc.append(tfd)

for i in range(5,29):
    df.ix[((df.iloc[:,i].diff().isin(list(range(2*100+i-4,18*100+i-4,100))).shift(-1).fillna(False))&(df.month!=17)), df.columns[i]] = i-4
    pr(df[df.iloc[:,i]==i-4])

pr(df)
df.to_pickle(inputPath+'Output/'+name+'-MarkedNewProducts07-'+str(n)+'.pkl')

fd=df.loc[(df.sav==1) | (df.gua==2) | (df.cur==3) | (df.der==4) | (df.pay==5) | (df.jon==6) | (df.mpa==7) | (df.par==8) | (df.pap==9) | (df.std==10) | (df.mts==11) | (df.ltd==12) | (df.e_a==13) | (df.fun==14) | (df.mor==15) | (df.pen==16) | (df.loa==17) | (df.tax==18) | (df.crc==19) | (df.sec==20) | (df.hom==21) | (df.pyr==22) | (df.npn==23) | (df.did==24)]

for i in range(5,29):
    fd.ix[~df.iloc[:,i].isin(range(1,25)), fd.columns[i]] = 0

end = time.time()
print('time spent =', end-start)

pr(fd)
fd.to_pickle(inputPath+'Output/'+name+'-NewProductData07-1-'+str(n)+'.pkl')
#fd=pd.read_pickle(inputPath+'Output/'+'NewProductData07.pkl')

pr(nc)
nc.to_pickle(inputPath+'Output/'+name+'-NewCustomerData07-'+str(n)+'.pkl')


#fd[fd.cust_id==15889].fac3.value_counts().index[0]
#(sb1.ncodpers.values==subm.ncodpers.values).all()


###########################################################


print(50*'#')
print(datetime.datetime.now())
start = time.time()

periods={
    1:[0,395000],
    2:[394999,805000],
    3:[804999,1085000],
    4:[1084999,1357000],
    5:[1356999,1553690]
}

per=1

#df=df[(df.cust_id>periods[per][0])&(df.cust_id<periods[per][1])]
#df=df[df.month.isin([5,13,14,15,16,17])]
df=df.sort_values(by=['cust_id', 'month'])

print('Finding New Products purchased Part '+str(per))

###################################################################################################3

dic={
1: 'ind_ahor_fin_ult1', 2: 'ind_aval_fin_ult1', 3: 'ind_cco_fin_ult1', 4:  'ind_cder_fin_ult1', 5: 'ind_cno_fin_ult1', 6: 'ind_ctju_fin_ult1', 7: 'ind_ctma_fin_ult1', 8: 'ind_ctop_fin_ult1', 9: 'ind_ctpp_fin_ult1', 10: 'ind_deco_fin_ult1', 11: 'ind_deme_fin_ult1', 12: 'ind_dela_fin_ult1', 13: 'ind_ecue_fin_ult1', 14: 'ind_fond_fin_ult1', 15:  'ind_hip_fin_ult1', 16: 'ind_plan_fin_ult1', 17: 'ind_pres_fin_ult1', 18: 'ind_reca_fin_ult1', 19: 'ind_tjcr_fin_ult1', 20: 'ind_valo_fin_ult1', 21: 'ind_viv_fin_ult1', 22: 'ind_nomina_ult1', 23: 'ind_nom_pens_ult1', 24: 'ind_recibo_ult1'
}

def RemLast(this, last):
    lst=[i for i, letter in enumerate(last) if letter == '1']
    for l in lst:
        this=this[:l]+'0'+this[l+1:]
    return(this)

def Predict(cid):
    #fdf2=df[df.fac2==fd[(fd.cust_id==id)&(fd.month==16)].fac2.value_counts().index[0]].products.value_counts()
    cr1=fd[fd.fac1==df[df.cust_id==cid].fac1.value_counts().index[0]].products.value_counts()
    cr2=fd[fd.fac2==df[df.cust_id==cid].fac2.value_counts().index[0]].products.value_counts()
    last=df[df.cust_id==cid][df.month==17].products.values[0]
    prds=[]
    j=-1
    while len(prds)<7 and j<50:
        j+=1
        if j<len(cr1):
            pre1=RemLast(cr1.index[j], last)
            if pre1!='000000000000000000000000':
                for l in [i for i, letter in enumerate(pre1) if letter == '1']:
                    prds.append(dic[l+1])
        if j<len(cr2):
            pre2=RemLast(cr2.index[j], last)
            if pre2!='000000000000000000000000':
                for l in [i for i, letter in enumerate(pre2) if letter == '1']:
                    prds.append(dic[l+1])
        prds=list(set(prds))
    if len(prds)>7:
        prds=prds[:7]
    print(' '.join(prds))
    return(' '.join(prds))


sbm=pd.read_csv(inputPath+'sample_submission.csv',index_col=1)
sbm.sort_index(inplace=True)
sbm=sbm[(sbm.index>periods[per][0])&(sbm.index<periods[per][1])]

#it=int(len(sbm.index)/2)
it=0
sbm['added_products']=''

print(50*'#')
print(datetime.datetime.now())
start = time.time()
for cid in sbm.index:
    it=it+1
    if it%1000==0:
        sbm.to_csv(inputPath+'Output/sample_submission07-3-'+str(per)+'.csv')
        print(50*'#')
        print(datetime.datetime.now())
        end = time.time()
        print('time spent =', end-start)
        start = time.time()
    print(it, cid)
    try:
        sbm.loc[cid]=Predict(cid)
    except Exception as e:
        print(50*'#'+' ERROR HAPPEND IN FINDING PRODUCTS '+50*'#')
        print('Error Text: '+str(e))

sbm.to_csv(inputPath+'Output/sample_submission07-3-'+str(per)+'.csv')

