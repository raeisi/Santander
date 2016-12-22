##########  This requires Pandas and Tables   ##############
##########  Python 32 bit gives memory error when the memory consumption goes more than 2GB#########
# CSV Files will be Exported to python Pickle  or Table h5 files.
# It can load large CSV files due to use of chunksize in read_csv
# These codes have been found and gathered from internet and just uploaded here
# for my own convenience for future use and documentation
# There is no guarantee what so ever that the code work for any other system.
# You may use it on your own risk  with no warranty from author what so ever
# Open Source MIT Licence applies https://opensource.org/licenses/MIT

# Python 3 Codes on Windows Machine
# Paths should be  assigned for input and output file folder

# Pandas and Numpy require on Python 3 64 bit
# Download  them from http://www.lfd.uci.edu/~gohlke/pythonlibs/
# Then install it using $python -m install pandas-0.19.1-cp35-cp35m-win_amd64.whl
# And $python -m install  tables-3.3.0-cp35-cp35m-win_amd64
# In General I suggest to imstall below files from same website.
# geopy, lxml, seaborn, matplotlib, scipy, pymssql, pyodbc

# This Code needs around 40GB Memory to be finished

import numpy as np
import pandas as pd
import datetime
import time
import os, sys

# Paths should be revised for Linux machines
inputPath="YOUR INPUT PATH"
outputPath="YOUR OUPUT PATH"

# Reading @name and @ext from command line
name=(sys.argv)[1]
exts=(sys.argv)[2:]

def export_to(name,exts):
	#Reading CSV File
	print(50*'#')
	print(datetime.datetime.now())
	start = time.time()
	print('Reading', name+'.csv')
	c=pd.read_csv(inputPath+name+'.csv', low_memory=False, chunksize=10000, dtype=object)
	df=pd.concat(c, ignore_index=True).convert_objects()
	#Printing Memory Usage
	print(df.memory_usage().sum())
	print(datetime.datetime.now())
	end = time.time()
	print('time spent =', end-start)
	#Writing EXT Files
	for ext in exts:
		print(50*'#')
		print(datetime.datetime.now())
		start=time.time()
		print('Writing '+name+"."+ext)
		if ext == 'pkl':
			try:
				df.to_pickle(outputPath+name+"."+ext)
			except Exception as e:
				print(ext + ' Write has issues.\nError Code => '+str(e))
		elif ext == 'h5':
			try:
				df.to_hdf(outputPath+name+"."+ext, 'df', mode='w', format='table')
			except Exception as e:
				print(ext + ' Write has issues.\nError Code => '+str(e))
		else:
			print('Extension is not accepted yet. if you want it code it')
		end = time.time()
		print('time spent =', end-start)
	del df

#export_to('test_ver2', ['pkl', 'h5'])
export_to(name, exts)

# Usage Example
# $python exportTo.py train_ver2 pkl h5
