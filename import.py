! ls #find the list

# Open a file: file
file = open('moby_dick.txt', 'r')

# Print it
print(file.read())

# Check whether file is closed
print(file.closed)

# Close file
file.close()

# Check whether file is closed
print(file.closed)

# Read & print the first 3 lines
with open('moby_dick.txt') as file:
    print(file.readline())
    print(file.readline())
    print(file.readline())
################################################
#Numpy
    # comma-delimited: ','
    # tab-delimited: '\t'
################################################
#import data and reshape
import numpy as np
file = 'digits.csv'
digits = np.loadtxt(file, delimiter=',') # comma-delimited
print(type(digits))# Print datatype of digits
im = digits[21, 1:]# Select and reshape a row
im_sq = np.reshape(im, (28, 28))
plt.imshow(im_sq, cmap='Greys', interpolation='nearest')# Plot reshaped data
plt.show()
#import data with skip 1 low, only first and third columns
import numpy as np
file = 'digits_header.txt'
data = np.loadtxt(file, delimiter='\t', skiprows=1, usecols=[0,2]) # tab-delimited
print(data)


#import plot
file = 'seaslug.txt'
data_float = np.loadtxt(file, delimiter='\t', dtype=float, skiprows=1)# Import data
print(data_float[9])
plt.scatter(data_float[:, 0], data_float[:, 1])# Plot
plt.xlabel('time (min.)')
plt.ylabel('percentage of larvae')
plt.show()

# mixed datatypes 
file = 'titanic.csv'
d=np.recfromcsv(file,delimiter=',',dtype=None,names=True)
print(d[:3])
################################################
#Pandas
################################################
import pandas as pd
file = 'titanic.csv'
df = pd.read_csv(file)
print(df.head())


file = 'digits.csv'
data = pd.read_csv(file, nrows=5, header=None) # Read the first 5 rows
data_array = data.values# Build a numpy array
print(type(data_array))

#import and histogram
import matplotlib.pyplot as plt
file = 'titanic_corrupt.txt'
data = pd.read_csv(file, sep='\t', comment='#', na_values=['Nothing'])
print(data.head())
pd.DataFrame.hist(data[['Age']])
plt.xlabel('Age (years)')
plt.ylabel('count')
plt.show()
################################################
#Different file types
    #Excel has many spreadsheets,not a single take, it is not a flat
################################################
#get the list of files
import os
wd = os.getcwd()
os.listdir(wd) #get the list of files
#Loading a pickled file
import pickle
with open('data.pkl', 'rb') as file:
    d = pickle.load(file) #read and binary ='rb
print(d)
print(type(d))

#Excel file open
import pandas as pd
file = 'battledeath.xlsx'# Assign spreadsheet filename: file
xls = pd.ExcelFile(file)
print(xls.sheet_names)# Print sheet names

#Select Spreadsheet
#1)
df1 = xls.parse('2004')
print(df1.head())
#2) using index
df2 = xls.parse(0)
print(df2.head())