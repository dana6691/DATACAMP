################################################################################################
#Indexing
################################################################################################
## Assign Index Column
import pandas as pd
election = pd.read_csv(filename, index_col='county')

# Create a separate dataframe with the columns ['winner', 'total', 'voters']: results
results = pd.DataFrame(election[['winner', 'total', 'voters']])
print(results.head())

## Select index + column name
election.loc['Bedford', 'winner'] # country name is index [index name , column name]
election.iloc[4,4] # Same
################################################
#Slicing DataFrames
################################################
df = pd.read_csv('sales.csv', index_col='month') 
df['salt']['Jan'] #column name , index name
df.eggs['Mar']
df.loc['May', 'spam'] # index name, column name
df.iloc[4, 2] # index number, column number

df[['salt','eggs']] #selecting columns in DataFrame
df['eggs']  #selecting columns in Series

#Slicing 
df['eggs'][1:4]
df.loc[:, 'eggs':'salt']
df.loc['Jan':'Apr',:] 
df.loc['Mar':'May', 'salt':'spam'] 
df.iloc[2:5, 1:] 
df.loc['Jan':'May', ['eggs', 'spam']] 
df.iloc[[0,4,5], 0:2] 

## Slicing rows
# Slice the row labels 'Perry' to 'Potter'
p_counties = election.loc['Perry':'Potter',:]
print(p_counties)

# reverse order
p_counties_rev = election.loc['Potter':'Perry':-1]
print(p_counties_rev)

## Slicing columns
# starting column to 'Obama': left_columns
left_columns = election.loc[:,:'Obama']
print(left_columns.head())

# from 'Obama' to 'winner': middle_columns
middle_columns = election.loc[:,'Obama':'winner']
print(middle_columns.head())

# from 'Romney' to the end: 'right_columns'
right_columns = election.loc[:,'Romney':]
print(right_columns.head())

## Subselecting DataFrames with lists
# Create the list of row labels: rows
rows = ['Philadelphia', 'Centre', 'Fulton']
# Create the list of column labels: cols
cols = ['winner', 'Obama', 'Romney']
three_counties = election.loc[rows,cols]
print(three_counties)
################################################
#Filtering DataFrames
    #filter:  df[(df.salt >= 50) & (df.eggs < 200)]
    #all nonzero:  df2.loc[:, df2.all()] 
    #any nonzeros:  df2.loc[:, df2.any()] 
    #any NaNs:  df.loc[:, df.isnull().any()]
    #without NaNs:  df.loc[:, df.notnull().all()]  
    #Drop any NaNs: df.dropna(how='any') 
################################################
## Subsetting
high_turnout_df = election[election['turnout'] > 70]
print(high_turnout_df)


##Filtering columns using other columns
import numpy as np
too_close = election['margin'] < 1
election.loc[too_close, 'winner'] = np.nan # Assign np.nan to 'winner' column, too_close row
print(election.info())

## Filtering using NaNs, Any, All
# Select the 'age' and 'cabin' columns: df
df = titanic[['age', 'cabin']]

# Drop rows in df with how='any' and print the shape
print(df.dropna(how='any').shape)

# Drop rows in df with how='all' and print the shape
print(df.dropna(how='all').shape)

# Drop columns in titanic with less than 1000 non-missing values
print(titanic.dropna(thresh=1000, axis='columns').info())