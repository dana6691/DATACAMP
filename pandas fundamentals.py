########################################
#Pandas
########################################
df.head()
df.tail()
df.info()
import numpy as np
AAPL.iloc[::3, -1] = np.nan
AAPL.head(6)
type(AAPL)
AAPL.shape
AAPL.columns 
AAPL.index
AAPL.iloc[:5,:] 

import numpy as np
# Create array of DataFrame values: np_vals
np_vals = np.array(df)

# Create new array of base 10 logarithm values: np_vals_log10
np_vals_log10 = np.array(np.log10(df))

# Create array of new DataFrame by passing df to np.log10(): df_log10
df_log10 = np.array(np.log10(df))

# Print original and new data containers
[print(x, 'has type', type(eval(x))) for x in ['np_vals', 'np_vals_log10', 'df', 'df_log10']]
########################################
#Building DataFrames from scratch
########################################
import pandas as pd
data = {'weekday': ['Sun', 'Sun', 'Mon', 'Mon'],
        'city': ['Austin', 'Dallas', 'Austin', 'Dallas',
        'visitors': [139, 237, 326, 456],
        'signups': [7, 12, 3, 5]}
users = pd.DataFrame(data)


## Zip the 2 lists and make it dictionary
zipped = list(zip(list_keys,list_values))
print(zipped)
data = dict(zipped)# Build a dictionary
df = pd.DataFrame(data)
print(df)

##create list and column name assign
list_labels = ['year', 'artist', 'song', 'chart weeks']
df.columns = list_labels# Assign the list of labels to the columns attribute: df.columns

##Building DataFrames
state = 'PA'
data = {'state':state, 'city':cities}# Construct a dictionary
df = pd.DataFrame(data)# Construct a DataFrame 
print(df)
########################################
#Importing & exporting data
########################################
## Read files
df1 = pd.read_csv(data_file)
new_labels = ['year', 'population']# Create a list of the new column labels
df2 = pd.read_csv(data_file, header=0, names=new_labels)
print(df1)
print(df2)

##Delimiters, headers, and saved file to CSV,Excel
df1 = pd.read_csv(file_messy)
print(df1.head())
df2 = pd.read_csv(file_messy, delimiter=' ', header=3, comment='#')
print(df2.head())
df2.to_csv(file_clean, index=False)# Save to a CSV file without the index
df2.to_excel('file_clean.xlsx', index=False)# Save to excel file without the index
########################################
#Plotting
########################################
##plot with color, title, labels
df.plot(c='red')
plt.title('Temperature in Austin')
plt.xlabel('Hours since midnight August 1, 2010')
plt.ylabel('Temperature (degrees F)')
plt.show()

## Plot all columns (default)
df.plot()
plt.show()

## Plot all columns as subplots
df.plot(subplots=True)
plt.show()

## Plot one columns
column_list1 = ['Dew Point (deg F)']
df[column_list1].plot()
plt.show()

# PPLot two columns
column_list2 = ['Temperature (deg F)','Dew Point (deg F)']
df[column_list2].plot()
plt.show()
########################################
#Visual exploratory

#Histogram options
● bins (integer): number of intervals or bins
● range (tuple): extrema of bins (minimum, maximum)
● normed (boolean): whether to normalize to one
● cumulative (boolean): compute Cumulative Distribution
Function (CDF) 
########################################
##pandas line plots
y_columns = ['AAPL' , 'IBM'] #list of y-axis column
df.plot(x='Month', y=y_columns)
plt.title('Monthly stock prices')
plt.ylabel('Price ($US)')
plt.show()

##pandas scatter plots
df.plot(kind='scatter', x='hp', y='mpg', s=sizes)
plt.title('Fuel efficiency vs Horse-power')
plt.xlabel('Horse-power')
plt.ylabel('Fuel efficiency (mpg)')
plt.show()

##pandas Box plot
cols = ['weight' , 'mpg']
df[cols].plot(subplots=True,kind="box")
plt.show()

##pandas hist, pdf and cdf
# This formats the plots such that they appear on separate rows
fig, axes = plt.subplots(nrows=2, ncols=1)
# Plot the PDF
df.fraction.plot(ax=axes[0], kind='hist', bins=30, normed=True, range=(0,.3))
plt.show()
# Plot the CDF
df.fraction.plot(ax=axes[1], kind='hist', bins=30, normed=True, cumulative=True, range=(0,.3))
plt.show()
########################################
#.describe()
#.count()
#.mean() .std() .median() .quantile([0.5, 0.25 ])
#.min(), .max()
########################################
##Mean, Min,Max
# Print the minimum value of the Engineering column
print(df['Engineering'].min())
# Print the maximum value of the Engineering column
print(df['Engineering'].max())
# Construct the mean percentage per year: mean
mean = df.mean(axis='columns')
# Plot the average percentage per year
mean.plot()

##.describe, boxplot
# Summary statistics of the fare column with .describe()
print(df.fare.describe())
# Box plot 
df.fare.plot(kind='box')
plt.show()
`
## count(), percentiles(), box plot
# Print the number of countries reported in 2015
print(df['2015'].count())
# Print the 5th and 95th percentiles
print(df.quantile([0.05, 0.95]))
# Generate a box plot
years = ['1800','1850','1900','1950','2000']
df[years].plot(kind='box')
plt.show()

## Mean, Standard Deviation
print(january.mean(), march.mean())
print(january.std(), march.std())
########################################
# Filter
########################################
# Filter the US population from the origin column: us
us = df[df['origin']=='US']

# box plots on 3 separate rows and 1 column
fig, axes = plt.subplots(nrows=3, ncols=1)
titanic.loc[titanic['pclass'] == 1].plot(ax=axes[0], y='fare', kind='box')d
titanic.loc[titanic['pclass'] == 2].plot(ax=axes[1], y='fare', kind='box')
titanic.loc[titanic['pclass'] == 3].plot(ax=axes[2], y='fare', kind='box')
plt.show()
################################################################################
# Time series in pandas
################################################################################
## Using pandas to read datetime objects
df3 = pd.read_csv(filename, index_col='Date', parse_dates=True)

## Creating DatetimeIndex
time_format = '%Y-%m-%d %H:%M'# format string
my_datetimes = pd.to_datetime(date_list, format=time_format) #datetime object 
time_series = pd.Series(temperature_list, index=my_datetimes)# time_series

##Partial string indexing and slicing
ts1 = ts0.loc['2010-10-11 21:00:00':'2010-10-11 22:00:00']# from 9pm to 10pm on '2010-10-11': ts1
ts2 = ts0.loc['2010-07-04']# Extract '2010-07-04' 
ts3 = ts0.loc['2010-12-15':'2010-12-31']#from '2010-12-15' to '2010-12-31'

##Reindexing the Index
# Reindex without fill method
ts3 = ts2.reindex(ts1.index)
# Reindex with forward fill
ts4 = ts2.reindex(ts1.index,method='ffill')
sum12 = ts1 + ts2# Combine ts1 + ts2




