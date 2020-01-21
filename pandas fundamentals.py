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

## Selecting
sales.loc['2015-02-19 11:00:00', 'Company'] #single time
sales.loc['2015-2-5'] #whole day
sales.loc['2015-2'] #whole month
sales.loc['2015']   #whole year

##Partial string indexing and slicing
ts1 = ts0.loc['2010-10-11 21:00:00':'2010-10-11 22:00:00']# from 9pm to 10pm on '2010-10-11': ts1
ts2 = ts0.loc['2010-07-04']# Extract '2010-07-04' 
ts3 = ts0.loc['2010-12-15':'2010-12-31']#from '2010-12-15' to '2010-12-31'

##Reindexing the Index
# Reindex without fill method
ts3 = ts2.reindex(ts1.index)
sales.reindex(evening_2_11) #evening time of 2/11
# Reindex with forward fill
ts4 = ts2.reindex(ts1.index,method='ffill') #front missing data filling
sum12 = ts1 + ts2# Combine ts1 + ts2
########################################
#Resampling time series data
        #aggregate data
                #daily_mean = sales.resample('D').mean()  #daily mean
                        # print(daily_mean.loc['2015-2-2']) 
                        # print(sales.loc['2015-2-2', 'Units'])
                # sales.resample('D').sum().max() 
                # sales.resample('2W').count() # 2 Weekly        
                # two_days = sales.loc['2015-2-4': '2015-2-5', 'Units'] 
                        # two_days.resample('4H').ffill() 
########################################
##resampling by hourly and daily
# 6 hour data and aggregate by mean
df1 = df['Temperature'].resample('6h').mean()
# daily data and count the number of data points
df2 = df['Temperature'].resample('d').count()

##resampling by month
# Extract temperature data for August
august = df['Temperature']['2010-August']
# daily highest temperatures in August: august_highs
august_highs = august.resample('D').max()
# Extract temperature data for February
february = df['Temperature']['2010-February']
# daily lowest temperatures in February
february_lows = february.resample('D').min()

##Rolling(=smoothing)
unsmoothed = df['Temperature']['2010-Aug-01':'2010-Aug-15']
smoothed = unsmoothed.rolling(window=24).mean()# Apply a rolling mean with a 24 hour window: smoothed
august = pd.DataFrame({'smoothed':smoothed, 'unsmoothed':unsmoothed})# Create a new DataFrame
august.plot()# Plot both smoothed and unsmoothed
plt.show()

##Resampling and Rolling(=smoothing)
august = df['Temperature']['2010-August']# August 2010 data
daily_highs = august.resample('D').max()# daily data, aggregating by max
daily_highs_smoothed = daily_highs.rolling(window=7).mean()# Use a rolling 7-day window
print(daily_highs_smoothed)
########################################
#Manipulating time series data
        #set time zone:
        #convert time zone:
########################################
##Method chaining and filtering
# Strip 
df.columns = df.columns.str.strip()#extra whitespace from the column names
sales['Company'].str.upper()
# Substring to Dallas
dallas = df['Destination Airport'].str.contains('DAL')
# Compute the total number of Dallas departures each day
daily_departures = dallas.resample('D').sum()
# summary statistics for daily Dallas departures
stats = daily_departures.describe()

##Missing values and interpolation
# Reset the index of ts2 to ts1, and then use linear interpolation to fill in the NaNs: ts2_interp
ts2_interp = ts2.reindex(ts1.index).interpolate(how='linear')

# Compute the absolute difference of ts1 and ts2_interp: differences 
differences = np.abs(ts1 - ts2_interp)

# Generate and print summary statistics of the differences
print(differences.describe())

##Time zones and conversion
# Build a Boolean mask to filter for the 'LAX'
mask = df['Destination Airport'] == 'LAX'
la = df[mask]
# Combine two columns of data to create a datetime series
times_tz_none = pd.to_datetime( la['Date (MM/DD/YYYY)'] + ' ' + la['Wheels-off Time'] )
# Localize the time to US/Central: times_tz_central
times_tz_central = times_tz_none.dt.tz_localize('US/Central')
# Convert the datetimes from US/Central to US/Pacific
times_tz_pacific = times_tz_central.dt.tz_convert('US/Pacific')
########################################
#Time series visualization
        #set time zone:
        #convert time zone:
########################################
##Plotting time series, datetime indexing
# Plot the raw data before setting the datetime index
df.plot()
plt.show()
# Convert the 'Date' column into a collection of datetime objects: df.Date
df.Date = pd.to_datetime(df.Date)
# Set the index to be the converted 'Date' column
df.set_index('Date',inplace=True)
# Re-plot the DataFrame to see that the axis is now datetime aware!
df.plot()
plt.show()

##Plotting date ranges, partial indexing
# Plot the summer data
df.Temperature['2010-Jun':'2010-Aug'].plot()
plt.show()
plt.clf()
# Plot the one week data
df.Temperature['2010-06-10':'2010-06-17'].plot()
plt.show()
plt.clf()
################################################################################
#Case Study - Sunlight in Austin
################################################################################
##read file
import pandas as pd
df = pd.read_csv(data_file)
print(df.head())
# Read in the data file with header=None
df_headers = pd.read_csv(data_file, header=None)
print(df_headers.head())

##Re-assigning column names
column_labels_list = column_labels.split(',')# Split on the comma to create a list
df.columns = column_labels_list# Assign the new column labels
df_dropped = df.drop(list_to_drop,axis='columns')# Remove the appropriate columns
print(df_dropped.head())

##Cleaning and tidying datetime data
df_dropped['date'] = df_dropped['date'].astype(str)# Convert the date column to string
df_dropped['Time'] = df_dropped['Time'].apply(lambda x:'{:0>4}'.format(x))# Pad leading zeros to the Time column
date_string = df_dropped['date'] + df_dropped['Time']# Concatenate
date_times = pd.to_datetime(date_string, format='%Y%m%d%H%M')# Convert the date_string Series to datetime
df_clean = df_dropped.set_index(date_times)# Set the index to be the new date_time

##Cleaning the numeric columns
print(df_clean.loc['2011-06-20 8:00:00':'2011-06-20 9:00:00', 'dry_bulb_faren'])# print dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
df_clean['dry_bulb_faren'] = pd.to_numeric(df_clean['dry_bulb_faren'], errors='coerce')# Convert to numeric values
print(df_clean.loc['2011-06-20 8:00:00':'2011-06-20 9:00:00', 'dry_bulb_faren'])# Print the transformed dry_bulb_faren 
df_clean['wind_speed'] = pd.to_numeric(df_clean['wind_speed'], errors='coerce')# Convert the wind_speed and dew_point_faren columns to numeric values
df_clean['dew_point_faren'] = pd.to_numeric(df_clean['dew_point_faren'],errors='coerce')

##Signal min, max, median
print(df_clean['dry_bulb_faren'].median()) #median
print(df_clean.loc['2011-Apr':'2011-Jun', 'dry_bulb_faren'].median()) #median for '2011-Apr':'2011-Jun'
print(df_clean.loc['2011-Jan', 'dry_bulb_faren'].median()) #median for month of January

##Signal variance
daily_mean_2011 = df_clean.resample('D').mean() #daily_mean
daily_temp_2011 = daily_mean_2011['dry_bulb_faren'].values # daily_mean of  dry_bulb_faren column
daily_climate = df_climate.resample('D').mean() #daily_mean
daily_temp_climate = daily_climate.reset_index()['Temperature']# Extract the Temperature column 
difference = daily_temp_2011 - daily_temp_climate # difference
print(difference.mean())

##Sunny or cloudy
#Sunny --filter
is_sky_clear = df_clean['sky_condition']=='CLR'
sunny = df_clean.loc[is_sky_clear]
sunny_daily_max = sunny.resample('D').max()
#cloudy --contain
is_sky_overcast = df_clean['sky_condition'].str.contains('OVC')
overcast = df_clean.loc[is_sky_overcast]
overcast_daily_max = overcast.resample('D').max()

sunny_daily_max_mean = sunny_daily_max.mean() # mean of sunny_daily_max
overcast_daily_max_mean = overcast_daily_max.mean() # mean of overcast_daily
print(sunny_daily_max_mean-overcast_daily_max_mean) # difference


####Visual exploratory data analysis
##Weekly average temperature and visibility
import matplotlib.pyplot as plt
weekly_mean = df_clean[['visibility' ,'dry_bulb_faren']].resample('W').mean() #multiple columns 
print(weekly_mean.corr())
weekly_mean.plot(subplots=True)# Plot weekly_mean with subplots=True
plt.show()

##Daily hours of clear sky
is_sky_clear = df_clean['sky_condition']=='CLR' #sky_condition 'CLR'?
resampled = is_sky_clear.resample("D")
sunny_hours = resampled.sum()
total_hours = resampled.count()
sunny_fraction = sunny_hours/total_hours# Calculate the fraction of hours per day that were sunny

sunny_fraction.plot(kind='box')# Make a box plot of sunny_fraction
plt.show()

##Heat or humidity
monthly_max = df_clean[['dew_point_faren' , 'dry_bulb_faren']].resample('m').max()# Resample dew_point_faren and dry_bulb_faren by Month
monthly_max.plot(kind="hist",bins=8, alpha=0.5,  subplots=True)# Generate a histogram 
plt.show()

august_max = df_climate.loc['2010-August','Temperature'].max()# Extract the maximum temperature in August 2010 
print(august_max)
august_2011 = df_clean.loc['2011-August','dry_bulb_faren'].resample('D').max()# Resample August 2011 temps in df_clean by day & aggregate the max value
august_2011_high = august_2011[august_2011 > august_max]# Filter for days in august_2011 where the value exceeds august_max
august_2011_high.plot( kind='hist', bins=25, normed=True, cumulative=True)# Construct a CDF
plt.show()