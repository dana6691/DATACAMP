################################################################################
#Preparing data
#Reading multiple data files
################################################################################
## Import pandas
import pandas as  pd
bronze = pd.read_csv('Bronze.csv')

## Reading multiple files
filenames = ['Gold.csv', 'Silver.csv', 'Bronze.csv']
dataframes = []
for filename in filenames:
    dataframes.append(pd.read_csv(filename))

## Combining DataFrames from multiple data files
import pandas as pd
medals = gold.copy() #copy
new_labels = ['NOC', 'Country', 'Gold']
medals.columns = new_labels # Rename the columns
medals['Silver'] = silver['Total']
medals['Bronze'] = bronze['Total']
########################################
#Reindexing DataFrames
########################################
weather1 = pd.read_csv('monthly_max_temp.csv',index_col='Month')
print(weather1.head())

# Sort the index
weather2 = weather1.sort_index()
print(weather2.head())

# Sort the index in reverse 
weather3 = weather1.sort_index(ascending=False)
print(weather3.head())

# Sort weather1 numerically of 'Max TemperatureF': weather4
weather4 = weather1.sort_values('Max TemperatureF')
print(weather4.head())
########################################
#Reindexing DataFrame from a list
########################################
# Reindex weather1 using the list year: weather2
weather2 = weather1.reindex(year)
print(weather2)

# Reindex weather1 using the list year with forward-fill: weather3
weather3 = weather1.reindex(year).ffill()
print(weather3)
########################################
#Reindexing using another DataFrame Index
########################################
# Reindex names_1981 with index of names_1881: common_names
common_names = names_1981.reindex(names_1881.index)
print(common_names.shape)

# Drop rows with null counts: common_names
common_names = common_names.dropna()
print(common_names.shape)
########################################
#Arithmetic with Series & DataFrames
########################################
temps_f = weather[['Min TemperatureF', 'Mean TemperatureF',  'Max TemperatureF']]
temps_c = (temps_f - 32) * 5/9
# Rename 'F' in column names with 'C': temps_c.columns
temps_c.columns = temps_c.columns.str.replace('F', 'C')
print(temps_c.head())

##Computing percentage growth of GDP
import pandas as pd
gdp = pd.read_csv('GDP.csv',parse_dates=True , index_col='DATE')
# Slice all the gdp data from 2008 onward: post2008
post2008 = gdp.loc['2008':]
print(post2008.tail(8))
# Resample post2008 by year, keeping last(): yearly
yearly = post2008.resample("A").last()
print(yearly)
# Compute percentage growth of yearly: yearly['growth']
yearly['growth'] = yearly.pct_change() * 100
print(yearly)

##Converting currency of stocks
import pandas as pd
sp500 = pd.read_csv('sp500.csv',parse_dates=True , index_col='Date')
exchange = pd.read_csv('exchange.csv',parse_dates=True , index_col='Date')

# Subset 'Open' & 'Close' columns from sp500: dollars
dollars = sp500[['Open','Close']]
print(dollars.head())

# Convert dollars to pounds: pounds
pounds = dollars.multiply(exchange['GBP/USD'],axis='rows')
print(pounds.head())

################################################################################
#Appending & concatenating Series
    #.append().rest_index(drop=True)
    #pd.concat() , ignore_index
################################################################################
########################################
#Arithmetic with Series & DataFrames
########################################
import pandas as pd
jan = pd.read_csv('sales-jan-2015.csv', index_col='Date', parse_dates=True)
feb = pd.read_csv('sales-feb-2015.csv', index_col='Date', parse_dates=True)
mar = pd.read_csv('sales-mar-2015.csv', index_col='Date', parse_dates=True)

# Extract the 'Units' column 
jan_units = jan['Units']
feb_units = feb['Units']
mar_units = mar['Units']

# Append feb_units and then mar_units to jan_units: quarter1
quarter1 = jan_units.append(feb_units).append(mar_units)

# Print the first slice
print(quarter1.loc['jan 27, 2015':'feb 2, 2015'])
# Print the second slice
print(quarter1.loc['feb 26, 2015':'mar 7, 2015'])
# Compute & print total sales in quarter1
print(quarter1.sum())
########################################
#Concatenating pandas Series along row axis
########################################
units = []
for month in [jan, feb, mar]:
    units.append(month['Units'])
# Concatenate the list: quarter1
quarter1 = pd.concat(units,axis='rows')
print(quarter1.loc['jan 27, 2015':'feb 2, 2015'])
print(quarter1.loc['feb 26, 2015':'mar 7, 2015'])
########################################
#Appending & concatenating DataFrames
########################################
## Appending DataFrames with ignore_index
names_1881['year'] = 1881
names_1981['year'] = 1981

# Append names_1981 after names_1881 with ignore_index=True: combined_names
combined_names = names_1881.append(names_1981,ignore_index=True)
print(names_1981.shape)
print(names_1881.shape)
print(combined_names.shape)

# Print all rows that contain the name 'Morgan'
print(combined_names[combined_names['name'] == 'Morgan'])

## Concatenating pandas DataFrames along column axis
weather_list = [weather_max , weather_mean]
weather = pd.concat(weather_list,axis=1)# Concatenate weather_list horizontally
print(weather)

##Reading multiple files to build a DataFrame
medals =[]
for medal in medal_types:
    file_name = "%s_top5.csv" % medal
    columns = ['Country', medal]
    medal_df = pd.read_csv(file_name,header=0, index_col='Country', names=columns)
    medals.append(medal_df)
# Concatenate medals horizontally: medals_df
medals_df = pd.concat(medals,axis='columns')
print(medals_df)
########################################
#Concatenation, keys, & MultiIndexes
      #Stacking arrays vertically: np.concatenate([A, C], axis=0)
    #Stacking arrays horizontally:  np.concatenate([B, A], axis=1) 
    #multi-index on rows: pd.concat([rain2013, rain2014], keys=[2013, 2014], axis=0)
########################################
##Concatenating vertically to get MultiIndexed rows
for medal in medal_types:
    file_name = "%s_top5.csv" % medal
    medal_df = pd.read_csv(file_name,index_col='Country')
    medals.append(medal_df)
medals = pd.concat(medals,keys=['bronze', 'silver', 'gold'])
print(medals)

##Slicing MultiIndexed DataFrames
medals_sorted = medals.sort_index(level=0)# Sort the entries of medals
print(medals_sorted.loc[('bronze','Germany')]) #number of Bronze medals won by Germany
print(medals_sorted.loc['silver']) #silver medals
# Create alias for pd.IndexSlice: idx
idx = pd.IndexSlice
# Print all the data on medals won by the United Kingdom
print(medals_sorted.loc[idx[:,'United Kingdom'], :])

##Concatenating horizontally to get MultiIndexed columns
february = pd.concat(dataframes,keys=['Hardware', 'Software', 'Service'],axis=1)
print(february.info())
idx = pd.IndexSlice #Assign pd.IndexSlice: idx
# Create the slice: slice_2_8
slice_2_8 = february.loc['Feb. 2, 2015':'Feb. 8, 2015', idx[:, 'Company']]
print(slice_2_8)

##Concatenating DataFrames from a dict
# Make the list of tuples: month_list
month_list = (('january', jan), ('february', feb),  ('march', mar))
month_dict = {}

for month_name, month_data in month_list:
    month_dict[month_name] = month_data.groupby('Company').sum()
sales = pd.concat(month_dict)
print(sales)

idx = pd.IndexSlice# Print all sales by Mediacore
print(sales.loc[idx[:, 'Mediacore'], :])
########################################
#Outer & inner joins
    #Inner join: pd.concat([population, unemployment], axis=1, join='inner')
########################################
##Concatenating DataFrames with inner join
medal_list = [bronze, silver, gold]
# Concatenate medal_list horizontally using an inner join: medals
medals = pd.concat(medal_list,keys=['bronze', 'silver', 'gold'],axis=1,join='inner')
print(medals)

##Resampling & concatenating DataFrames with inner join
# Annually, percentage of change with an offset of ten years.
china_annual = china.resample('A').last().pct_change(10).dropna()
us_annual = us.resample('A').last().pct_change(10).dropna()

# Concatenate china_annual and us_annual
gdp = pd.concat([china_annual,us_annual],join='inner',axis=1 )
# Resample gdp and print
print(gdp.resample('10A').last())
################################################################################
#Merging DataFrames
    # pd.merge(population, cities)
    # Merging on: pd.merge(bronze, gold, on='NOC')
    # Merging on multiple columns: pd.merge(bronze, gold, on=['NOC', 'Country']) 
    # Using suffixes:  pd.merge(bronze, gold, on=['NOC', 'Country'], suffixes=['_bronze', '_gold']) 
    #  pd.merge(counties, cities, left_on='CITY NAME', right_on='City')
################################################################################
##Merging on a specific column
# Merge revenue with managers on 'city': merge_by_city
merge_by_city = pd.merge(revenue,managers,on='city')
print(merge_by_city)
# Merge revenue with managers on 'branch_id': merge_by_id
merge_by_id = pd.merge(revenue,managers,on='branch_id')
print(merge_by_id)

##Merging on columns with non-matching labels
combined = pd.merge(revenue,managers,left_on='city',right_on='branch') ## Merge revenue & managers on 'city' & 'branch'
print(combined)

##Merging on multiple columns
revenue['state'] = ['TX','CO','IL','CA']
managers['state'] = ['TX','CO','CA','MO']
# Merge revenue & managers on 'branch_id', 'city', & 'state': combined
combined = pd.merge(revenue, managers,on=['branch_id', 'city',  'state'] )
print(combined)
########################################
#Joining DataFrames
    #.join(how='left)
    #.join(how='right')
    #.join(how='outer')
    #.join(how='inner')
    '''
● df1.append(df2): stacking vertically
● pd.concat([df1, df2]):
● stacking many horizontally or vertically
● simple inner/outer joins on Indexes
● df1.join(df2): inner/outer/le!/right joins on Indexes
● pd.merge([df1, df2]): many joins on multiple columns'''
########################################
##Left & right merging on multiple columns
# Merge revenue and sales: revenue_and_sales
revenue_and_sales = pd.merge(revenue , sales,how='right' , on=['city', 'state'])
print(revenue_and_sales)

# Merge sales and managers: sales_and_managers
sales_and_managers = pd.merge(sales , managers,how='left', left_on=['city', 'state'],right_on=['branch', 'state'])
print(revenue_and_sales)
print(sales_and_managers)

##Merging DataFrames with outer join
# Perform the first merge: merge_default
merge_default = pd.merge(sales_and_managers,revenue_and_sales)
print(merge_default)

# Perform the second merge: merge_outer
merge_outer = pd.merge(sales_and_managers,revenue_and_sales,how='outer')
print(merge_outer)

# Perform the third merge: merge_outer_on
merge_outer_on = pd.merge(sales_and_managers,revenue_and_sales,how='outer',on =['city','state'])
print(merge_outer_on)

########################################
#Ordered merges
########################################
##Using merge_ordered()
# Perform the first ordered merge: tx_weather
tx_weather = pd.merge_ordered(austin , houston)
print(tx_weather)
# Perform the second ordered merge: tx_weather_suff
tx_weather_suff = pd.merge_ordered(austin , houston,on='date' , suffixes=['_aus','_hus'])
print(tx_weather_suff)
# Perform the third ordered merge: tx_weather_ffill
tx_weather_ffill = pd.merge_ordered(austin , houston,on='date' , suffixes=['_aus','_hus'],fill_method='ffill')
print(tx_weather_ffill)

##Using merge_asof()
# Merge auto and oil: merged
merged = pd.merge_asof(auto,oil,left_on='yr' , right_on='Date')
print(merged.tail())

# Resample merged: yearly
yearly = merged.resample('A',on='Date')[['mpg','Price']].mean()
print(yearly)
print(yearly.corr())
########################################
#Case Study: Medals in the Summer Olympics
########################################
import pandas as pd
file_path = 'Summer Olympic medallists 1896 to 2008 - EDITIONS.tsv'
editions = pd.read_csv(file_path, sep='\t')
# Extract the relevant columns: editions
editions = editions[['Edition', 'Grand Total', 'City', 'Country']]
print(editions)

import pandas as pd
file_path = 'Summer Olympic medallists 1896 to 2008 - IOC COUNTRY CODES.csv'
ioc_codes = pd.read_csv(file_path)
# Extract the relevant columns: ioc_codes
ioc_codes = ioc_codes[['Country' , 'NOC']]
print(ioc_codes.head())
print(ioc_codes.tail())

import pandas as pd
medals_dict = {}

for year in editions['Edition']:
    file_path = 'summer_{:d}.csv'.format(year)
    medals_dict[year] = pd.read_csv(file_path)
    # Extract relevant columns: medals_dict[year]
    medals_dict[year] = medals_dict[year][['Athlete', 'NOC', 'Medal']]
    # Assign year to column 'Edition' of medals_dict
    medals_dict[year]['Edition'] = year
# Concatenate medals_dict: medals
medals = pd.concat(medals_dict, ignore_index=True)
print(medals.head())
print(medals.tail())

##Counting medals by country/edition in a pivot table
# Construct the pivot_table: medal_counts
medal_counts = medals.pivot_table(aggfunc='count',index='Edition',values='Athlete',columns='NOC')
print(medal_counts.head())
print(medal_counts.tail())

##Computing fraction of medals per Olympic edition
# Set Index of editions: totals
totals = editions.set_index('Edition' )

# Reassign totals['Grand Total']: totals
totals = totals['Grand Total']

# Divide medal_counts by totals: fractions
fractions = medal_counts.divide(totals,axis='rows')
print(fractions.head())
print(fractions.tail())

##Computing percentage change in fraction of medals won
# Apply the expanding mean: mean_fractions
mean_fractions = fractions.expanding().mean()

# Compute the percentage change: fractions_change
fractions_change = mean_fractions.pct_change()*100

# Reset the index of fractions_change: fractions_change
fractions_change = fractions_change.reset_index('Edition')
print(fractions_change.head())
print(fractions_change.tail())

##Building hosts DataFrame
import pandas as pd

# Left join editions and ioc_codes: hosts
hosts = pd.merge(editions,ioc_codes,how='left')

# Extract relevant columns and set index: hosts
hosts = hosts[['Edition' , 'NOC']].set_index('Edition' )

# Fix missing 'NOC' values of hosts
print(hosts.loc[hosts.NOC.isnull()])
hosts.loc[1972, 'NOC'] = 'FRG'
hosts.loc[1980, 'NOC'] = 'URS'
hosts.loc[1988, 'NOC'] = 'KOR'

# Reset Index of hosts: hosts
hosts = hosts.reset_index()
print(hosts)


##Reshaping for analysis
import pandas as pd
# Reshape fractions_change: reshaped
reshaped = pd.melt(fractions_change,id_vars='Edition' , value_name='Change')
print(reshaped.shape, fractions_change.shape)

# Extract rows from reshaped where 'NOC' == 'CHN': chn
chn = reshaped.loc[reshaped.NOC == 'CHN']
print(chn.tail())

##Merging to compute influence
import pandas as pd

# Merge reshaped and hosts: merged
merged = pd.merge(reshaped , hosts,how='inner')
print(merged.head())

# Set Index of merged and sort it: influence
influence = merged.set_index('Edition').sort_index()
print(influence.head())

##Plotting influence of host country
# Import pyplot
import matplotlib.pyplot as plt

# Extract influence['Change']: change
change = influence['Change']

# Make bar plot of change: ax
ax = change.plot(kind='bar')

# Customize the plot to improve readability
ax.set_ylabel("% Change of Host Country Medal Count")
ax.set_title("Is there a Host Country Advantage?")
ax.set_xticklabels(editions['City'])
plt.show()
