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










################################################

################################################


################################################################################################
#Grouping data
################################################################################################

################################################
#Grouping by multiple columns
################################################
# Aggregate  column of by_class by count
count_by_class = titanic.groupby('pclass')['survived'].count()
print(count_by_class)

# Group titanic by 'embarked' and 'pclass'
by_mult = titanic.groupby(['embarked' , 'pclass'])
# Aggregate 'survived' column of by_mult by count
count_mult = by_mult['survived'].count()
print(count_mult)
################################################
#Grouping by another series
################################################
#data import
life = pd.read_csv(life_fname, index_col='Country')
#data2 import
regions = pd.read_csv(regions_fname, index_col='Country')

# Group life by regions['region']: life_by_region
life_by_region = life.groupby(regions['region'])
# Print the mean over the '2010' column of life_by_region
print(life_by_region['2010'].mean())
################################################
#Groupby and aggregation
  #agg()
################################################
## Computing multiple aggregates of multiple columns
# Group titanic by 'pclass'
by_class = titanic.groupby('pclass')
# Select 'age' and 'fare'
by_class_sub = by_class[['age','fare']]
# Aggregate by_class_sub by 'max' and 'median': aggregated
aggregated = by_class_sub.agg(['max' , 'median'])
# Print the maximum age in each class
print(aggregated.loc[:, ('age','max')])
# Print the median fare in each class
print(aggregated.loc[:, ('fare','median')])

##Aggregating on index levels/fields
# Read the CSV file into a DataFrame and sort the index: gapminder
gapminder = pd.read_csv('gapminder.csv',index_col=['Year','region','Country']).sort_index()
# Group gapminder by : by_year_region
by_year_region = gapminder.groupby(level=['Year' , 'region'])
# Define the function to compute spread: spread
def spread(series):
    return series.max() - series.min()
# Create the dictionary: aggregator
aggregator = {'population':'sum', 'child_mortality':'mean', 'gdp':spread}
# Aggregate by_year_region using the dictionary: aggregated
aggregated = by_year_region.agg({'population':'sum', 'child_mortality':'mean' , 'gdp':spread})
# Print the last 6 entries of aggregated 
print(aggregated.tail(6))

##Grouping on a function of the index
#read file
sales = pd.read_csv('sales.csv',index_col='Date', parse_dates=True)
#groupby
by_day = sales.groupby(sales.index.strftime('%a'))
# Create sum: units_sum
units_sum = by_day['Units'].sum()
print(units_sum)
################################################
#Groupby and transformation
  #apply()
################################################
## Detecting outliers with Z-Scores
#standardized
standardized = gapminder_2010.groupby('region')['life','fertility'].transform(zscore)
#boolean series to identigy outliers
outliers= (standardized['life'] <-3)|(standardized['fertility']>3)
#Filter gapminder_2010 by the outliers
gm_outliers = gapminder_2010.loc[outliers]
print(gm_outliers)

## Filling missing data (imputation) by group
#groupby object: by_sex_class
by_sex_class = titanic.groupby(['sex','pclass'])
#function that imputes median
def impute_median(series):
    return series.fillna(series.median())
#Impute age and assign to titanic['age']
titanic.age = by_sex_class['age'].transform(impute_median)
# Print the output of titanic.tail(10)
print(titanic.tail(10))

## Other transformations with .apply
#Group gapminder_2010 by 'region': regional
regional = gapminder_2010.groupby('region')
#Apply the disparity function on regional: reg_disp
reg_disp = regional.apply(disparity)
# Print the disparity of 'United States', 'United Kingdom', and 'China'
print(reg_disp.loc[['United States','United Kingdom','China']])
################################################
#Groupby and filtering
    #groupby object:iteration
    #boolean groupby
################################################
## Grouping and filtering with .apply()
#Create a groupby object using titanic over the 'sex' column: by_sex
by_sex = titanic.groupby('sex')
#Call by_sex.apply with the function
def c_deck_survival(gr):
    c_passengers = gr['cabin'].str.startswith('C').fillna(False)
    return gr.loc[c_passengers, 'survived'].mean()

c_surv_by_sex = by_sex.apply(c_deck_survival)
#survival rates
print(c_surv_by_sex)

## Grouping and filtering with .filter()
#Read the CSV file
sales = pd.read_csv('sales.csv', index_col='Date', parse_dates=True)
#Group sales by 'Company':
by_company = sales.groupby('Company')
#Compute the sum of the 'Units' of by_company
by_com_sum = by_company['Units'].sum()
print(by_com_sum)
#Filter 'Units' where the sum is > 35: by_com_filt
by_com_filt = by_company.filter(lambda g:g['Units'].sum() > 35)
print(by_com_filt)

## Filtering and grouping with .map()
#Create the Boolean Series
under10 = (titanic['age'] < 10).map({True:'under 10', False:'over 10'})
#Group by under10 and compute the survival rate
survived_mean_1 = titanic.groupby(under10)['survived'].mean()
print(survived_mean_1)
#Group by under10 and pclass and compute the survival rate
survived_mean_2 = titanic.groupby([under10,'pclass'])['survived'].mean()
print(survived_mean_2)
################################################################################################
#Case study- Summer Olympics
################################################################################################
#Group and Aggregation
USA_edition_grouped = medals.loc[medals.NOC == 'USA'].groupby('Edition')
USA_edition_grouped['Medal'].count()

## Using .value_counts() for ranking
#Select the 'NOC' column of : country_names
country_names = medals['NOC']
#Count the number of medals won by each country: medal_counts
medal_counts = country_names.value_counts()
#Print top 15 countries ranked by medals
print(medal_counts.head(15))

## Using .pivot_table() to count medals by type
#Construct the pivot table: counted
counted = medals.pivot_table(index='NOC',columns='Medal',values='Athlete',aggfunc='count')
#Create the new column: counted['totals']
counted['totals'] = counted.sum(axis='columns')
#Sort counted by the 'totals' column
counted = counted.sort_values('totals', ascending=False)
#Print the top 15 rows of counted
print(counted.head(15))

## Applying .drop_duplicates()
#Select columns
ev_gen = medals[['Event_gender' , 'Gender']]
#Drop duplicate pairs: ev_gen_uniques
ev_gen_uniques = ev_gen.drop_duplicates()
print(ev_gen_uniques)

## Finding possible errors with .groupby()
#Group medals by the two columns: medals_by_gender
medals_by_gender = medals.groupby(['Event_gender' , 'Gender'])
#Create a DataFrame with a group count: medal_count_by_gender
medal_count_by_gender = medals_by_gender.count()
print(medal_count_by_gender)

## suspicious data
#Create the Boolean Series
sus = (medals.Event_gender == 'W') & (medals.Gender == 'Men')
#Create a DataFrame with the suspicious row
suspect = medals[sus]
print(suspect)

## Constructing alternative country rankings
## Using .nunique() to rank by distinct sports
#Group medals by 'NOC': country_grouped
country_grouped = medals.groupby('NOC')
#Compute the number of distinct sports in which each country won medals
Nsports = country_grouped['Sport'].nunique()
#Sort the values of Nsports in descending order
Nsports = Nsports.sort_values(ascending=False)
#Print the top 15 rows of Nsports
print(Nsports.head(15))

## Counting USA vs. USSR Cold War Olympic Sports
#Create a Boolean Series that is True when 'Edition' is between 1952 and 1988: during_cold_war
during_cold_war = (medals['Edition'] >= 1952) & (medals['Edition'] <= 1988)
#Extract rows for which 'NOC' is either 'USA' or 'URS': is_usa_urs
is_usa_urs = medals.NOC.isin(['USA', 'URS'])
#Use during_cold_war and is_usa_urs to create the DataFrame: cold_war_medals
cold_war_medals = medals.loc[during_cold_war & is_usa_urs]
#Group cold_war_medals by 'NOC'
country_grouped = cold_war_medals.groupby('NOC')
#Create Nsports
Nsports = country_grouped['Sport'].nunique().sort_values(ascending=False)
print(Nsports)

## Counting USA vs. USSR Cold War Olympic Medals
#Create the pivot table: medals_won_by_country
medals_won_by_country = pd.pivot_table(medals,values='Athlete', index=['Edition'],
                    columns=['NOC'], aggfunc='count')
#Slice medals_won_by_country
cold_war_usa_urs_medals = medals_won_by_country.loc[1952:1988, ['USA','URS']]
#Create most_medals 
most_medals = cold_war_usa_urs_medals.idxmax(axis='columns')

## Visualizing USA Medal Counts by Edition: Line Plot
#Create the DataFrame: usa
usa = medals[medals.NOC == 'USA']
#Group usa by ['Edition', 'Medal'] and aggregate over 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()
#Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')
#Plot the DataFrame usa_medals_by_year
usa_medals_by_year.plot()
plt.show()

## Area Plot
usa_medals_by_year.plot.area()
plt.show()

## Visualizing USA Medal Counts by Edition: Area Plot with Ordered Medals
#Redefine 'Medal' as an ordered categorical
medals.Medal = pd.Categorical(values = medals.Medal,categories=['Bronze', 'Silver', 'Gold'],ordered=True)
#Create the DataFrame: usa
usa = medals[medals.NOC == 'USA']
#Group usa by 'Edition', 'Medal', and 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()
#Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')
#Create an area plot of usa_medals_by_year
usa_medals_by_year.plot.area()
plt.show()