
################################################
#Frequency count
################################################
# Print the value counts for 'Borough'
print(df['Borough'].value_counts(dropna=False))
################################################
#Plots
################################################
#Histogram
import matplotlib.pyplot as plt
print(df['Existing Zoning Sqft'].describe())# Describe the column
df['Existing Zoning Sqft'].plot(kind='hist', rot=70, logx=True, logy=True)# Plot the histogram
plt.show()
#Boxplot
import pandas as pd
df.boxplot(column='initial_cost', by='Borough', rot=90)
plt.show()
#scatter plot
df.plot(kind='scatter', x='initial_cost', y='total_est_fee', rot=70)
plt.show()
################################################
#Tidy Data
################################################
airquality_melt = pd.melt(airquality, id_vars=['Month', 'Day'])# Melt airquality
print(airquality_melt.head())
airquality_melt = pd.melt(airquality, id_vars=['Month', 'Day'], var_name='measurement', value_name='reading') #rename variable and value
print(airquality_melt.head())
################################################
#Pivotting data (Un-melting)
################################################
# Pivot airquality_melt: airquality_pivot
airquality_pivot = airquality_melt.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading') #measurement becomes column
print(airquality_pivot.head())

# Resetting index after pivotting
print(airquality_pivot.index)# Print the index 
airquality_pivot_reset = airquality_pivot.reset_index()# Reset the index
print(airquality_pivot_reset.index)# Print the new index
print(airquality_pivot_reset.head())

#pivoting duplicate values
airquality_pivot = airquality_dup.pivot_table(index=['Month','Day'], columns='measurement', values='reading', aggfunc=np.mean)
print(airquality_pivot.head()) #before reset_index
airquality_pivot = airquality_pivot.reset_index()# Reset the index
print(airquality_pivot.head())
print(airquality.head())
################################################
#Parsing(one column into many columns)
################################################
#split with .str
tb_melt = pd.melt(tb, id_vars=['country', 'year'])# Melt 
tb_melt['gender'] = tb_melt.variable.str[0]# Create the 'gender' column (first letter of the variable column)
tb_melt['age_group'] = tb_melt.variable.str[1:]# Create the 'age_group' column(slicing the rest of the variable column)
print(tb_melt.head())

#split with split(), .get()
ebola_melt = pd.melt(ebola, id_vars=['Date', 'Day'], var_name='type_country', value_name='counts')
ebola_melt['str_split'] = ebola_melt.type_country.str.split('_')# Create the 'str_split' column
ebola_melt['type'] = ebola_melt.str_split.str.get(0)# Create the 'type' column
ebola_melt['country'] = ebola_melt.str_split.str.get(1)# Create the 'country' column
print(ebola_melt.head())
################################################
#Concatenating
################################################
#combine row
row_concat = pd.concat([uber1, uber2, uber3])# Row Concatenate uber1, uber2, and uber3
print(row_concat.shape)
print(row_concat.head())
#combine column
ebola_tidy = pd.concat([ebola_melt, status_country],axis=1)
print(ebola_tidy.shape)
print(ebola_tidy.head())
#select the row index =0 only
concatenated = pd.concat([weather_p1],[weather_p2])
concatenated = concatenated.loc[0,:]
#combining multiple files into one file
import glob
import pandas as pd
pattern = '*.csv' #all the csv files , #'part_?.csv' = file names with part_1, part_2....
csv_files = glob.glob(pattern) # Save all file
frames = [] # Create an empty list
for csv in csv_files:
    df = pd.read_csv(csv)
    frames.append(df)# Append df to frames

uber = pd.concat(frames)# Concatenate frames into a single DataFrame
print(uber.shape)
print(uber.head())
#Merge data
o2o = pd.merge(left=site, right=visited, left_on='name', right_on='site') #same variable, different name('name', 'site')
print(o2o)
################################################
#Cleaning data for analysis
################################################
#convert data types
tips.smoker = tips.smoker.astype('category')# Convert the smoker column to type 'category'
print(tips.info())
#convert to numeric
tips['total_bill'] = pd.to_numeric(tips['total_bill'], errors='coerce') # Convert 'total_bill' to a numeric dtype #error to missing value
print(tips.info())
#string manipulation
import re
prog = re.compile('\d{3}-\d{3}-\d{4}') #format xxx-xxx-xxxx
result = prog.match('123-456-7890')# See if the pattern matches
print(bool(result)) #true
result2 =  prog.match('1123-456-7890')# See if the pattern matches
print(bool(result2))#false
pattern2 = bool(re.match(pattern='\$\d*\.\d{2}', string='$123.45'))# Write the second pattern
print(pattern2)
pattern3 = bool(re.match(pattern='\w*', string='Australia'))# Write the third pattern
print(pattern3)
#Extracting numeric value from strings
import re
matches = re.findall('\d+', 'the recipe calls for 10 strawberries and 1 banana') # Find the numeric values
print(matches) #answer = 10,1

#Use function to clean data
def recode_gender(gender):
    # Return 0 if gender is 'Female'
    if gender == 'Female':
        return 0   
    # Return 1 if gender is 'Male'    
    elif gender == 'Male':
        return 1   
    # Return np.nan    
    else:
        return np.nan
tips['recode'] = tips.sex.apply(recode_gender)# Apply the function to the sex column



#Drop duplicate data
tracks = billboard[['year', 'artist', 'track', 'time']]# Create the new DataFrame: tracks
print(tracks.info())
tracks_no_duplicates = tracks.drop_duplicates()# Drop the duplicates:
print(tracks_no_duplicates.info())
#Drop NA
tracks = tracks.dropna()
#Replace NA with mean
oz_mean = airquality['Ozone'].mean()# Calculate the mean
airquality['Ozone'] = airquality['Ozone'].fillna(oz_mean)# Replace all the missing values in the Ozone column with the mean
print(airquality.info())
#Testing dataset
assert pd.notnull(ebola).all().all()# Assert that there are no missing values
assert (ebola >= 0).all().all()# Assert that all values are >= 0
################################################
#Final Project
################################################
g1800s.head()
g1800s.info()
g1800s.describe()
g1800s.columns
g1800s.shape
import matplotlib.pyplot as plt
g1800s.plot(kind='scatter', x='1800', y='1899')# Create the scatter plot
plt.xlabel('Life Expectancy by Country in 1800')
plt.ylabel('Life Expectancy by Country in 1899')
plt.xlim(20, 55)# Specify axis limits
plt.ylim(20, 55)
plt.show()
#akes a row of data, drops all missing values/if all remaining values are greater than or equal to 0
def check_null_or_valid(row_data):
    no_na = row_data.dropna()
    numeric = pd.to_numeric(no_na)
    ge0 = numeric >= 0
    return ge0

# Check whether the first column is 'Life expectancy'
assert g1800s.columns[0] == 'Life expectancy'

# Check whether the values in the row are valid
assert g1800s.iloc[:, 1:].apply(check_null_or_valid, axis=1).all().all()

# Check that there is only one instance of each country
assert g1800s['Life expectancy'].value_counts()[0] == 1

# Concatenate the DataFrames column-wise
gapminder = pd.concat([g1800s, g1900s, g2000s],axis=1)
print(gapminder.shape)
print(gapminder.head())

# Melt/column rename
import pandas as pd
gapminder: gapminder_melt
gapminder_melt = pd.melt(gapminder,id_vars='Life expectancy')
gapminder_melt.columns = ['country', 'year', 'life_expectancy']
print(gapminder_melt.head())

# Convert the year column to numeric
gapminder.year = pd.to_numeric(gapminder.year)

# Test type object
assert gapminder.country.dtypes == np.object
assert gapminder.year.dtypes==np.int64
assert gapminder.life_expectancy.dtypes==np.float64

countries = gapminder['country']

# Drop all the duplicates
countries = countries.drop_duplicates()

# Create the Boolean vector
pattern = '^[A-Za-z\.\s]*$'
mask = countries.str.contains(pattern)

# Invert the mask: mask_inverse
mask_inverse = ~mask
invalid_countries = countries.loc[mask_inverse]# Subset countries 

# Assert that country does not contain any missing values
assert pd.notnull(gapminder.country).all()
assert gapminder['year'].notnull().all()

# Drop the missing values
gapminder = gapminder.dropna()
print(gapminder.shape)

# Add first subplot
plt.subplot(2, 1, 1) 
gapminder.life_expectancy.plot(kind='hist')# Create a histogram 
gapminder_agg = gapminder.groupby('year')['life_expectancy'].mean()# Group gapminder
print(gapminder_agg.head())
print(gapminder_agg.tail())

# Add second subplot
plt.subplot(2, 1, 2)
gapminder_agg.plot()# Create a line plot 
plt.title('Life expectancy over the years')# Add title and specify axis labels
plt.ylabel('Life expectancy')
plt.xlabel('Year')
plt.tight_layout()
plt.show()

# Save both DataFrames to csv files
gapminder.to_csv('gapminder.csv')
gapminder_agg.to_csv('gapminder_agg.csv')
