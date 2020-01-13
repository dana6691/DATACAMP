################################################
#Dictionaries
################################################
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']
# Get index of 'germany'
ind_ger = countries.index('germany')
# Use ind_ger to print out capital of Germany
print(capitals[ind_ger])

##Create dictionary
europe = { 'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }
print(europe)

# Print out the keys in europe
print(europe.keys())

# Print out value that belongs to key 'norway'
print(europe['norway'])

##Add, Update, Delete
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }
# Add italy to europe
europe['italy'] ='rome'
# Do they have italy in europe ? True/False
print('italy' in europe)
# Update capital of germany
europe['germany'] ='berlin'
# Remove australia
del(europe['australia'])

#Sub-disctionary
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }
# Print out the capital of France
print(europe['france'])
# Create sub-dictionary data
data = {'capital':'rome','population':59.83}
# Add data to europe under key 'italy'
europe['italy']=data
# Print europe
print(europe)
################################################
#Pandas
################################################
# Create dictionary my_dict 
import pandas as pd
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
my_dict = {'country':names,
    'drives_right':dr,
    'cars_per_cap':cpc
}
# Build a DataFrame cars
cars=pd.DataFrame(my_dict)
print(cars)

#Create Row Labels
import pandas as pd
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
cars_dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }
cars = pd.DataFrame(cars_dict)
print(cars)
row_labels = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']
cars.index=row_labels
print(cars)

##import csv file
import pandas as pd
cars= pd.read_csv('cars.csv')

##First column as a row index
cars = pd.read_csv('cars.csv',index_col=0)

#Select Column in (Series, Dataframe)
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Print out country column as Pandas Series
print(cars['country'])
# Print out country column as Pandas DataFrame
print(cars[['country']])
# Print out DataFrame with country and drives_right columns
print(cars[['country','drives_right']])

##Select rows
# Print out first 3 observations
print(cars[0:3])
# Print out fourth, fifth and sixth observation
print(cars[3:6])

##Print rows as a Series
print(cars.loc['JPN'])
print(cars.loc[['AUS', 'EG']])
## print row as a DataFrame
print(cars.iloc[2])
print(cars.iloc[1,6])

## Print out drives_right value of Morocco
print(cars.loc[['MOR'],'drives_right'])
## Print sub-DataFrame(row index, column names)
print(cars.loc[['RU', 'MOR'],['country','drives_right']])


# Print out drives_right column as DataFrame
print(cars.loc[:, ['drives_right']])
cars.iloc[:, [2]]
# Print out cars_per_cap and drives_right as DataFrame
print(cars.loc[:, ['drives_right','cars_per_cap']])
################################################################################################
#Logic, Control Flow and Filtering
    #logical_and()
    #logical_or()
    #logical_not()
    #np.logical_and(bmi > 21, bmi < 22) 
    #bmi[np.logical_and(bmi > 21, bmi < 22)]
################################################################################################
# Comparison of booleans
print(True == False)

# Comparison of integers
print(-5 * 15 != 75)

# Comparison of strings
print("pyscript" == "PyScript")

# Compare a boolean with a numeric
print(True == 1)

# Comparison of integers
x = -3 * 6
print(x>=-10)

# Comparison of strings
y = "test"
print("test"<= y)

# Comparison of booleans
print(True >False)
################################################
#Filtering Pandas DataFrame
################################################
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
dr = cars["drives_right"]# Extract drives_right column as Series: dr
sel = cars[dr]# subset cars with True of "drives_right"
print(sel)

#easier way
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
sel = cars[cars['drives_right']]

# Subset 2(true/false)
car_maniac = cars[cars["cars_per_cap"] > 500]

# Subset 3(true/false)
medium = cars[np.logical_and(100 < cars["cars_per_cap"],  cars["cars_per_cap"]< 500 )]
################################################
#while loop
################################################
# Initialize offset
offset = -6

# Code the while loop
while offset != 0 :
    print("correcting...")
    if offset > 0:
      offset = offset -1
    else : 
      offset = offset +1    
    print(offset)
################################################
#for loop
################################################   
#One parameter 
areas = [11.25, 18.0, 20.0, 10.75, 9.50]
for i in areas:
    print(i)

#Two parameter 
for index, area in enumerate(areas) :
    print("room " + str(index+1) + ": " + str(area))

#Loop over list of lists
house = [["hallway", 11.25], 
         ["kitchen", 18.0], 
         ["living room", 20.0], 
         ["bedroom", 10.75], 
         ["bathroom", 9.50]]
for x in house:
    print("the " + x[0] + " is " + str(x[1]) + " sqm")

#Dictionary Loop
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }
for key,value in europe.items():
    print("the capital of " + key + " is " + value)

#1-D array
import numpy as np
for x in height:
    print(str(x) + "inches")

#2-D array
for x in np.nditor(baseball):
    print(x)
################################################
##Iterate over dataframe
################################################
# Iterate over rows of cars
for lab,row in cars.iterrows():
    print(lab) #each row index label
    print(row)

# Adapt for loop
for lab, row in cars.iterrows() :
    print(lab + ": " + str(row['cars_per_cap']))

# Add column
for lab,row in cars.iterrows():
    cars.loc[lab, "COUNTRY"] = row["country"].upper()

# Add column(2) = same but shorter
cars["COUNTRY"] = cars["country"].apply(str.upper)
################################################################################################
#Case Study: Hacker Statistics
################################################################################################
#Random generator
import numpy as np
np.random.seed(123)
print(np.random.rand())

# Simulate a dice (1-6)
print(np.random.randint(1,7))

# Finish the control construct
if dice <= 2 :
    step = step - 1
elif dice <=5 :
    step= step +1
else :
    step = step + np.random.randint(1,7)
print(dice,step)


## Random Walk
all_walks = []
for i in range(10) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)
    all_walks.append(random_walk)
np_aw = np.array(all_walks)

# Plot np_aw and show
plt.plot(np_aw)
plt.show()
plt.clf()

# Transpose np_aw: np_aw_t
np_aw_t = np.transpose(np_aw)
plt.plot(np_aw_t)
plt.show()