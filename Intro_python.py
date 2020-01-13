##########################################
#list
##########################################
##same data types
# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50
areas = [hall,kit,liv,bed,bath]
print(areas)

##Combine with different types
areas = ["hallway", hall, "kitchen", kit, "living room", liv, "bedroom", bed, "bathroom", bath]
print(areas)

##Possitlbe list
[1 + 2, "a" * 5, 3]
[[1, 2, 3], [4, 5, 7]]
[1, 3, 4, 2]

##List of lists
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50
house = [["hallway", hall],
         ["kitchen", kit],
         ["living room", liv],
         ["bedroom",bed],
         ['bathroom',bath]]
print(house)
'''[["hallway",  11.25],
         ["kitchen", 18.0],
         ["living room",  20.0],
         ["bedroom",10.75],
         ['bathroom', 9.50]]'''
##########################################
#Subsetting Lists
##########################################
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Print out second element from areas
print(areas[1])

# Print out last element from areas
print(areas[-1])

# Print out the area of the living room
print(areas[5])

# Sum of kitchen and bedroom area: eat_sleep_area
eat_sleep_area = areas[3] + areas[7]
print(eat_sleep_area)

# Use slicing 
downstairs= areas[:6]
upstairs= areas[6:]
print(downstairs,upstairs)

#Replace list elements
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]
areas[9] = 10.5
areas[4] = "chill zone"

#Extend a list
areas = ["hallway", 11.25, "kitchen", 18.0, "chill zone", 20.0,
         "bedroom", 10.75, "bathroom", 10.50]
areas_1 = areas+ ["poolhouse", 24.5]
areas_2 = areas_1+ ["garage",15.45]

#Delete list elements
del(areas[10:11])

#Create list from the list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]
# Create areas_copy
areas_copy = list(areas)
# Change areas_copy
areas_copy[0] = 5.0
##########################################
#Function:type, len, int, float,str, help, max, min, sorted, bool, complex
##########################################
#functions
# Print out type of var1
print(type(var1))
# Print out length of var1
print(len(var1))
# Convert var2 to an integer: out2
out2 = int(var2)
#get help
help(max)
?max

##Multiple arguments
first = [11.25, 18.0, 20.0]
second = [10.75, 9.50]
# Paste together first and second: full
full = first+ second
# Sort full in descending order: full_sorted
full_sorted = sorted(full,reverse=True)
##########################################
#Built-in function: 
    #str: capitalize(),replace(),count(), upper()
    #float: bit_length(), conjugate()
    #list : index(),count(), append(), reverse()
##########################################
# string to experiment with: place
place = "poolhouse"

# Use upper() on place: place_up
place_up = place.upper()

# Print out place and place_up
print(place,place_up)

# Print out the number of o's in place
print(place.count('o'))

##List 
# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Print out the index of the element 20.0
print(areas.index(20))

# Print out how often 9.50 appears in areas
print(areas.count(9.5))

# Use append twice to add poolhouse and garage size
areas.append(24.5)
areas.append(15.45)

# Print out areas
print(areas)

# Reverse the orders of the elements in areas
areas.reverse()

# Print out areas
print(areas)
##########################################
#Packages
##########################################
import math
r = 0.43
C = 2*math.pi *r
A = math.pi*(r**2)
print("Circumference: " + str(C))
print("Area: " + str(A))

from math import radians
r = 192500
# Travel distance of Moon over 12 degrees. Store in dist.
dist = r*radians(12)
print(dist)

####################################################################################
#Numpy
####################################################################################
# Create a numpy array
import numpy as np
baseball = [180, 215, 210, 210, 188, 176, 209, 200]
np_baseball = np.array(baseball)
print(type(np_baseball))

#Create new array
import numpy as np
np_height_in = np.array(height_in)
np_height_m = np_height_in*0.0254

#bmi calculation
np_height_m = np.array(height_in) * 0.0254
np_weight_kg= np.array(weight_lb)*0.453592
bmi = np_weight_kg/(np_height_m**2)
print(bmi)

# Create the light array(answered in boolean)
light = np.array(bmi <21)

# Print out BMIs of all baseball players whose BMI is below 21(BMI under 21!)
print(bmi[bmi <21])

# Print out the weight at index 50
print(np_weight_lb[50])

# Print out sub-array of np_height_in: index 100 up to and including index 110
print(np_height_in[100:111])

##2D Numpy Arrays
# Create baseball, a list of lists
baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]
import numpy as np
np_baseball = np.array(baseball)
print(type(np_baseball))
print(np_baseball.shape)# Print out the shape

# Create np_baseball (2 cols)
np_baseball = np.array(baseball)
# Print out the 50th row of np_baseball
print(np_baseball[49,:])
# Select the entire second column of np_baseball: np_weight_lb
np_weight_lb = np_baseball[:,1]
# Print out height of 124th player
print(np_baseball[123, 0])

##array multiply(=product) calculation
# Create np_baseball (3 cols)
np_baseball = np.array(baseball)
# Print out addition of np_baseball and updated
print(np_baseball + updated)
# Create numpy array: conversion
conversion = np.array([0.0254, 0.453592 , 1])
# Print out product of np_baseball and c
print(np_baseball*conversion)

##Mean, Median, Std, corrcoef
#first column
np_height_in = np_baseball[:,0]
# Print out the mean of np_height_in
print(np.mean(np_height_in))
# Print out the median of np_height_in
print(np.median(np_height_in))
# Print mean height (first column)
avg = np.mean(np_baseball[:,0])
print("Average: " + str(avg))
# Print median height. Replace 'None'
med = np.median(np_baseball[:,0])
print("Median: " + str(med))
# Print out the standard deviation on height. Replace 'None'
stddev = np.std(np_baseball[:,0])
print("Standard Deviation: " + str(stddev))
# Print out correlation between first and second column. Replace 'None'
corr = np.corrcoef(np_baseball[:,0],np_baseball[:,1])
print("Correlation: " + str(corr))

np_positions = np.array(positions)
np_heights = np.array(heights)
gk_heights = np_heights[np_positions == 'GK']# Heights of the goalkeepers: gk_heights
other_heights = np_heights[np_positions != 'GK']# Heights of the other players: other_heights

# Print out the median height of goalkeepers. Replace 'None'
print("Median height of goalkeepers: " + str(np.median(gk_heights)))
# Print out the median height of other players. Replace 'None'
print("Median height of other players: " + str(np.median(other_heights)))
