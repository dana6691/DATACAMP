################################################################################################
#User-defined function
################################################################################################
# Use print
def shout(word):
    shout_word = word + '!!!'
    print(shout_word)
shout('congratulations')

# Use return
def shout(word):
    shout_word = word + "!!!"
    return(shout_word)
yell = shout('congratulations')
print(yell)


##Multiple parameters
def shout(word1, word2):
    shout1 = word1 + '!!!'
    shout2 = word2 + '!!!'
    new_shout = shout1 + shout2
    return new_shout
yell = shout('congratulations', 'you')
print(yell)

##three-element tuple
num1, num2, num3 = nums
even_nums = (2, num2, num3)

##Tuple
def shout_all(word1, word2):
    shout1 = word1+"!!!"
    shout2= word2+"!!!"
    shout_words = (shout1,shout2)
    return shout_words
yell1,yell2 = shout_all('congratulations','you')
print(yell1)
print(yell2)

##final
import pandas as np
df = np.read_csv('tweets.csv')

def count_entries(df, col_name):
    langs_count = {}
    col = df[col_name]
    for entry in col:
        if entry in langs_count.keys():
            langs_count[entry]=langs_count[entry]+1
        else:
            langs_count[entry]=1
    return langs_count
result = count_entries(tweets_df,'lang')
print(result)
################################################################################################
#scope and UDF
################################################################################################
team = "teen titans"
def change_team():
    global team
    team = "justice league"
print(team)

change_team() #print global
print(team) #print inside
################################################
##Nested function
################################################
#Tuple 
def three_shouts(word1, word2, word3):
    def inner(word):
        return word + '!!!'
    return (inner(word1), inner(word2), inner(word3))
print(three_shouts('a', 'b', 'c'))

#Nested
def echo(n):
    def inner_echo(word1):
        echo_word = word1 * n
        return echo_word
    return inner_echo
twice = echo(2)
thrice = echo(3)
print(twice('hello'), thrice('hello'))

#nonlocal and nested
def echo_shout(word):
    echo_word = word*2
    print(echo_word)
    def shout():
        nonlocal echo_word
        echo_word = echo_word + '!!!'
    shout()
    print(echo_word)
echo_shout('hello')
################################################
##Functions with one default argument (1)
################################################
def shout_echo(word1, echo=1):
    echo_word = word1*echo
    shout_word = echo_word + "!!!"
    return shout_word
no_echo = shout_echo("Hey")
with_echo = shout_echo("Hey",echo=5)
print(no_echo)
print(with_echo)
################################################
##Functions with multiple default argument (2)
################################################
def shout_echo(word1, echo=1,intense=False):
    echo_word = word1 * echo
    # Make echo_word uppercase if intense is True
    if intense is True:
        echo_word_new = echo_word.upper() + '!!!'
    else:
        echo_word_new = echo_word + '!!!'
    return echo_word_new
with_big_echo = shout_echo("Hey",echo=5,intense=True)
big_no_echo = shout_echo("Hey",intense=True)
print(with_big_echo)
print(big_no_echo)
################################################
##Functions with variable-length arguments (*args) -- use for tuple
################################################
def gibberish(*args):
    hodgepodge = ''
    for word in args:
        hodgepodge += word
    return hodgepodge

one_word = gibberish("luke")
many_words = gibberish("luke", "leia", "han", "obi", "darth")
print(one_word)
print(many_words)
'''"
luke
lukeleiahanobidarth"'''
################################################
##Functions with variable-length keyword arguments (**kwargs) -- use for dictionary
################################################
def report_status(**kwargs):
    print("\nBEGIN: REPORT\n")
    for keys, values in kwargs.items():
        # Print out the keys and values, separated by a colon ':'
        print(keys + ": " + values)
    print("\nEND REPORT")

report_status(name="luke", affiliation="jedi" , status="missing")
report_status(name="anakin", affiliation="sith lord", status="deceased")
'''
BEGIN: REPORT
name="luke"
affiliation="jedi"
status="missing"
END REPORT
'''
################################################
#All together
################################################
# Define count_entries()
def count_entries(df, col_name):
    cols_count = {}
    col = df[col_name]
    for entry in col:
        if entry in cols_count.keys():
            cols_count[entry] += 1
        else:
            cols_count[entry] = 1
    return cols_count
result1 = count_entries(tweets_df,'lang')
result2 = count_entries(tweets_df,'source')
print(result1)
print(result2)

def count_entries(df, *args):
    cols_count = {}

    for col_name in args:
        col = df[col_name]
        for entry in col:
            if entry in cols_count.keys():
                cols_count[entry] += 1
            else:
                cols_count[entry] = 1
    return cols_count

result1 = count_entries(tweets_df, 'lang')
result2 = count_entries(tweets_df, 'lang', 'source')
print(result1)
print(result2)
################################################################################################
#Lambda functions and error-handling
################################################################################################
echo_word = (lambda word1,echo : word1*echo)
result = echo_word('hey',5)
print(result)

##Map() and lambda functions
spells = ["protego", "accio", "expecto patronum", "legilimens"]
shout_spells = map(lambda item: item + '!!!'  ,spells)
shout_spells_list = list(shout_spells) #convert to list
print(shout_spells_list)

##Filter() and lambda functions
fellowship = ['frodo', 'samwise', 'merry', 'pippin', 'aragorn', 'boromir', 'legolas', 'gimli', 'gandalf']
result = filter(lambda member: len(member) >6, fellowship)
result_list = list(result)
print(result_list)

##Concatenate word with Reduce()
from functools import reduce 
stark = ['robb', 'sansa', 'arya', 'brandon', 'rickon']
result = reduce(lambda item1, item2: item1+item2, stark)
print(result)
''' robbsansaaryabrandonrickon '''
################################################
#Error handling
################################################
##Error handling with try-except
def shout_echo(word1,echo=1):
    echo_word =''
    shout_words=''
    try:
        echo_word = word1*echo
        shout_words = echo_word + '!!!'
    except:
        # Print error message
        print("word1 must be a string and echo must be an integer.")
    return shout_words
shout_echo("particle", echo="accelerator")

##Error handling by raising an error
def shout_echo(word1, echo=1):
    if echo < 0:
        raise ValueError('echo must be greater than or equal to 0')
    echo_word = word1 * echo
    shout_word = echo_word + '!!!'
    return shout_word

shout_echo("particle", echo=5)
################################################
#Bring it all together
################################################
# first 2 characters in a tweet x are 'RT'
result = filter(lambda x: x[0:2] == "RT", tweets_df['text'])
res_list = list (result)
for tweet in res_list:
    print(tweet)

"""Return a dictionary with counts of
    occurrences as value for each key."""
def count_entries(df, col_name='lang'):
    # Initialize an empty dictionary: cols_count
    cols_count = {}

    try:
        # Extract column from DataFrame: col
        col = df[col_name]
        
        for entry in col:
            if entry in cols_count.keys():
                cols_count[entry] += 1
            else:
                cols_count[entry] = 1

        return cols_count
    except:
        'The DataFrame does not have a ' + col_name + ' column.'

result1 = count_entries(tweets_df, 'lang')
print(result1)


"""Return a dictionary with counts of
    occurrences as value for each key."""
def count_entries(df, col_name='lang'):
    # Raise a ValueError if col_name is NOT in DataFrame
    if col_name not in df.columns:
        raise ValueError('The DataFrame does not have a ' + col_name + ' column.')
    
    cols_count = {}
    col = df[col_name]
    
    for entry in col:
        if entry in cols_count.keys():
            cols_count[entry] += 1
        else:
            cols_count[entry] = 1
    return cols_count

result1= count_entries(tweets_df,'lang')
print(result1)
################################################################################################
#Python Data Science Toolbox (Part 2)
#Iterable
    #Examples: lists, strings, dictionaries, file connections
#Iterator
    #Produces next value with 'next()'
################################################################################################
##Iterating over lists
flash = ['jay garrick', 'barry allen', 'wally west', 'bart allen']
#Method (1)
for person in flash:
    print(person)
#Method (2)
superhero = iter(flash)
print(next(superhero))  #jay garrick
print(next(superhero)) #barry allen
print(next(superhero))
print(next(superhero))

##Iterating at once with *
word = 'Data'
it = iter(word)
print(*it)  # D a t a 

##Iterating over range
#Method (1)
small_value = iter(range(3))
print(next(small_value))
print(next(small_value))
print(next(small_value))
#Method (2)
for num in range(3):
    print(num)

##iterator for range(10 ** 100): googol
googol = iter(range(10**100))
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))

##Iterators as function arguments
values = range(10,21)
print(values)
values_list = list(values) #create a list
print(values_list)
values_sum = sum(values) #sum of values
print(values_sum)
################################################
#Playing with iterators
    #enumerate(): make a list of tuples, put number in front of the list
    #zip(): zipped multiple lists into iterators of tuples
################################################
## enumerate example
mutants = ['charles xavier', 
            'bobby drake', 
            'kurt wagner', 
            'max eisenhardt', 
            'kitty pryde']
# Create a list of tuples: mutant_list
mutant_list = list(enumerate(mutants))
print(mutant_list)
# Unpack and print the tuple pairs
for index1, value1 in enumerate(mutants):
    print(index1, value1)
# Change the start index
for index2, value2 in enumerate(mutants,1):
    print(index2, value2)

## zip example
# Create a list of tuples
mutant_data = list(zip(mutants, aliases, powers))
print(mutant_data)

# Create a zip object 
mutant_zip = zip(mutants, aliases, powers)
print(mutant_zip)
# Unpack the zip object 
for value1, value2, value3 in mutant_zip:
    print(value1, value2, value3)

# Create a zip object and unpack all
z1 = zip(mutants, powers)
print(*z1)

# Create a zip object and 'Unzip' the tuples
z1 = zip(mutants, powers)
result1, result2 = zip(*z1)
print(result1)
print(result2)
################################################
#iterators to load large files into memory
    #Condition
    #Dictionary
################################################
#iterate over the chunk, iterate over the column, 'lang'
counts_dict = {}
for chunk in pd.read_csv('tweets.csv', chunksize=10):
    for entry in chunk['lang']:
        if entry in counts_dict.keys():
            counts_dict[entry] += 1
        else:
            counts_dict[entry] = 1
print(counts_dict)
#define function
def count_entries(csv_file, c_size, colname):
    counts_dict = {}

    for chunk in pd.read_csv(csv_file, chunksize=c_size):
        for entry in chunk[colname]:
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1
    return counts_dict

result_counts = count_entries('tweets.csv',10,'lang')
print(result_counts)
################################################################################################
#List comprehensions: []
    #- collapse FOR loops for building lists into a single line
    #- components: iterable, literator  variable, output expression
################################################################################################
## only first character of list elements
doctor = ['house', 'cuddy', 'chase', 'thirteen', 'wilson']
[doc[0] for doc in doctor]

## Create list comprehension: squares
squares = [i**2 for i in range(0,10)]

## Create 5x5 matrix
matrix = [[col for col in range(5)] for row in range(5)]
for row in matrix:
    print(row)

## Condition(if) in comprehensions
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']
new_fellowship = [member for member in fellowship if len(member) >= 7]
print(new_fellowship)

## Condition(if-else) in comprehensions
new_fellowship = [member if len(member) >= 7 else '' for member in fellowship]
print(new_fellowship)

## Dictionary in comprehensions
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']
new_fellowship = {member: len(member) for member in fellowship }
print(new_fellowship)
################################################
#List comprehensions:[] vs generators:()
################################################
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']
# List comprehension
fellow1 = [member for member in fellowship if len(member) >= 7]
# Generator expression
fellow2 = (member for member in fellowship if len(member) >= 7)


## Generator (1)
result = (num for num in range(0,31))
# Print the first 5 values
print(next(result))
print(next(result))
print(next(result))
print(next(result))
print(next(result))
# Print the rest
for value in result:
    print(value)

## Generator (2)
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']
lengths = (len(person) for person in lannister) 
for value in lengths:
    print(value)

## Generator (3): defined
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']
def get_lengths(input_list):
    for person in input_list:
        yield len(person)
#print
for value in get_lengths(lannister):
    print(value)
################################################
#List comprehensions example
################################################
#list comprehension
tweet_time = df['created_at']
tweet_clock_time = [entry[11:19] for entry in tweet_time]
print(tweet_clock_time)

#Condition with list comprehension
tweet_clock_time = [entry[11:19] for entry in tweet_time if entry[17:19] == '19']
print(tweet_clock_time)


################################################################################################
#case study: world bank data
################################################################################################
zipped_lists = zip(feature_names, row_vals)

# Create a dictionary: rs_dict
rs_dict = dict(zipped_lists)
print(rs_dict)

# Define lists2dict()
def lists2dict(list1, list2):
    zipped_lists = zip(list1, list2)
    rs_dict = dict(zipped_lists)
    return rs_dict

rs_fxn = lists2dict(feature_names,row_vals)
print(rs_fxn)

#Turn list of lists into list of dicts
list_of_dicts = [lists2dict(feature_names, sublist) for sublist in row_lists]
print(list_of_dicts[0])# Print the first two dictionaries in list_of_dicts
print(list_of_dicts[1])

#Turn into DataFrame
import pandas as pd 
df = pd.DataFrame(list_of_dicts)
print(df.head())

##Processing data in chunks
# Open a connection to the file
with open('world_dev_ind.csv') as file:

    # Skip the column names
    file.readline()

    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Process only the first 1000 rows
    for j in range(0, 1000):

        # Split the current line into a list: line
        line = file.readline().split(',')

        # Get the value for the first column: first_col
        first_col = line[0]

        # If the column value is in the dict, increment its value
        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1

        # Else, add to the dict and set value to 1
        else:
            counts_dict[first_col] = 1

# Print the resulting dictionary
print(counts_dict)


## Writing generator
# Define read_large_file()
def read_large_file(file_object):
    # Loop indefinitely until the end of the file
    while True:
        # Read a line from the file: data
        data = file_object.readline()

        # Break if this is the end of the file
        if not data:
            break

        # Yield the line of data
        yield data
        
# Open a connection to the file
with open('world_dev_ind.csv') as file:
    # Create a generator object for the file: gen_file
    gen_file = read_large_file(file)

    # Print the first three lines of the file
    print(next(gen_file))
    print(next(gen_file))
    print(next(gen_file))


# Initialize an empty dictionary: counts_dict
counts_dict = {}
# Open a connection to the file
with open('world_dev_ind.csv') as file:

    # Iterate over the generator from read_large_file()
    for line in read_large_file(file):

        row = line.split(',')
        first_col = row[0]

        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1
        else:
            counts_dict[first_col] = 1

# Print            
print(counts_dict)


##Writing an iterator to load data in chunks
import pandas as pd
# Initialize reader object: df_reader
df_reader = pd.read_csv('ind_pop.csv', chunksize=10)

# Print two chunks
print(next(df_reader))
print(next(df_reader))

##Writing an iterator to load data in chunks (2)
# Initialize reader object: urb_pop_reader
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)

# Get the first DataFrame chunk: df_urb_pop
df_urb_pop = next(urb_pop_reader)

# Check out the head of the DataFrame
print(df_urb_pop.head())

# Check out specific country: df_pop_ceb
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']

# Zip DataFrame columns of interest: pops
pops = zip(df_pop_ceb['Total Population'] , df_pop_ceb['Urban population (% of total)'])

# Turn zip object into list: pops_list
pops_list = list(pops)

# Print pops_list
print(pops_list)

##Writing an iterator to load data in chunks (3)
# Code from previous exercise
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)
df_urb_pop = next(urb_pop_reader)
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']
pops = zip(df_pop_ceb['Total Population'], 
           df_pop_ceb['Urban population (% of total)'])
pops_list = list(pops)

# Use list comprehension to create new DataFrame column 'Total Urban Population'
df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]

# Plot urban population data
df_pop_ceb.plot(kind='scatter', x='Year', y='Total Urban Population')
plt.show()

##Writing an iterator to load data in chunks (4)
# Initialize reader object: urb_pop_reader
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)

# Initialize empty DataFrame: data
data = pd.DataFrame()

# Iterate over each DataFrame chunk
for df_urb_pop in urb_pop_reader:

    # Check out specific country: df_pop_ceb
    df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']

    # Zip DataFrame columns of interest: pops
    pops = zip(df_pop_ceb['Total Population'],
                df_pop_ceb['Urban population (% of total)'])

    # Turn zip object into list: pops_list
    pops_list = list(pops)

    # Use list comprehension to create new DataFrame column 'Total Urban Population'
    df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]
    
    # Append DataFrame chunk to data: data
    data = data.append(df_pop_ceb)

# Plot urban population data
data.plot(kind='scatter', x='Year', y='Total Urban Population')
plt.show()


##Writing an iterator to load data in chunks (5)
# Define plot_pop()
def plot_pop(filename, country_code):

    # Initialize reader object: urb_pop_reader
    urb_pop_reader = pd.read_csv(filename, chunksize=1000)

    # Initialize empty DataFrame: data
    data = pd.DataFrame()
    
    # Iterate over each DataFrame chunk
    for df_urb_pop in urb_pop_reader:
        # Check out specific country: df_pop_ceb
        df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == country_code]

        # Zip DataFrame columns of interest: pops
        pops = zip(df_pop_ceb['Total Population'],
                    df_pop_ceb['Urban population (% of total)'])

        # Turn zip object into list: pops_list
        pops_list = list(pops)

        # Use list comprehension to create new DataFrame column 'Total Urban Population'
        df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]
    
        # Append DataFrame chunk to data: data
        data = data.append(df_pop_ceb)

    # Plot urban population data
    data.plot(kind='scatter', x='Year', y='Total Urban Population')
    plt.show()

# Set the filename: fn
fn = 'ind_pop_data.csv'

# Call plot_pop for country code 'CEB'
plot_pop(fn,country_code='CEB')

# Call plot_pop for country code 'ARB'
plot_pop(fn,country_code='ARB')