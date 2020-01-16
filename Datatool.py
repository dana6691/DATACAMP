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
    #Examples: lists, strings, dictionaries, le connections
#Iterator
    #Produces next value with next()
################################################################################################
##Iterating over iterables
flash = ['jay garrick', 'barry allen', 'wally west', 'bart allen']
for person in flash:
    print(person)
superhero = iter(flash)
print(next(superhero))
print(next(superhero))
print(next(superhero))
print(next(superhero))

##Iterating over iterables (2)
# Create an iterator for range(3): small_value
small_value = iter(range(3))
print(next(small_value))
print(next(small_value))
print(next(small_value))
# Loop over range(3) and print the values
for num in range(3):
    print(num)
# Create an iterator for range(10 ** 100): googol
googol = iter(range(10**100))
# Print the first 5 values from googol
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))

##Iterators as function arguments
values = range(10,21)
print(values)
# Create a list of integers: values_list
values_list = list(values)
print(values_list)
# Get the sum of values: values_sum
values_sum = sum(values)
print(values_sum)
################################################
#Playing with iterators
    #enumerate
    #zip()
################################################
mutants = ['charles xavier', 
            'bobby drake', 
            'kurt wagner', 
            'max eisenhardt', 
            'kitty pryde']

# Create a list of tuples: mutant_list
mutant_list = list(enumerate(mutants))

# Print the list of tuples
print(mutant_list)


# Unpack and print the tuple pairs
for index1, value1 in enumerate(mutants):
    print(index1, value1)

# Change the start index
for index2, value2 in enumerate(mutants,1):
    print(index2, value2)