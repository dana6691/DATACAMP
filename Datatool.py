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

#nonlocal????
def echo_shout(word):
    echo_word = word*2
    print(echo_word)
    def shout():
        nonlocal echo_word
        echo_word = echo_word + '!!!'
    shout()
    print(echo_word)
echo_shout('hello')