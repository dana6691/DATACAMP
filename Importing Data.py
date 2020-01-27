################################################################################################
###########################Introduction to Importing Data in Python############################# 
#Introduction and flat files
################################################################################################
! ls #find the list

# Open a file: file
file = open('moby_dick.txt', 'r')
print(file.read())

# Check whether file is closed
print(file.closed)

# Close file
file.close()

# Read & print the first 3 lines
with open('moby_dick.txt') as file:
    print(file.readline())
    print(file.readline())
    print(file.readline())
################################################
#Numpy
    # comma-delimited: ','
    # tab-delimited: '\t'
################################################
#import data and reshape
import numpy as np
file = 'digits.csv'
digits = np.loadtxt(file, delimiter=',') # comma-delimited
print(type(digits))# Print datatype of digits
im = digits[21, 1:]# Select and reshape a row
im_sq = np.reshape(im, (28, 28))
plt.imshow(im_sq, cmap='Greys', interpolation='nearest')# Plot reshaped data
plt.show()

#import data with skip 1 low, only first and third columns
import numpy as np
file = 'digits_header.txt'
data = np.loadtxt(file, delimiter='\t', skiprows=1, usecols=[0,2]) # tab-delimited
print(data)

#import plot
file = 'seaslug.txt'
data_float = np.loadtxt(file, delimiter='\t', dtype=float, skiprows=1)# Import data
print(data_float[9])
plt.scatter(data_float[:, 0], data_float[:, 1])# Plot
plt.xlabel('time (min.)')
plt.ylabel('percentage of larvae')
plt.show()

# mixed datatypes 
file = 'titanic.csv'
d=np.recfromcsv(file,delimiter=',',dtype=None,names=True)
print(d[:3])
################################################
#Pandas
################################################
#import data
import pandas as pd
file = 'titanic.csv'
df = pd.read_csv(file)
print(df.head())

# Read the first 5 rows
file = 'digits.csv'
data = pd.read_csv(file, nrows=5, header=None) 
data_array = data.values# Build a numpy array
print(type(data_array))

#import and histogram
import matplotlib.pyplot as plt
file = 'titanic_corrupt.txt'
data = pd.read_csv(file, sep='\t', comment='#', na_values=['Nothing'])
print(data.head())
pd.DataFrame.hist(data[['Age']])
plt.xlabel('Age (years)')
plt.ylabel('count')
plt.show()
################################################################################################
#Other file types
    #Excel has many spreadsheets,not a single take, it is not a flat
    #Excel, Matlab SAS, Stata, HDF5
################################################################################################
#get the list of files
import os
wd = os.getcwd()
os.listdir(wd) #get the list of files

#Loading a pickled file
import pickle
with open('data.pkl', 'rb') as file:
    d = pickle.load(file) #read and binary ='rb
print(d)
print(type(d))

#Excel file open
import pandas as pd
file = 'battledeath.xlsx'# Assign spreadsheet filename: file
xls = pd.ExcelFile(file)
print(xls.sheet_names)# Print sheet names

#Select Spreadsheet
#1)
df1 = xls.parse('2004')
print(df1.head())
#2) using index
df2 = xls.parse(0)
print(df2.head())

# First spreadsheet parse, Rename the columns
df1 = xls.parse(0, skiprows=[0], names=['Country' , 'AAM due to War (2002)'])
print(df1.head())

# Second spreadsheet column parse, Rename the column
df2 = xls.parse(1, usecols=[0], skiprows=[0], names=['Country'])
print(df2.head())
################################################
#Importing SAS/Stata
    #SAS: from sas7bdat import SAS7BDAT
    #Stata: df = pd.read_stata('disarea.dta')
################################################
#SAS 
from sas7bdat import SAS7BDAT
with SAS7BDAT('sales.sas7bdat') as file:# Save file to a DataFrame
    df_sas = file.to_data_frame()
print(df_sas.head())
pd.DataFrame.hist(df_sas[['P']])# Plot histogram
plt.ylabel('count')
plt.show()

#Stata
import pandas as pd
df = pd.read_stata('disarea.dta')
print(df.head())
pd.DataFrame.hist(df[['disa10']])# Plot histogram
plt.xlabel('Extent of disease')
plt.ylabel('Number of countries')
plt.show()
################################################
#Importing HDF5
    #h5py_data = h5py.File(h5py_file, 'r')
    #Stata: df = pd.read_stata('disarea.dta')
################################################
import numpy as np
import h5py
file = 'LIGO_data.hdf5'
data = h5py.File(file, 'r')
print(type(data))# datatype 
# Print the keys of the file
for key in data.keys():
    print(key)


group =data['strain']
for key in group.keys():# Check out keys of group
    print(key)
strain = data['strain']['Strain'].value # time series data: strain
num_samples= 10000
time = np.arange(0, 1, 1/num_samples)
# Plot data
plt.plot(time, strain[:num_samples])
plt.xlabel('GPS Time (s)')
plt.ylabel('strain')
plt.show()
################################################
#Importing MATLAB files
################################################
import scipy.io
mat = scipy.io.loadmat('albeck_gene_expression.mat')
print(type(mat))

# Print the keys of the MATLAB dictionary
print(mat.keys())
# Print the type of the value corresponding to the key 'CYratioCyt'
print(type(mat['CYratioCyt']))
# Print the shape of the value corresponding to the key 'CYratioCyt'
print(mat['CYratioCyt'].shape)

# Subset the array and plot it
data = mat['CYratioCyt'][25, 5:]
fig = plt.figure()
plt.plot(data)
plt.xlabel('time (min.)')
plt.ylabel('normalized fluorescence (measure of expression)')
plt.show()
################################################################################################
# Introduction to relational databases
    #Relational Database Management Systems
'''     ● PostgreSQL
        ● MySQL
        ● SQLite
        ● SQL = Structured Query Language'''
################################################################################################
from sqlalchemy import create_engine
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Save the table names to a list: table_names
table_names = engine.table_names()
print(table_names)


## SQL Queries!
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Chinook.sqlite')
con = engine.connect()# Open engine connection
rs = con.execute("SELECT * FROM Album")# Perform query
df = pd.DataFrame(rs.fetchall())# save to DataFrame
con.close()# Close connection
print(df.head())# Print head of DataFrame df

## Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("SELECT LastName, Title FROM Employee")
    df = pd.DataFrame(rs.fetchmany(3)) #only 3 first results
    df.columns = rs.keys() # Set the DataFrame's column names
# Print the length of the DataFrame df
print(len(df))
# Print the head of the DataFrame df
print(df.head())


## Filtering your database -- WHERE
engine = create_engine('sqlite:///Chinook.sqlite') #Create engine
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("select * from Employee where EmployeeId >= 6")
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()
print(df.head())

# ORDER BY
engine = create_engine('sqlite:///Chinook.sqlite')
with engine.connect() as con:
    rs = con.execute("select * from Employee order by BirthDate asc")
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()
print(df.head())

## Pandas to SQL Query 
#Method1
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Chinook.sqlite')
df = pd.read_sql_query("select * from Album", engine)
print(df.head())
#Method2
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Album")
    df1 = pd.DataFrame(rs.fetchall())
    df1.columns = rs.keys()
print(df.equals(df1))

## Pandas for more complex querying
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Chinook.sqlite')
df = pd.read_sql_query("select * from Employee where EmployeeId >=6 order by BirthDate ", engine)
print(df.head())

################################################
#table relationships
    #Inner Join
################################################
##INNER JOIN
import pandas as pd
from sqlalchemy import create_engine
engine = create_engine('sqlite:///Chinook.sqlite')
with engine.connect() as con:
    rs = con.execute("SELECT Title, Name FROM Album INNER JOIN Artist on Album.ArtistID = Artist.ArtistID")
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()
print(df.head())

##Filtering your INNER JOIN
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Chinook.sqlite')
df = pd.read_sql_query("select * from PlaylistTrack INNER JOIN Track on PlaylistTrack.TrackId = Track.TrackId where Milliseconds < 250000",engine)
# Print head of DataFrame df
print(df.head())
# Print head of DataFrame
print(df.head())
################################################################################################
###########################Intermediate Importing Data in Python############################# 
#Importing flat files from the web
    #flat files: csv,txt
    #pickled filed: excel, many others
    #web data
     '''    1)Import and locally save datasets from the web
            2)Load datasets into pandas DataFrames
            3)Make HTTP requests (GET requests)
            4)Scrape web data such as HTML
            5)Parse HTML into useful data (BeautifulSoup)
            6)Use the urllib and requests packages'''
################################################################################################
##Importing flat files from the web
from urllib.request import urlretrieve
import pandas as pd
# url
url = 'https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'
# Save file locally
urlretrieve(url,'winequality-red.csv' )
# Read file into a DataFrame and print its head
df = pd.read_csv('winequality-red.csv', sep=';')
print(df.head())

##Opening and reading flat files(CSV) from the web
import matplotlib.pyplot as plt
import pandas as pd
url = 'https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'#url
# Read file into a DataFrame
df = pd.read_csv(url, sep= ';')
print(df.head())
# Plot first column of df
pd.DataFrame.hist(df.ix[:, 0:1])
plt.xlabel('fixed acidity (g(tartaric acid)/dm$^3$)')
plt.ylabel('count')
plt.show()

##Importing non-flat files(EXCEL) from the web
import pandas as pd
url = 'http://s3.amazonaws.com/assets.datacamp.com/course/importing_data_into_r/latitude.xls'
xls = pd.read_excel(url,sheet_name=None) 
# Print the sheetnames to the shell
print(xls.keys())
# Print the head of the first sheet 
print(xls['1700'].head())

################################################
#HTTP requests to import files from the web
    '''
URL
    Uniform/Universal Resource Locator
    References to web resources
    Focus: web addresses
    Ingredients:
    Protocol identier - http:
    Resource name - datacamp.com
    These specify web addresses uniquely

HTTP
    HyperText Transfer Protocol
    Foundation of data communication for the web
    HTTPS - more secure form of HTTP
    Going to a website = sending HTTP request
    GET request
    urlretrieve() performs a GET request
    HTML - HyperText Markup Language
    '''
#urllib package
    #Provides interface for fetching data across the web
    #urlopen() - accepts URLs instead of le names
#request package
################################################
##urllib - perform HTTP request 
from urllib.request import urlopen, Request  
url = "http://www.datacamp.com/teach/documentation"
# This packages the request: request
request = Request(url)
# Sends the request and catches the response: response
response = urlopen(request)
print(type(response)) # datatype
response.close()


##urllib -  Printing HTTP request results
from urllib.request import urlopen, Request
url = "http://www.datacamp.com/teach/documentation"
request = Request(url)# This packages the request
response = urlopen(request)
# Extract the response
html = response.read() 
print(html)
response.close()

##requests - perform HTTP request 
import requests
url = "http://www.datacamp.com/teach/documentation"
# Packages the request, send the request and catch the response: r
r = requests.get(url)
# Extract the response: text
text = r.text
print(text)
################################################
#Scraping the web in Python
'''
HTML
    Mix of unstructured and structured data
    Structured data:
    Has pre-defined data model, or
    Organized in a defined manner
    Unstructured data: neither of these properties
BeautifulSoup
    Parse and extract structured data from HTML
    Make tag soup beautiful and extract information
'''
################################################
##Parsing HTML with BeautifulSoup
import requests
from bs4 import BeautifulSoup
url = 'https://www.python.org/~guido/'
# Package the request, send the request and catch the response: 
r=requests.get(url)
# Extracts the response as html: 
html_doc = r.text
# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)
# Prettify the BeautifulSoup object: 
pretty_soup = soup.prettify()
print(pretty_soup)

##Turning a webpage into data using BeautifulSoup: getting the text
import requests
from bs4 import BeautifulSoup
url = 'https://www.python.org/~guido/'
r = requests.get(url)
html_doc = r.text
soup = BeautifulSoup(html_doc)
# Get the title of Guido's webpage: 
guido_title = soup.title
print(guido_title)
# Get Guido's text: 
guido_text = soup.get_text()
print(guido_text)


##BeautifulSoup: getting the hyperlinks
import requests
from bs4 import BeautifulSoup
url = 'https://www.python.org/~guido/'
r = requests.get(url)
html_doc = r.text
soup = BeautifulSoup(html_doc)
print(soup.title)
# Find all 'a' tags (which define hyperlinks): 
a_tags = soup.find_all("a")
# Print the URLs to the shell
for link in a_tags:
    print(link.get('href'))

################################################################################################
#Introduction to APIs and JSONs
    #APIs(Application Programming Interfaces) : a set of protocols that building and interacting with software application
        # code that allows two software programs to communicate to each other
        # Ex) extract twitter data, using Twitter API, connecting Twitter and python 
        # Ex) Wikipedia API: allow automated pulling processing information to python
    #JSONs(JavaScript Object Notations): standard formation of transferring API data
        #real-time server to browser communication
        #human readable
        #datatype = dictionary = {'key':'value'}
################################################################################################
# Import json file
with open("a_movie.json") as json_file:
    json_data = json.load(json_file)

# Print all key-value pair in json_data
for key, value in json_data.items():
    print(key + ':', value)

# Print selective Keys Only
for key, value in json_data.items():
    if key in ('Title','Year'):
        print(key + ':', value)
################################################
#APIs and interacting with the world wide web
################################################
## Connecting to an API in Python
import requests
url = 'http://www.omdbapi.com/?apikey=72bc447a&t=the+social+network'
# send the request and catch the response: r
r = requests.get(url)
# Print the text of the response
print(r.text)

## JSON–from the web to Python
import requests
url = 'http://www.omdbapi.com/?apikey=72bc447a&t=social+network'
r = requests.get(url)
# Decode the JSON data into a dictionary: json_data
json_data = r.json()
# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])

## Checking out the Wikipedia API
import requests
url = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles=pizza"
r = requests.get(url)
json_data = r.json()

# Print the Wikipedia page extract
pizza_extract = json_data['query']['pages']['24768']['extract']
print(pizza_extract)
################################################
# The Twitter API and Authentication
    '''1)https://developer.twitter.com/en/docs
        2) Apps (right top corner)
        3) access 
    '''
################################################
#%%
import tweepy, json



# %%
