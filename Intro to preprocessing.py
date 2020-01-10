################################################################################################
#Preprocessing data
################################################
Volunteer.describe()
Volunteer.shape
Print(volunteer.dtypes)
Print(volunteer.dropna())
Volunteer.isnull().sum()

# Check how many values are missing in the category_desc column
print(volunteer['category_desc'].isnull().sum())

# Subset the volunteer dataset
volunteer_subset = volunteer[volunteer['category_desc'].notnull()]

# Print out the shape of the subset
print(volunteer_subset.shape)

volunteer.dtypes

## Converting a column type
# Print the head of the hits column
print(volunteer["hits"].head())

# Convert the hits column to type int
volunteer["hits"] = volunteer["hits"].astype("int")

# Look at the dtypes of the dataset
print(volunteer.dtypes)


#check variable counts
volunteer.category_desc.value_counts()
################################################
#Stratified sampling
################################################
# Create a data with all columns except category_desc
volunteer_X = volunteer.drop("category_desc", axis=1)

# Create a category_desc labels dataset
volunteer_y = volunteer[["category_desc"]]

# Use stratified sampling to split up the dataset according to the volunteer_y dataset
X_train, X_test, y_train, y_test = train_test_split(volunteer_X, volunteer_y, stratify=volunteer_y)

# Print out the category_desc counts on the training y labels
print(y_train["category_desc"].value_counts())
################################################
#Standardized data
	#use for linear space, linearity assumption
	#features with high variance
	#continuous variable
	#1)log normalization
################################################
## Log-normalization
#check variance 
Print(wine.var())
# Print out the variance of the Proline column
print(wine["Proline"].var())

# Apply the log normalization function to the Proline column
wine['Proline_log'] = np.log(wine["Proline"])

# Check the variance of the normalized Proline column
print(wine['Proline_log'].var())

################################################
# Scaling data
	#model with linear characteristics
	#center features around 0 and transform to unit variance
################################################
# Import StandardScaler from scikit-learn
from sklearn.preprocessing import StandardScaler

# Create the scaler
ss = StandardScaler()

# Take a subset of the DataFrame you want to scale 
wine_subset = wine[['Ash','Alcalinity of ash','Magnesium']]

# Apply the scaler to the DataFrame subset
wine_subset_scaled = ss.fit_transform(wine_subset)
################################################
#Data modeling
################################################
# Split the dataset and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Fit the k-nearest neighbors model to the training data
knn.fit(X_train,y_train)

# Score the model on the test data
print(knn.score(X_test,y_test))
‘’’accuracy is very low 0,57’’’


## Knn on scaled data
# Create the scaling method.
ss = StandardScaler()

# Apply the scaling method to the dataset used for modeling.
X_scaled = ss.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# Fit the k-nearest neighbors model to the training data.
knn.fit(X_train,y_train)

# Score the model on the test data
print(knn.score(X_test,y_test))

################################################
#Feature engineering
	#categorical variable: encoding
################################################
## Encoding categorical variables - binary
enc = LabelEncoder() # Set up the LabelEncoder object
hiking["Accessible_enc"] = enc.fit_transform(hiking["Accessible"]) # Apply the encoding to the "Accessible" column
print(hiking[['Accessible_enc', "Accessible"]].head()) # Compare the two columns

## Encoding categorical variables - one-hot
category_enc = pd.get_dummies(volunteer["category_desc"]) # Transform the category_desc column
print(category_enc.head()) # Take a look at the encoded columns
################################################
#Feature engineering
	#Numerical variable
################################################
run_columns = ['run1', 'run2', 'run3', 'run4', 'run5'] # Create a list of the columns to average
running_times_5k["mean"] = running_times_5k.apply(lambda row: row[run_columns].mean(), axis=1) # Use apply to create a mean column
print(running_times_5k) # Take a look at the results

## Engineering numerical features – datetime
# First, convert string column to date column
volunteer["start_date_converted"] = pd.to_datetime(volunteer["start_date_date"])

# Extract just the month from the converted column
volunteer["start_date_month"] = volunteer["start_date_converted"].apply(lambda row: row.month)

# Take a look at the converted and new month columns
print(volunteer[['start_date_month', 'start_date_converted']].head())
################################################
#Feature engineering
	#Text variable
################################################
## Engineering features from strings - extraction
# Write a pattern to extract numbers and decimals
def return_mileage(length):
    pattern = re.compile(r"\d+\.\d+")
    
    # Search the text for matches
    mile = re.match(pattern, length)
    
    # If a value is returned, use group(0) to return the found value
    if mile is not None:
        return float(mile.group(0)) 
# Apply the function to the Length column and take a look at both columns
hiking["Length_num"] = hiking["Length"].apply(lambda row: return_mileage(row))
print(hiking[["Length", "Length_num"]].head())

##Engineering features from strings - tf/idf
title_text = volunteer["title"] # Take the title text
tfidf_vec = TfidfVectorizer() # Create the vectorizer method
text_tfidf = tfidf_vec.fit_transform(title_text) # Transform the text into tf-idf vectors

##Text classification using tf/idf vectors
y = volunteer["category_desc"]
X_train, X_test, y_train, y_test = train_test_split(text_tfidf.toarray(), y, stratify=y)
nb.fit(X_train, y_train) #fit
print(nb.score(X_test, y_test)) # Print out the model's accuracy

################################################
#Feature Selection
		#Removing redundant features
 			#remove noisy features
			#correlated features
			#duplicated features
################################################
# Create a list of redundant column names to drop
to_drop = ["category_desc", "created_date", "locality", "region", "vol_requests"]

# Drop those columns from the dataset
volunteer_subset = volunteer.drop(to_drop, axis=1)

# Print out the head of the new dataset
print(volunteer_subset.head())

##Checking for correlated features
# Print out the column correlations of the wine dataset
print(wine.corr())

# Take a minute to find the column where the correlation value is greater than 0.75 at least twice
to_drop = "Flavanoids"

# Drop that column from the DataFrame
wine = wine.drop(to_drop, axis=1)
