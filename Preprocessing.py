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
################################################################################################
#Standardized data
	#use for linear space, linearity assumption
	#features with high variance
	#continuous variable
	#1)log normalization
################################################################################################
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

################################################################################################
#Feature engineering
	#categorical variable: encoding
################################################################################################
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

################################################################################################
#Feature Selection
		#Removing redundant features
 			#remove noisy features
			#correlated features
			#duplicated features
################################################################################################
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









##########################################
#Bias-variance tradeoff
    #High variance
        #Low training error but high testing error
        #when models are overt and have high complexity
    #High Bias: failing to find the relationship between the data and the response
        #High training/testing error
        #when models are underfit
##########################################
#Error due to under/over-fitting
# Update the rfr model
rfr = RandomForestRegressor(n_estimators=25,
                            random_state=1111,
                            max_features=2)
rfr = RandomForestRegressor(n_estimators=25,
                            random_state=1111,
                            max_features=11)
rfr = RandomForestRegressor(n_estimators=25,
                            random_state=1111,
                            max_features=4)

rfr.fit(X_train, y_train)
print('The training error is {0:.2f}'.format(mae(y_train, rfr.predict(X_train))))
print('The testing error is {0:.2f}'.format(mae(y_test, rfr.predict(X_test))))
'''
The training error is 3.88
The testing error is 9.15
The training error is 3.57
The testing error is 10.05
The training error is 3.6
The testing error is 8.79
'''
from sklearn.metrics import accuracy_score

test_scores, train_scores = [], []
for i in [1, 2, 3, 4, 5, 10, 20, 50]:
    rfc = RandomForestClassifier(n_estimators=i, random_state=1111)
    rfc.fit(X_train, y_train)
    # Create predictions for the X_train and X_test datasets.
    train_predictions = rfc.predict(X_train)
    test_predictions = rfc.predict(X_test)
    # Append the accuracy score for the test and train predictions.
    train_scores.append(round(accuracy_score(y_train, train_predictions), 2))
    test_scores.append(round(accuracy_score(y_test, test_predictions), 2))
# Print the train and test scores.
print("The training scores were: {}".format(train_scores))
print("The testing scores were: {}".format(test_scores))
'''
The training scores were: [0.94,0.93,0.98,0.97,0.99,1,1,1]
The testing scores were: [0.83,0.79,0.89,0.91,0.91,0.93,0.97,0.98]
Every time you use more trees, you achieve higher accuracy. 
'''
####################################################################################
#Cross Validation
####################################################################################
# Create two different samples of 200 observations 
sample1 = tic_tac_toe.sample(200, random_state=1111)
sample2 = tic_tac_toe.sample(200, random_state=1171)

# Print the number of common observations 
print(len([index for index in sample1.index if index in sample2.index]))

# Print the number of observations in the Class column for both samples 
print(sample1['Class'].value_counts())
print(sample2['Class'].value_counts())
''' positive/negative:134/66, positive/negative:123/77'''
##########################################
## 1)scikit-learn's KFold()
##########################################
from sklearn.model_selection import KFold
# Use KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1111)
splits = kf.split(X) # Create splits
# Print the number of indices
for train_index, val_index in splits:
    print("Number of training indices: %s" % len(train_index))
    print("Number of validation indices: %s" % len(val_index))
'''This dataset has 85 rows. 
    You have created five splits - each containing 68 training and 17 validation indices.'''

#check the accuracy of this model Using KFold indices
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
rfc = RandomForestRegressor(n_estimators=25, random_state=1111)
# Access the training and validation indices of splits
for train_index, val_index in splits:
    # Setup the training and validation data
    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index]
    rfc.fit(X_train, y_train)
    # Make predictions, and print the accuracy
    predictions = rfc.predict(X_val)
    print("Split accuracy: " + str(mean_squared_error(y_val, predictions)))
##########################################
## 2) scikit-learn's methods
##########################################
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer

rfc = RandomForestRegressor(n_estimators=25, random_state=1111)
mse = make_scorer(mean_squared_error)
cv = cross_val_score(estimator=rfc,
                     X=X_train,
                     y=y_train,
                     cv=10,
                     scoring=mse)
print(cv.mean()) # Print the mean of error = 155.56
##########################################
## 3)Leave-one-out-cross-validation(LOOCV): 
                                #use when training data is limited(small)
                                #when you want absolute best error estimate
                                #compuational expensive, lots of parameter or lots of data is not appropriate
##########################################
from sklearn.metrics import mean_absolute_error, make_scorer
mae_scorer = make_scorer(mean_absolute_error)# Create scorer

rfr = RandomForestRegressor(n_estimators=15, random_state=1111)

# Implement LOOCV
scores = cross_val_score(rfr, X=X, y=y, cv=85, scoring=mae_scorer)

# Print the mean and standard deviation
print("The mean of the errors is: %s." % np.mean(scores))
print("The standard deviation of the errors is: %s." % np.std(scores))
'''5-fold cross-validation will train on only 80% of the data at a time.
    candy dataset only has 85 rows though, use cv=85 in LOOCV
'''
####################################################################################
#Hyperparameter tuning
    #model parameters: results occur from training model
    #model hyperparameter: manually set before the training occurs
        #Random forrest
            #n_estimators: number of decision tress in the forest
            #max_depth: max depth of the decision tress
            #max_features: number of features to consider when making a split
            #min_samples_split: min number of samples required to make a split
####################################################################################
# Review the parameters of rfr
print(rfr.get_params())
# Maximum Depth
max_depth = [4, 8, 12]
# Minimum samples for a split
min_samples_split = [2, 5, 10]
# Max features 
max_features = [4, 6, 8, 10]

## Running a model using ranges
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(
    n_estimators=100,
    max_depth=random.choice(max_depth),
    min_samples_split=random.choice(min_samples_split),
    max_features=random.choice(max_features))
print(rfr.get_params())# Print out the parameters
##########################################
#RandomizedSearchCV
##########################################
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
param_dist = {"max_depth": [2, 4, 6,  8],
              "max_features": [2, 4, 6, 8, 10],
              "min_samples_split": [2, 4, 8, 16]}
rfr = RandomForestRegressor(n_estimators=10, random_state=1111)# Create a random forest regression model
scorer = make_scorer(mean_squared_error)# Create a scorer to use (use the mean squared error)

#Implementing RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
random_search =\
    RandomizedSearchCV(
        estimator=rfr,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring=scorer)
##########################################
#Selecting your final model
##########################################
#find the best parameter sets
rs.best_estimator_  

##Selecting the best precision model
from sklearn.metrics import precision_score, make_scorer
precision = make_scorer(precision_score)
rs = RandomizedSearchCV(
  estimator=rfc, param_distributions=param_dist,
  scoring = precision,
  cv=5, n_iter=10, random_state=1111)
rs.fit(X, y)
# print the mean test scores:
print('The accuracy for each run was: {}.'.format(rs.cv_results_['mean_test_score']))
# print the best model score:
print('The best accuracy for a single model was: {}'.format(rs.best_score_))
