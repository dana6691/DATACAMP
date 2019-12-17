################################################
#
################################################
#read test data
import pandas as pd
test = pd.read_csv('test.csv')
print(train.columns.tolist())
print(test.columns.tolist())
################################################
#visualization
################################################
#histogram
train.hist(bins=30,alpha=0.5)
plt.show()
################################################
#Submission
################################################
#train simple model
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv('train.csv')# Read the train data
rf = RandomForestRegressor()# Create a Random Forest object
rf.fit(X=train[['store', 'item']], y=train['sales'])# Train a model

#Prepare a submission
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')
print(sample_submission.head())

test['sales'] = rf.predict(test[['store', 'item']])# Get predictions for the test set
test[['id', 'sales']].to_csv('kaggle_submission.csv', index=False)# Write test predictions using the sample_submission format
################################################
#Competition metric
    '''1)AUC(area under the ROC): classification
        2)F1 Score(F1) : classification
        3)Mean Log Loss(LogLoss) : classification
        4)Mean Absolute Error(MAE): Regression
        5)Mean Squared Error(MSE): Regression
        6)Mean Average Precision at K(MAPK,MAP@K): Ranking'''
    #Submission = private + public data
    #Whem training MSE is way lower than testing MSE == overfitting
################################################
#Store Item Demand Forecasting Challenge. 
import xgboost as xgb

# Create DMatrix on train data
dtrain = xgb.DMatrix(data=train[['store', 'item']],
                     label=train['sales'])

# Define xgboost parameters
params = {'objective': 'reg:linear',
          'max_depth': 15,
          'silent': 1}

# Train xgboost model
xg_depth_15 = xgb.train(params=params, dtrain=dtrain)





from sklearn.metrics import mean_squared_error

dtrain = xgb.DMatrix(data=train[['store', 'item']])
dtest = xgb.DMatrix(data=test[['store', 'item']])

# For each of 3 trained models
for model in [xg_depth_2, xg_depth_8, xg_depth_15]:
    # Make predictions
    train_pred = model.predict(dtrain)     
    test_pred = model.predict(dtest)          
    
    # Calculate metrics
    mse_train = mean_squared_error(train['sales'], train_pred)                  
    mse_test = mean_squared_error(test['sales'], test_pred)
    print('MSE Train: {:.3f}. MSE Test: {:.3f}'.format(mse_train, mse_test))
################################################
#Workflow
    #Understand the problem -> EDA -> Local Validation -> Modeling
        #Data type: tabular, time series, images, text
        #Problem type: classification, regressoin, ranking
        #Evaluation metrics: ROC,AUC, F1 score, MAE, MSE
            #Mean Squared Error (MSE) for the regression problem
            #Logarithmic Loss (LogLoss) for the binary classification problem:
################################################
#MSE
import numpy as np
from sklearn.metrics import mean_squared_error
# Define your own MSE function
def own_mse(y_true, y_pred):
    squares = np.power(y_true - y_pred, 2)
    err = np.mean(squares)
    return err
print('Sklearn MSE: {:.5f}. '.format(mean_squared_error(y_regression_true, y_regression_pred)))
print('Your MSE: {:.5f}. '.format(own_mse(y_regression_true, y_regression_pred)))

#LogLoss
import numpy as np
from sklearn.metrics import log_loss
# Define your own LogLoss function
def own_logloss(y_true, prob_pred):
    terms = y_true * np.log(prob_pred) + (1 - y_true) * np.log(1 - prob_pred)
    err = np.mean(terms) 
    return -err
print('Sklearn LogLoss: {:.5f}'.format(log_loss(y_classification_true, y_classification_pred)))
print('Your LogLoss: {:.5f}'.format(own_logloss(y_classification_true, y_classification_pred)))
################################################
#EDA
    #know size of the data
    #properties of the target variable
    #properties of the features
    #general idea of feature engineering
################################################
print('Train shape:', train.shape) #shape
print('Test shape:', test.shape)
print(train.head())# Train head()
print(train.fare_amount.describe())# Describe the target variable
print(train.passenger_count.value_counts())# Train distribution of passengers within rides

train['distance_km'] = haversine_distance (train)# Calculate the ride distance
plt.scatter(x=train['fare_amount'], y=train['distance_km'], alpha=0.5)# Draw a scatterplot
plt.xlabel('Fare amount')
plt.ylabel('Distance, km')
plt.title('Fare amount based on the distance')
plt.ylim(0, 50)# Limit on the distance
plt.show()

train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)# Create hour feature
train['hour'] = train.pickup_datetime.dt.hour
hour_price = train.groupby('hour', as_index=False)['fare_amount'].median()# Find median fare_amount for each hour
plt.plot(hour_price['hour'], hour_price['fare_amount'], marker='o')# Plot the line plot
plt.xlabel('Hour of the day')
plt.ylabel('Median fare amount')
plt.title('Fare amount based on day time')
plt.xticks(range(24))
plt.show()
################################################
#Local Validation
    #solution of overfitting:
        #Holdout split: split training data(30,70%), use one for training and predicting, and another for model quality check
        #K-fold cross validation: train model k-times, train data except for single fold  
        #Strafied K-fold: 
################################################
#K-fold cross-validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=True, random_state=123)# Create a KFold object
fold = 0
for train_index, test_index in kf.split(train):
    # Obtain training and testing folds
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
    print('Fold: {}'.format(fold))
    print('CV train shape: {}'.format(cv_train.shape))
    print('Medium interest listings in CV train: {}\n'.format(sum(cv_train.interest_level == 'medium')))
    fold += 1


#Stratified K-Fold
from sklearn.model_selection import StratifiedKFold
str_kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)# Create a StratifiedKFold object
fold = 0
for train_index, test_index in str_kf.split(train, train['interest_level']):
    # Obtain training and testing folds
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
    print('Fold: {}'.format(fold))
    print('CV train shape: {}'.format(cv_train.shape))
    print('Medium interest listings in CV train: {}\n'.format(sum(cv_train.interest_level == 'medium')))
    fold += 1   
################################################
#K-fold cross-validation for time data: split by K, then (Train-Test) (Train-Train-Test) (Train-Train-Train-Test) then use for prediction
    #Different validation method:overall validation score(mean of them)
################################################
#Time data demand forecasting ("Store Item Demand Forecasting Challenge")
from sklearn.model_selection import TimeSeriesSplit
time_kfold = TimeSeriesSplit(n_splits=3)# Create TimeSeriesSplit object
train = train.sort_values('date')# Sort train data by date
fold = 0
for train_index, test_index in time_kfold.split(train):
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
    
    print('Fold :', fold)
    print('Train date range: from {} to {}'.format(cv_train.date.min(), cv_train.date.max()))
    print('Test date range: from {} to {}\n'.format(cv_test.date.min(), cv_test.date.max()))
    fold += 1
#Overall validation score
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
train = train.sort_values('date')# Sort train data by date
kf = TimeSeriesSplit(n_splits=3)# Initialize 3-fold time cross-validation
mse_scores = get_fold_mse(train, kf)# Get MSE scores for each cross-validation split
print('Mean validation MSE: {:.5f}'.format(np.mean(mse_scores)))
print('MSE by fold: {}'.format(mse_scores))
print('Overall validation MSE: {:.5f}'.format(np.mean(mse_scores) + np.std(mse_scores)))
    '''
    print('Mean validation MSE: {:.5f}'.format(np.mean(mse_scores)))
    print('MSE by fold: {}'.format(mse_scores))
    print('Overall validation MSE: {:.5f}'.format(np.mean(mse_scores) + np.std(mse_scores)))
    '''
################################################
#Feature engineering
#Modeling = Local Validation = preprocessing data + create new features + improve models + Apply
################################################
#Arithmetical features
print('RMSE before feature engineering:', get_kfold_rmse(train))# Look at the initial RMSE
train['TotalArea'] = train['TotalBsmtSF'] + train['FirstFlrSF'] + train['SecondFlrSF']# Find the total area of the house
print('RMSE with total area:', get_kfold_rmse(train))

#Date features(concatenate and split back)
taxi = pd.concat([train, test])# Concatenate train and test together
taxi['pickup_datetime'] = pd.to_datetime(taxi['pickup_datetime'])# Convert pickup date to datetime object
taxi['dayofweek'] = taxi['pickup_datetime'].dt.dayofweek # Create a day of week feature
taxi['hour'] = taxi['pickup_datetime'].dt.hour# Create an hour feature

new_train = taxi[taxi['id'].isin(train['id'])]# Split back into train and test
new_test = taxi[taxi['id'].isin(test['id'])]
################################################
#Categorical Features
    #categorical variable into number
        #1) Label encoding: simply change string to number, create 1 column
        #2) One-Hot encoding: make dummies, created 4 columns
    #Binary Features: Always Label-encoding
################################################
#Label encoding
houses = pd.concat([train, test])# Concatenate train and test together
from sklearn.preprocessing import LabelEncoder# Label encoder
le = LabelEncoder()
houses['RoofStyle_enc'] = le.fit_transform(houses['RoofStyle'])# Create new features
houses['CentralAir_enc'] = le.fit_transform(houses['CentralAir'])
print(houses[['RoofStyle', 'RoofStyle_enc', 'CentralAir', 'CentralAir_enc']].head())# Look at new features

#One-Hot encoding
houses = pd.concat([train, test])# Concatenate train and test together
print(houses['RoofStyle'].value_counts(), '\n') # check frequency table
print(houses['CentralAir'].value_counts()) #It is binary

#Label encode binary 'CentralAir' feature
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
houses['CentralAir_enc'] = le.fit_transform(houses['CentralAir'])

#One-Hot encoded features
ohe = pd.get_dummies(houses['RoofStyle'], prefix='RoofStyle')
houses = pd.concat([houses, ohe], axis=1)# Concatenate OHE features to houses
print(houses[[col for col in houses.columns if 'RoofStyle' in col]].head(3))# Look at OHE features
################################################
#Target encoding
    #high cardinality categorical features
        #label encoder provides distinct number for each category
        #one-hot encoder creates new features for each category value
        #Mean target encoding
            # 1) calculate mean on train, apply to the test
            # 2) split train into k folds. calculate mean on (k-1)folds, apply to K-th fold
            # 3) Add mean target encoded feature to the model
            # (proportion of number of 1 /(0 + 1) for "A" "B") 
            # out-of-fold method. SPlit train into fold 1 and 2. calculate fold 1 for proporton and fold 2 for proportion
# binary classification usually mean target encoding is used
# regression mean could be changed to median, quartiles, etc.
# multi-class classification with N classes we create N features with target mean for each category in one vs. all fashion
################################################
#Mean target encoding
def test_mean_target_encoding(train, test, target, categorical, alpha=5):
    global_mean= train[target].mean() #calculate global mean

    train_groups = train.groupby(categorical) # Group by the categorical feature and calculate its properties
    category_sum = train_groups[target].sum()
    category_size = train_groups.size()

    train_statistics = (category_sum + global_mean*alpha) / (category_size + alpha)
    test_feature = test[categorical].map(train_statistics).fillna(global_mean)# Apply statistics to the test data and fill new categories
    return test_feature.values

def train_mean_target_encoding(train, target, categorical, alpha=5):
    kf = KFold(n_splits=5, random_state=123, shuffle=True)# Create 5-fold cross-validation
    train_feature = pd.Series(index=train.index)
    
    # For each folds split
    for train_index, test_index in kf.split(train):
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
        cv_test_feature = test_mean_target_encoding(cv_train, cv_test, target, categorical, alpha)# Calculate out-of-fold statistics and apply to cv_test
        train_feature.iloc[test_index] = cv_test_feature      # Save new feature for this particular fold 
    return train_feature.values

def mean_target_encoding(train, test, target, categorical, alpha=5):
    train_feature = train_mean_target_encoding(train, target, categorical, alpha)# Get the train feature
    test_feature = test_mean_target_encoding(train, test, target, categorical, alpha)# Get the test feature
    return train_feature, test_feature# Return new features to add to the model

#K-fold cross-validation
#Binary classification - playground competition
kf = KFold(n_splits=5, random_state=123, shuffle=True)

for train_index, test_index in kf.split(bryant_shots):
    cv_train, cv_test = bryant_shots.iloc[train_index], bryant_shots.iloc[test_index]
    # Create mean target encoded feature
    cv_train['game_id_enc'], cv_test['game_id_enc'] = mean_target_encoding(train=cv_train,
                                                                           test=cv_test,
                                                                           target='shot_made_flag',
                                                                           categorical='game_id',
                                                                           alpha=5)
    print(cv_train[['game_id', 'shot_made_flag', 'game_id_enc']].sample(n=1))# Look at the encoding

# Create mean target encoded feature
train['RoofStyle_enc'], test['RoofStyle_enc'] = mean_target_encoding(train=train,
                                                                     test=test,
                                                                     target='SalePrice',
                                                                     categorical='RoofStyle',
                                                                     alpha=10)
print(test[['RoofStyle', 'RoofStyle_enc']].drop_duplicates())  
'''
    Hip roof are the most pricy, while houses with the Gambrel roof are the cheapest.
'''
################################################
#Missing data
    #Numerical: 1)mean/median imputation 2) constant value imputation
    #Categorical: 1) most frequenct category imputation 2) New category imputation
################################################
# Find the number of missing values in each column
twosigma = pd.read_csv("twosigma_train.csv")
print(twosigma.isnull().sum())
print(twosigma[['building_id', 'price']].head()) #columns with the missing values

# Create mean imputer
from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(strategy='mean')
rental_listings[['price']] = mean_imputer.fit_transform(rental_listings[['price']])

# Create constant imputer
from sklearn.impute import SimpleImputer
constant_imputer = SimpleImputer(strategy="constant", fill_value="MISSING")
rental_listings[['building_id']] = constant_imputer.fit_transform(rental_listings[['building_id']])# building_id imputation
################################################################################################
#Baseline Model
    #Local/public validation: If local RSME decreases, public should decreases, then it is reliale. 

################################################
#Replicate validation score
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# Calculate the mean fare_amount on the validation_train data
naive_prediction = np.mean(validation_train['fare_amount'])

# Assign naive prediction to all the holdout observations
validation_test['pred'] = naive_prediction

# Measure the local RMSE
rmse = sqrt(mean_squared_error(validation_test['fare_amount'], validation_test['pred']))
print('Validation RMSE for Baseline I model: {:.3f}'.format(rmse))


#Baseline based on the date
# Get pickup hour from the pickup_datetime column
train['hour'] = train['pickup_datetime'].dt.hour
test['hour'] = test['pickup_datetime'].dt.hour


hour_groups = train.groupby('hour')['fare_amount'].mean()# Calculate average fare_amount grouped by pickup hour 
test['fare_amount'] = test.hour.map(hour_groups)# Make predictions on the test set

# Write predictions
test[['id','fare_amount']].to_csv('hour_mean_sub.csv', index=False)
'''
    baseline achieves 1409th place on the Public Leaderboard which is slightly better than grouping by the number of passengers. 
'''


#Baseline based on the gradient boosting
from sklearn.ensemble import RandomForestRegressor
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
            'dropoff_latitude', 'passenger_count', 'hour']# Select only numeric features
rf = RandomForestRegressor()# Train a Random Forest model
rf.fit(train[features], train.fare_amount)
test['fare_amount'] = rf.predict(test[features])# Make predictions on the test data
test[['id','fare_amount']].to_csv('rf_sub.csv', index=False)# Write predictions
'''
    final baseline achieves the 1051st place on the Public Leaderboard which is slightly better than the Gradient Boosting from the video. So, now you know how to build fast and simple baseline models to validate your initial pipeline.
'''
################################################
#Hyperparameter tuning
################################################

