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
        #Holdout set: split training data, use one for training and predicting, and another for model quality check
        #K-fold cross validation: train model k-times, train data except for single fold  
        #Strafied K-fold: 
################################################
#K-fold cross-validation
from sklearn.model_selection import KFold

# Create a KFold object
kf = KFold(n_splits=3, shuffle=True, random_state=123)

# Loop through each split
fold = 0
for train_index, test_index in kf.split(train):
    # Obtain training and testing folds
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
    print('Fold: {}'.format(fold))
    print('CV train shape: {}'.format(cv_train.shape))
    print('Medium interest listings in CV train: {}\n'.format(sum(cv_train.interest_level == 'medium')))
    fold += 1


    