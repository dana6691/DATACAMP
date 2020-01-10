################################################
#Model validation
    '''1)ensuring your model performs as expected on new data
       2)testing model performance on holdout datasets
       3)selecting best model, parameters, accuracy metrics
       4)achieving the best accuracy'''
################################################
## Seen vs. unseen data
# The model is fit using X_train and y_train
model.fit(X_train, y_train)

# Create vectors of predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Train/Test Errors
train_error = mae(y_true=y_train, y_pred=train_predictions)
test_error = mae(y_true=y_test, y_pred=test_predictions)

# Print the accuracy for seen and unseen data
print("Model error on seen data: {0:.2f}.".format(train_error))
print("Model error on unseen data: {0:.2f}.".format(test_error))
'''Model error on seen data: 3.28.
    Model error on unseen data: 11.07'''
################################################
#random forest parameters
    #n_estimators: number of trees in the forest
    #max_depth: max depth of the trees
    #random_state: random seed
################################################
from sklearn.ensemble import RandomForestRegression
from sklearn.ensemble import RandomForestClassifier
rfr = RandomForestRegression(random_state=1111)
rfc=RandomForestClassifier(random_state=1111)
rfr.get_params()#find all parameters
rfc.get_params()

#set parameters and fit a model
rfr.n_estimators = 100 #number of trees
rfr.max_depth = 6 #maximum depth
rfr.random_state = 1111 #random state
rfr.fit(X_train, y_train)# Fit the model

#feature importance
for i, item in enumerate(rfr.feature_importances_):
      # Use i and item to print out the feature importance of each column
    print("{0:s}: {1:.2f}".format(X_train.columns[i], item))
'''chocolate is the most important variable.'''
################################################
#Classification Model
################################################
#Classification Prediction
rfc.fit(X_train, y_train)# Fit the rfc model. 
classification_predictions = rfc.predict(X_test)# Create arrays of predictions
probability_predictions = rfc.predict_proba(X_test)

# count of binary predictions
print(pd.Series(classification_predictions).value_counts())

# first value from probability_predictions
print('The first predicted probabilities are: {}'.format(probability_predictions[0]))
'''
1    563
0    204
The first predicted probabilities are: [0.26524423 0.73475577]
'''

#model parameters
rfc = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1111)
print(rfc)# Print the classification model
print('The random state is: {}'.format(rfc.random_state))# model's random state parameter
print('Printing the parameters dictionary: {}'.format(rfc.get_params()))# Print all parameters

#model, fit, prediction, evaluate overall accuracy
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1111)
rfc.fit(X_train, y_train)# Fit
predictions = rfc.predict(X_test) #predictions
print(predictions[0:5])
print(rfc.score(X_test, y_test)) #accuracy
'''
the first five predictions were all 1, indicating that Player One is predicted to win all five of those games. 
You also see the model accuracy was only 82%.
'''

################################################

################################################
# Create dummy variables using pandas
X = pd.get_dummies(tic_tac_toe.iloc[:,0:9])
y = tic_tac_toe.iloc[:, 9]

## Create one holdout sets
# Create training and testing datasets. Use 10% for the test set
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.1, random_state=1111)


## Create two holdout sets
#1) Create temporary training and final testing datasets
X_temp, X_test, y_temp, y_test  =\
    train_test_split(X, y, test_size=0.20, random_state=1111)

#2) Create the final training and validation datasets
X_train, X_val, y_train, y_val  =\
    train_test_split(X_temp, y_temp, test_size=0.25, random_state=1111)
'''
Anytime we are evaluating model performance repeatedly we need to create training, validation, and testing datasets.
'''
################################################
#Regression Model Accuracy Metrics
    #MAE(Mean Absolute Error): sum(|y-y^|)/n
        #Simplest and most intuitive metric
        #Treats all points equally
        #Not sensitive to outliers
    #MSE(Mean squared error: sum(y-y^)^2/n
        #Most widely used regression metric
        #Allows outlier errors to contribute more to the overall error
        #Random family road trips could lead to large errors in predictions
################################################
#Mean absolute error
from sklearn.metrics import mean_absolute_error

# Manually calculate the MAE
n = len(predictions)
mae_one = sum(np.absolute(y_test - predictions)) / n
print('With a manual calculation, the error is {}'.format(mae_one))

# Use scikit-learn to calculate the MAE
mae_two = mean_absolute_error(y_test, predictions)
print('Using scikit-lean, the error is {}'.format(mae_two))
'''
    With a manual calculation, the error is 5.9
    Using scikit-lean, the error is 5.9
These predictions were about six wins off on average. This isn't too bad considering NBA teams play 82 games a year. Let's see how these errors would look if you used the mean squared error instead.
'''

#Mean squared error
from sklearn.metrics import mean_squared_error

n = len(predictions)
# Finish the manual calculation of the MSE
mse_one = sum((y_test - predictions)**2) / n
print('With a manual calculation, the error is {}'.format(mse_one))

# Use the scikit-learn function to calculate MSE
mse_two = mean_squared_error(y_test,predictions)
print('Using scikit-lean, the error is {}'.format(mse_two))
'''
    With a manual calculation, the error is 49.1
    Using scikit-lean, the error is 49.1
MSE of 49.1, which is the average squared error of using your model. Although the MSE is not as interpretable as the MAE, it will help us select a model that has fewer 'large' errors.
'''

# Find the East conference teams
east_teams = labels == "E"

# Create arrays for the true and predicted values
true_east = y_test[east_teams]
preds_east = predictions[east_teams]

# Print the accuracy metrics
print('The MAE for East teams is {}'.format(
    mae(true_east, preds_east)))

# Print the West accuracy
print('The MAE for West conference is {}'.format(west_error))
################################################
#Classficiation
    #Precision
    #Recall (also called sensitivity)
    #Accuracy
    #Specicity
    #F1-Score, and its variations
################################################
#Accuracy is true positives (predicted 1, actual 1) + true negatives (predicted 0, actual 0) divided by the total.
#Precision is true positives divided by the total number of predicted positive values.
#Recall is true positives divided by the total number of actual positive values.

from sklearn.metrics import confusion_matrix

# Create predictions
test_predictions = rfc.predict(X_test)

# Create and print the confusion matrix
cm = confusion_matrix(y_test, test_predictions)
print(cm)

# Print the true positives (actual 1s that were predicted 1s)
print("The number of true positives is: {}".format(cm[1, 1]))


## precision_score
from sklearn.metrics import  precision_score
test_predictions = rfc.predict(X_test)
# Create precision or recall score based on the metric you imported
score = precision_score(y_test, test_predictions)
print("The precision value is {0:.2f}".format(score))
'''
With a precision of only 79%, you may need to try some other modeling techniques to improve this score.
'''
