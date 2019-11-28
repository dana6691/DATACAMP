################################################
#Supervised machine learning
#Generalization Error = bias^2 + variance + error
    #bias(accurate=low bias): how much f and f^ different
    #variance(precise=low variance): how much f^ is inconsistent over different training sets
    #model complexity: increase --> variance increase, bias decreases
        #goal: find the lowest generalization error
################################################
################################################
#Diagnosis bias and variance = estimating Generalization Error
    '''1)split data into training and test
        2) fit f^ to training set
        3) evaluate error of f^ on test set
        4) generalization error of f^ = test set error of f^'''
#Better model evaluation with Cross-Validation
    #K-fold CV: CV error = (E1+E2+E3..E10)/10
    #Hold-out CV
#If f^ high variance: CV error of f^ > training error of f^
    #Remedy overfitting
        '''1)decrease model complexity
           2)gather mode data'''
#if f^ high bias: CV error of f^ = training error of f^ >> desirede error
    #Remedy underfitting
        '''1)increase model complexity
           2)gather more relevant features'''

################################################
# Import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split

# Set SEED for reproducibility
SEED = 1

# Split the data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Instantiate a DecisionTreeRegressor dt
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26, random_state=SEED)


################################################
# Compute the array containing the 10-folds CV MSEs
MSE_CV_scores = - cross_val_score(dt, X_train, y_train, cv=10, 
                                  scoring='neg_mean_squared_error', 
                                  n_jobs=-1) 

# Compute the 10-folds CV RMSE
RMSE_CV = (MSE_CV_scores.mean())**(1/2)

# Print RMSE_CV
print('CV RMSE: {:.2f}'.format(RMSE_CV))
################################################
#Evaluate the training error
################################################
from sklearn.metrics import mean_squared_error as MSE 
dt.fit(X_train, y_train)# Fit dt to the training set
y_pred_train = dt.predict(X_train)# Predict the labels of the training set
RMSE_train = (MSE(y_train, y_pred_train))**(1/2)# Evaluate the training set RMSE of dt
print('Train RMSE: {:.2f}'.format(RMSE_train))# Print RMSE_train
################################################
#Ensemble learning
    
################################################

################################################
#Evaluate the training error
################################################
################################################
#Evaluate the training error
################################################
################################################
#Evaluate the training error
################################################
################################################
#Evaluate the training error
################################################
################################################
#Evaluate the training error
################################################