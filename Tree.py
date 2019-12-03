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
#Random Forest
    '''*base estimator: decision tree
        *each estimtaor is trained on a different bootstrap sample having the same size as the training set
        *further randomninzation in the training of individual tress
        *'d' features are sampled at each node *without* replacement'''
    '''1)each sample different features
        2)decision tree
        3)prediction
        4)final prediction made by majority voting(for classfication)
                                by averaging(for Regression)'''
#Bagging
    '''*base: Decision tree,Logistic Regression,Neural Network..
        *each estimator is trained on a distinct bootstrap sample of the training set
        *estimators use all features for training and prediction'''    
################################################
#fit
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=25,
            random_state=2) 
rf.fit(X_train, y_train) # Fit rf to the training set 

#evaluate prediction
from sklearn.metrics import mean_squared_error as MSE # Import mean_squared_error as MSE
y_pred = rf.predict(X_test)# Predict the test set labels
rmse_test = MSE(y_test, y_pred)**(1/2)# Evaluate the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))# Print rmse_test

#Visualizing features importantces
importances = pd.Series(data=rf.feature_importances_,
                        index= X_train.columns)# Create a pd.Series of features importances
importances_sorted = importances.sort_values()# Sort importances
importances_sorted.plot(kind='barh', color='lightgreen')# Draw a horizontal barplot of importances_sorted
plt.title('Features Importances')
plt.show()
################################################
#Boosting: ensemble method combining several weak learners to foram a strong learner
#AdaBoost(Adaptive Boosting)
    #focus on worngly predicted by predecessor
    #achieved by chainging the weights of training instances
    #each predictor is assigned a coefficients alpha(a)
    #alpha depends onn the predictors's training error
    #learning rate: alpha1 = eta*alpha1
        #classification:weighted majority voting prediction
        #Regression: weighted average prediction
################################################
from sklearn.tree import DecisionTreeClassifier# Import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier# Import AdaBoostClassifier
dt = DecisionTreeClassifier(max_depth=2, random_state=1)# Instantiate dt
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1)# Instantiate ada

ada.fit(X_train,y_train)# Fit ada to the training set
y_pred_proba = ada.predict_proba(X_test)[:,1]# Compute the probabilities of obtaining the positive class

from sklearn.metrics import roc_auc_score# Import roc_auc_score
ada_roc_auc = roc_auc_score(y_test, y_pred_proba)# Evaluate test-set roc_auc_score
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))# Print roc_auc_score
################################################
#Gradient Boosting Tress
    #sequential correction of predecessor's error
    #Do not tweak the weights of training instances
    #fit each predictor using its predecessor's residual errors
    #CART is used as as base learning
    #shrink after apply learning rate
        #prediction
            #regression: ypredi = y1 +learning rate*r1 + learningrate*r2....
            #classification
################################################
#bike shring demand dataset:  predict the bike rental demand using historical weather data from the Capital Bikeshare program in Washington, D.C.
from sklearn.ensemble import GradientBoostingRegressor# Import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
SEED=1
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=SEED)
gb = GradientBoostingRegressor(n_estimators=200, max_depth=4,random_state=2)
gb.fit(X_train, y_train) #fit
y_pred = gb.predict(X_test)# Predict test set labels

from sklearn.metrics import mean_squared_error as MSE
mse_test = MSE(y_test,y_pred)# Compute MSE
rmse_test = mse_test**(1/2)# Compute RMSE
print('Test set RMSE of gb: {:.3f}'.format(rmse_test))
################################################
#Stochastic Gradient Boosting
    #Each tree is trained on a random subset of rows of the training data
    #Sampled instances are 40-80% of training set are sampled (without replacement)
    #Features are sampled (without replacement) when choosing split points
        #Add further variance to ensemble of trees
################################################
from sklearn.ensemble import GradientBoostingRegressor# Import GradientBoostingRegressor
sgbr = GradientBoostingRegressor(max_depth=4, 
            subsample=0.9,
            max_features=0.75,
            n_estimators=200,                                
            random_state=2)
sgbr.fit(X_train, y_train)
y_pred =  sgbr.predict(X_test)
from sklearn.metrics import mean_squared_error as MSE
mse_test = MSE(y_test,y_pred)# Compute MSE
rmse_test = mse_test**(1/2)# Compute RMSE
print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))
################################################
#Tuning CART's Hyperparameter
    #Machine learning model:
        #Parameter: learned from data  ex) split-point of node, split-feature of a node
        #Hyperparameters: set prior to training ex) max_depth, min_samples_leaf...
#Hyperparameter tuning:search for a set of optimal hyperparameters for a learning algorithm
    #solution: find  in an optimal model
    #Optimal model: yields an optimal score
    #Score: classfication: accuracy
            #regression: R^2
    #Cross validation is used to estimate the generalization performance
 #Why tune hyperparameters?
    #default hyperparameters are not optimal for all problems, for the best performance
    '''1)Grid Search
        2)Random Search
        3)Bayesian Search
        4)Genetic Search'''
#Grid Search Cross Validation
    #-manually set a grid of discrete hyperparameter values
    #-set a metric for scoring model performance
    #-for each hyperparameters, evaluate each model's CV(cross validation)-score, best CV is the optimal hyperparameters
    #-the bigger the grid, the longer to take to find the solution
################################################
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
print(dt.get_params())
from sklearn.model_selection import GridSearchCV
params_dt = {'max_depth':[2,3,4] ,'min_samples_leaf':[0.12, 0.14, 0.16, 0.18]} # Define params_dt
grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt, scoring='accuracy',cv=10,n_jobs=1)
grid_dt = GridSearchCV(estimator=dt,param_grid=params_dt,scoring='roc_auc',cv=5, n_jobs=-1)

from sklearn.metrics import roc_auc_score
best_model = grid_dt.best_estimator_# Extract the best estimator
y_pred_proba = best_model.predict_proba(X_test)[:,1]# Predict the test set probabilities of the positive class
test_roc_auc = roc_auc_score(y_test, y_pred_proba)# Compute test_roc_auc
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))# Print test_roc_auc
################################################
#Tuning RF's Hyperparameter
    #Hyperparameter tuning is expensive, somtimes only slight improvement
    #Weight the impact of using hyperparameter
    #
################################################
from sklearn
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV
rf =RandomForestRegressor()
print(rf.get_params())
params_rf = {'n_estimators':[ 100, 350, 500],
    'max_features':['log2', 'auto', 'sqrt'],
    'min_samples_leaf':[2, 10, 30] }# Define the dictionary 'params_rf'
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       scoring='neg_mean_squared_error',
                       cv=3,
                       verbose=1,
                       n_jobs=-1)
best_model = grid_rf.best_estimator_
y_pred = best_model.predict(X_test)# Predict test set labels
rmse_test = MSE(y_test,y_pred)**(1/2)# Compute rmse_test
print('Test RMSE of best model: {:.3f}'.format(rmse_test)) 
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