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
#Ensemble learning:
    #train different models on the same dataset
    #each model make its predictions
    #meta-model: aggregates predictions of individual model
    #final prediction: more robust and less prone to error
    #*hard voting: if two classfiers use 1 for predictions and one use 0, then choose 1
#CARTs(categorical regression tree)
    '''-simple to understand
        - simple to interpret
        - flexibility: ability to describe non-linear dependencies
        - preprocessing: no need standardization or normalization
    '''
    '''- classification: can only produce orthogonal decision boundaries
        - sensitive to small variations in the training sets
        - high variance: unconstrained CARTs may overfitting the train set
        --> solution: ensemble learning
    '''
################################################
#Multiple classifiers
SEED=1
lr = LogisticRegression(random_state=SEED)
knn = KNeighborsClassifier(n_neighbors=27)
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]
for clf_name, clf in classifiers:    
    clf.fit(X_train, y_train)     # Fit clf to the training set
    y_pred = clf.predict(X_test)# Predict y_pred
    accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy
    print('{:s} : {:.3f}'.format(clf_name, accuracy))# Evaluate clf's accuracy on the test set
#VotingClassifier 
from sklearn.ensemble import VotingClassifier # Import VotingClassifier from sklearn.ensemble
vc = VotingClassifier(estimators=classifiers)     # Instantiate a VotingClassifier vc
vc.fit(X_train, y_train)   # Fit vc to the training set
y_pred = vc.predict(X_test)# Evaluate the test set predictions
accuracy = accuracy_score(y_test, y_pred)# Calculate accuracy score
print('Voting Classifier: {:.3f}'.format(accuracy))
################################################
#Bagging/ensemble
    #one algorithm/ multiple algorithms
    #subsets of training sets/ same training set
#Bagging(Bootstrap Aggregation)
    #reduces variance of individual model in ensemble
    #1) if original set(A,B,C) --> extract 3 times(allow to draw same dataset multiple times)
    #2) Different bootstrap samples and predicts all models on same algorithm
    #3) Collect predictions
    #4) final prediction: Classification: majority voting/ Regression: average
################################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier # Import BaggingClassifier
dt = DecisionTreeClassifier(random_state=1)

bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=1)
bc.fit(X_train, y_train)# Fit bc to the training set
y_pred = bc.predict(X_test)# Predict test set labels
acc_test = accuracy_score(y_test, y_pred)# Evaluate acc_test
print('Test set accuracy of bc: {:.2f}'.format(acc_test)) 
################################################
#Out-of-Bag
################################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
dt = DecisionTreeClassifier(min_samples_leaf=8, random_state=1)
bc = BaggingClassifier(base_estimator=dt, 
            n_estimators=50,
            oob_score=True,
            random_state=1)
bc.fit(X_train, y_train)# Fit bc to the training set
y_pred = bc.predict(X_test)# Predict test set labels
acc_test = accuracy_score(y_test, y_pred)# Evaluate test set accuracy
acc_oob = bc.oob_score_ # Evaluate OOB accuracy
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))
################################################
#Random Forrests
################################################
from sklearn.ensemble import RandomForestRegressor

# Instantiate rf
rf = RandomForestRegressor(n_estimators=25,
            random_state=2)
            
# Fit rf to the training set    
rf.fit(X_train, y_train) 
################################################
#Evaluate the training error
################################################
################################################
#Evaluate the training error
################################################