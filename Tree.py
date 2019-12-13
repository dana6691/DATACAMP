#####################################################
#Supervised Learning
#####################################################
#Decision Regions: region in the feature space, (negative/positive decision by the decision boundary)
#Classification And Regression Tree(CART)
    #sequence of if-else about individual features
    #Goal: infer class labels
    #Able to capture non-linear relationship between features and labels
    #Don't require feature scaling(ex)standardization)
        #ex)Cancer(Y/N)
#####################################################
#Classification tree
import pandas as pd
df2 = pd.read_csv("C:/Users/daheekim/Desktop/datacamp_data/breast_cancer.csv")
#print(df.head())
y=df2['diagnosis']
#print(list(df2.columns) )
X=df2['radius_mean','concave points_mean']#??????????????
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y, random_state=1)
df=DecisionTreeClassifier(max_depth=2, random_state=1)
df.fit(X_train,y_train)
y_pred=df.predict(X_test) # Predict test set labels
print(y_pred[0:5])

from sklearn.metrics import accuracy_score
y_pred = dt.predict(X_test)# Predict test set labels
acc = accuracy_score(y_test, y_pred)# Compute test set accuracy
print("Test set accuracy: {:.2f}".format(acc))
#####################################################
#Logistic Regreession VS Classification Tree
from sklearn.linear_model import  LogisticRegression
logreg = LogisticRegression(random_state=1)
logreg.fit(X_train, y_train)# Fit
clfs = [logreg, dt]# Define a list called clfs containing the two classifiers logreg and dt
plot_labeled_decision_regions(X_test, y_test, clfs)# Plot decision regions of the two classifiers
#####################################################
#Decision tree for Classification
    #node: Root(no parent node,2 children nodes)/Internal Node(1 parent node,2 children nodes)/Leaf()
        #The existence of a node depends on the state of its predecessors.
    #criteria to measure the impurity of a node:
        #1)gini index
        #2)entropy
    #at each node, split the data based on feature and split-point to maximize information gain(IG)
#####################################################
#Entropy 
from sklearn.tree import DecisionTreeClassifier
dt_entropy = DecisionTreeClassifier(max_depth=8,  criterion='entropy', random_state=1) #'entropy' as the information criterion
dt_entropy.fit(X_train, y_train)# Fit dt_entropy to the training set
#Gini index
from sklearn.tree import DecisionTreeClassifier
dt_gini = DecisionTreeClassifier(max_depth=8,  criterion='gini', random_state=1) #'gini' as the information criterion
dt_gini.fit(X_train, y_train)# Fit dt_gini to the training set
#Entropy vs Gini index
from sklearn.metrics import accuracy_score
y_pred= dt_entropy.predict(X_test) # Use dt_entropy
accuracy_entropy = accuracy_score(y_test, y_pred)# Evaluate accuracy_entropy
y_pred= dt_gini.predict(X_test) # Use dt_gini
accuracy_gini = accuracy_score(y_test, y_pred)# Evaluate accuracy_gini
print('Accuracy achieved by using entropy: ', accuracy_entropy)# Print accuracy_entropy
print('Accuracy achieved by using the gini index: ', accuracy_gini)# Print accuracy_gini
#####################################################
#Decision tree for Classification
#####################################################
#Regression tree
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=8,
             min_samples_leaf=0.13,
            random_state=3)
dt.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error as MSE
y_pred = dt.predict(X_test)# Compute y_pred
mse_dt = MSE(y_test, y_pred)# Compute mse_dt
rmse_dt = mse_dt**(1/2)# Compute rmse_dt
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))
#Linear regression
y_pred_lr = lr.predict(X_test)# Predict test 
mse_lr = MSE (y_pred_lr, y_test)# Compute mse_lr
rmse_lr = mse_lr**(1/2)# Compute rmse_lr
print('Linear Regression test set RMSE: {:.2f}'.format(rmse_lr))
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_dt))
################################################
#Supervised Learning
    #Fit the model f(x) that best approximates(f(x) can be logistic regression, decision tree, neural network)
    #discard noise as much as possible
    #low predictive error on unseen dataset
#difficulties
    #overfitting: predictive power is low
    #underfitting: training set error = test set error ,but errors are high
#Generalization Error = bias^2 + variance + irreducible error
    '''*bias(accurate=low bias): how much f and f^ different,High bias = Underfitting
       *variance(precise=low variance): how much f^ is inconsistent over different training sets, High variance = Overfitting
       *model complexity: flexibility of f^ (increase --> variance increase, bias decreases) ---> Bias-variance Tradeoff
          #goal: find the lowest model complexity, generalization error'''
################################################
################################################
#Diagnosis bias and variance = estimating Generalization Error
    '''1)split data into training and test
        2) fit f^ to training set
        3) evaluate error of f^ on test set
        4) generalization error of f^ = test set error of f^'''
#Better model evaluation with Cross-Validation
    #K-fold CV: CV error = (E1+E2+E3..E10)/10, training fold=9, validation fold=1
    #Hold-out CV
#If f^ high variance: CV error of f^ > training error of f^
    #Remedy overfitting
        '''1)decrease model complexity
           2)gather mode data'''
#if f^ high bias: CV error of f^ = training error of f^ >> desired error
    #Remedy underfitting
        '''1)increase model complexity
           2)gather more relevant features'''

################################################
from sklearn.model_selection import train_test_split
SEED = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)# Split the data into 70% train and 30% test
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26, random_state=SEED)
#Evaluate 10-fold CV error
MSE_CV_scores = - cross_val_score(dt, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1) # 10-folds CV MSEs
RMSE_CV = (MSE_CV_scores.mean())**(1/2)# Compute the 10-folds CV RMSE
print('CV RMSE: {:.2f}'.format(RMSE_CV))
#Evaluate the training error
from sklearn.metrics import mean_squared_error as MSE 
dt.fit(X_train, y_train)
y_pred_train = dt.predict(X_train)# Predict
RMSE_train = (MSE(y_train, y_pred_train))**(1/2)# Evaluate the training set RMSE of dt
print('Train RMSE: {:.2f}'.format(RMSE_train))
################################################
<<<<<<< HEAD
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
=======
#CARTs(Advan)
    #simpled to understand/interpret
    #easy to use
    #flexibility: ability to describe non-linear dependecies
    #no need to standardize or normalization features
#CARTs(DisAdvan)
    #Classification: only produce orthogonal decision boundaries
    #Sensitive to small variations in training set
    #High variance: unconstrained CARTs may overfit training set
        #Solution: Ensembal learning
#Ensemble learning
    #train different models on the same dataset
    #each model makes predictions
    #Meta-model: aggregates predictions of individual model
    #Final prediction: more robust and less prone to errors
#Ensemble Final prediction
    #Voting Classifier
        '''1)Hard Voting: binary classification(0/1)
           #Same training set
           #!= algorithm'''
    #Bagging(Bootstrap Aggregation) Classifier
        '''#one algorithm
           #Different subsets of training set
               #reduce variane of individual models in the ensemble
            #Classification: majority voting
            #Regression: averaing '''
        #problem: some instances may not be sampled at all
            #not sampled instances == Out of Bag(OOB)
################################################
#Ensemble
SEED=1
lr = LogisticRegression(random_state=SEED)
knn = KNN(n_neighbors=27)
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]# Define the list classifiers
for clf_name, clf in classifiers:    
    clf.fit(X_train, y_train)      # Fit clf to the training set  
    y_pred = clf.predict(X_test)    # Predict y_pred
    accuracy = accuracy_score(y_test, y_pred)     # Calculate accuracy
    print('{:s} : {:.3f}'.format(clf_name, accuracy))    # Evaluate clf's accuracy on the test set
#Voting Classifier 
from sklearn.ensemble import VotingClassifier 
vc = VotingClassifier(estimators=classifiers)
vc.fit(X_train, y_train)   
y_pred = vc.predict(X_test)# Evaluate the test set predictions
accuracy = accuracy_score(y_test, y_pred)# Calculate accuracy score
print('Voting Classifier: {:.3f}'.format(accuracy))
#Bagging Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier 
dt = DecisionTreeClassifier(random_state=1)
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=1)
bc.fit(X_train, y_train) #fit
y_pred = bc.predict(X_test)# Predict test set labels
acc_test = accuracy_score(y_test, y_pred)# Evaluate acc_test
print('Test set accuracy of bc: {:.2f}'.format(acc_test)) 
#OOB evaluation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
dt = DecisionTreeClassifier(min_samples_leaf=8, random_state=1)
bc = BaggingClassifier(base_estimator=dt, 
            n_estimators=50,
            oob_score=True,
            random_state=1)
bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)# Predict 
acc_test = accuracy_score(y_test, y_pred)# Evaluate test set accuracy
acc_oob = bc.oob_score_# Evaluate OOB accuracy
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))# Print acc_test and acc_oob
################################################
#Random Forest
    '''*base estimator: decision tree
        *each estimtaor is trained on a different bootstrap sample having the same size as the training set
        *further randomization in the training of individual tress
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
>>>>>>> fe8248e3e50286fce4a98deef0f29a264934be59
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
