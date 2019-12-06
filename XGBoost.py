#################################################
#Supervised machine learning
# regression task of predicting house prices in Ames, Iowa.
    #AUC(Area Under ROC Curve): metrics for binary classfication model
    #Confusion matrix
    #Accuracy score
#data type
    #categorical: one-hot encoded
    #numeric: scaled(Z-scored)
#Ranking problem: predicting an ordering on a set of choices
#Recommendation: recommand item based on history
#################################################
#################################################
#XGBoost
    '''speed and performance
        core algorithm is parallelizable
        consistently outperforms single-algorithm methods'''
#################################################
import xgboost as xgb
X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=123)
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)
xg_cl.fit(X_train,y_train)
preds = xg_cl.predict(X_test)
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]# Compute the accuracy: accuracy
print("accuracy: %f" % (accuracy))
#################################################
#Decision Tree
    #composed of series of binary questions
    #prediction happened at "leaves" tree
    #constructed iteratively
        #until a stopping criterion is met 
    #--> usually overfiting(low bias,high variance)   
#################################################
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
dt_clf_4 = DecisionTreeClassifier(max_depth=4) #decision tree
dt_clf_4.fit(X_train,y_train)
y_pred_4 = dt_clf_4.predict(X_test)
accuracy = float(np.sum(y_pred_4==y_test))/y_test.shape[0]# Compute the accuracy of the predictions: accuracy
print("accuracy:", accuracy)
#################################################
#Boosting
    #Weak learning: ML algorithm, that is slightly better than chance
    #strong learner: Any algorithm that can be tuned to achieve good performance
    #boosting converts weak learner to strong learner
        '''1)learning a set of weak models on subsets of the data
            2)weighinh each weak prediction 
            3)combine the weighted predictions and obtain a single weighted prediction'''
#Model evaluation thru Cross-validation: robuest method for estimating the perfformance of model on unseen data
    #generate many non-overlapping train/test splits on training data
    #reports average test set performance across all data splits
#################################################
# Measuring accuracy
churn_dmatrix = xgb.DMatrix(data=X, label=y)# Create the DMatrix: churn_dmatrix
params = {"objective":"reg:logistic", "max_depth":3}# Create the parameter dictionary: params
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=3, num_boost_round=5, metrics="error", as_pandas=True, seed=123)
print(cv_results)
print(((1-cv_results["test-error-mean"]).iloc[-1]))
# Measuring AUC
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=3, num_boost_round=5, metrics="auc", as_pandas=True, seed=123)
print(cv_results)
print((cv_results["test-auc-mean"]).iloc[-1])# Print the AUC
#################################################
#XGBoost When to use it?
    #Large number of training samples: greater than 1000 training samples, less 100 features
    #Mixture of categorical and numeric features or just numeric features
#Not use?
    #image recognition
    #Computer vision
    #Natural Lanuage preprocessing
    # # of training samples is significantly smaller than features
##Regression with XGBoost
    #Metrics: 
     '''1)RMSE(Root Mean Square Error)
        2)MAE(Mean Absolute Error)'''
#Objective(loss) function
    #Quantifies how far off the predictions from the actual result
    #Measures the difference between estimated and true values
    #Goal: find the model that yield the minimum value of the loss function
        #Regression: 'reg:linear
        #Classification, just decision:'reg:logistic'
        #Want probability: 'binary:logistic'
#Base learners
    #individual models=base learners
    #combined to create final prediction that is non-linear
    #two type: linear, tree
#################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_reg = xgb.XGBRegressor()
xg_reg.fit(X_train,y_train)# Fit the regressor to the training set
preds = xg_reg.predict(X_test)# Predict the labels of the test set: preds
rmse = np.sqrt(mean_squared_error(y_test, preds))# Compute the rmse: rmse
print("RMSE: %f" % (rmse))

#Linear base learners
DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test =  xgb.DMatrix(data=X_test, label=y_test)
params = {"booster":"gblinear", "objective":"reg:linear"}# Create the parameter dictionary: params
xg_reg = xgb.train(params = params, dtrain=DM_train, num_boost_round=5)# Train the model: xg_reg
preds = xg_reg.predict(DM_test)
rmse = np.sqrt(mean_squared_error(y_test,preds)) #RMSE
print("RMSE: %f" % (rmse))

#Evaluating model quality
housing_dmatrix = xgb.DMatrix(data=X, label=y)# Create the DMatrix:
params = {"objective":"reg:linear", "max_depth":4}# Create the parameter dictionary
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4, num_boost_round=5, metrics='mae', as_pandas=True, seed=123)# Perform cross-validation: cv_results
print(cv_results)
print((cv_results["test-mae-mean"]).tail(1))
#################################################
#Regularization in XGBoost
    #want model to contorl on complexity
    #want model to both accruate and as simple as possible
    #regularization parameters
        '''gamma: minimum loss reduction allowed for a split to occur 
            alpha: L1 regularization on leaf weights. larger values mean more regularization
            lambda - L2 regularization on leaf weights'''
#Base Learner:
    #Linear: sum of linear term, 
        #weighted sum of linear model --> rarely used
    #Tree based learner: decision tree
        #weighted sume of decision tress(nonlinear)
        # exclusively used
#################################################
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)
reg_params = [1, 10, 100]
params = {"objective":"reg:linear","max_depth":3}# Create the initial parameter dictionary for varying l2 strength:
rmses_l2 = []
for reg in reg_params:
    params["lambda"] = reg# Update l2 strength
    cv_results_rmse = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2, num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)
    rmses_l2.append(cv_results_rmse["test-rmse-mean"].tail(1).values[0])  # Append
print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2","rmse"]))

#Visualizing individual XGBoost trees
housing_dmatrix = xgb.DMatrix(data=X, label=y)# Create the DMatrix
params = {"objective":"reg:linear", "max_depth":2}# Create the parameter dictionary
xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)# Train the model

# Plot the first tree
xgb.plot_tree(xg_reg,num_trees=0)
plt.show()

# Plot the fifth tree
xgb.plot_tree(xg_reg,num_trees=4)
plt.show()

# Plot the last tree sideways
xgb.plot_tree(xg_reg,num_trees=9,rankdir="LR")
plt.show()

#Visualizing feature importances:
housing_dmatrix = xgb.DMatrix(data=X, label=y)# Create the DMatrix
params = {"objective":"reg:linear","max_depth":4 }# Create the parameter dictionary
xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)# Train the model:
xgb.plot_importance(xg_reg)# Plot the feature importances
plt.show()
#################################################
# Tune XGBoost
#################################################
#Un-Tuned model
import pandas as pd
import xgboost as xgb
import numpy as npb
housing_data=pd.read_csv("ames_housing_trimmed_processed.csv")
X,y=housing_data[housing_data.colunns.tolist()[:-1]],housing_data[housing_data.columns.tolist()[-1]]
housing_dmatrix = xgb.DMatrix(data=X,lavel=y)
untuned_params={"objective":"reg:linear"}
untuned_cv_results_rmse=xgb.cv(dtrain=housing_dmatrix, params=untuned_params,
nfold=4, metrics="rmse", as_pandas=True, seed=123)
#Tuned model
tuned_params={"objective":"reg:linear",'colsample_bytree':0.3,'learning_rate':0.1,'max_depth':5}
tuned_cv_results_rmse=xgb.cv(dtrain=housing_dmatrix, params=tuned_params,
nfold=4, num_boost_round=200,metrics="rmse", as_pandas=True, seed=123)

#Tuning the number of boosting rounds
housing_dmatrix = xgb.DMatrix(data=X, label=y)# Create the DMatrix
params = {"objective":"reg:linear", "max_depth":3}# Create the parameter dictionary
num_rounds = [5, 10, 15]# Create list of number of boosting rounds
final_rmse_per_round = []# Empty list 
for curr_num_rounds in num_rounds:
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=curr_num_rounds, metrics="rmse", as_pandas=True, seed=123)
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses,columns=["num_boosting_rounds","rmse"]))

#Automated boosting round selection 
housing_dmatrix = xgb.DMatrix(data=X, label=y)# Create your housing DMatrix
params = {"objective":"reg:linear", "max_depth":4}# Create the parameter dictionary
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=50, metrics="rmse", as_pandas=True, seed=123)
print(cv_results)
