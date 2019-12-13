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
#################################################
#Tuning eta
#################################################
housing_dmatrix = xgb.DMatrix(data=X, label=y)
params = {"objective":"reg:linear", "max_depth":3}# Create the parameter dictionary for each tree (boosting round)

# Create list of eta values and empty list to store final round rmse per xgboost model
eta_vals = [0.001, 0.01, 0.1]
best_rmse = []

for curr_val in eta_vals:
    params["eta"] = curr_val
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=10, metrics="rmse", as_pandas=True, seed=123)
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1]) # Append the final round rmse to best_rmse
print(pd.DataFrame(list(zip(eta_vals, best_rmse)), columns=["eta","best_rmse"]))
#################################################
#Tuning max_depth
#################################################
housing_dmatrix = xgb.DMatrix(data=X,label=y)
params = {"objective":"reg:linear"}# Create the parameter dictionary
max_depths = [2, 5, 10,  20]# Create list of max_depth values
best_rmse = []

for curr_val in max_depths:
    params["max_depth"] = curr_val
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2, num_boost_round=10, metrics="rmse", as_pandas=True, seed=123)
#################################################
#Tuning colsample_bytree
#################################################
housing_dmatrix = xgb.DMatrix(data=X,label=y)
params={"objective":"reg:linear","max_depth":3}# Create the parameter dictionary

colsample_bytree_vals = [0.1, 0.5, 0.8,  1]# Create list of hyperparameter values: colsample_bytree_vals
best_rmse = []

for curr_val in colsample_bytree_vals:
    params["colsample_bytree_vals"] = curr_val
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2,
                 num_boost_round=10, early_stopping_rounds=5,
                 metrics="rmse", as_pandas=True, seed=123)
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])
print(pd.DataFrame(list(zip(colsample_bytree_vals, best_rmse)), columns=["colsample_bytree","best_rmse"]))
#################################################
#find the optimal value(choose hyperparameter), lowest loss possible.
#Grid Search
    '''1) search over given set of hyperparameters
        2) number of models= number of distinct values per huperparameter * each hyperparameter
        3) Pick fianl model hyperparameter values that gives best corss-valudation metrics'''
#Random search
    '''1) create range of hyperparameter values per hyperparameter
        2) set the number of iteration you would like
        3) During the iteration, ramdomly draw a value in the range for each huperparameter searched
        4) After reached maximum number of iteration, select the hyperparameter configuration with the best evaluated score''' 
    #scoring method: 'neg_mean_squared_error'
#################################################
#Grid Search
#################################################
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter grid
gbm_param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'n_estimators': [50],
    'max_depth': [2, 5]
}
gbm = xgb.XGBRegressor()# Instantiate the regressor: gbm
# grid search
grid_mse = GridSearchCV(param_grid=gbm_param_grid,
                        estimator=gbm,
                        scoring="neg_mean_squared_error",cv=4)
grid_mse.fit(X,y)# Fit 
print("Best parameters found: ", grid_mse.best_params_)#best parameters and lowest RMSE
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))
#################################################
#Random search
#################################################
gbm_param_grid = {
    'n_estimators': [25],
    'max_depth': range(2, 12)
}
gbm = xgb.XGBRegressor(n_estimators=10)#regressor

# Perform random search: grid_mse
randomized_mse = RandomizedSearchCV(param_distributions=gbm_param_grid,
                        estimator=gbm,
                        scoring="neg_mean_squared_error",cv=4)
randomized_mse.fit(X,y)# Fit
print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))
#################################################
#Preprocessing 1: Label Encoder(convert categorical to integer) and 
                #OneHotEncoder (integers into dummy variables)
#Preprocessing 2: DictVectorizer
#################################################
#Label Encoder
from sklearn.preprocessing import LabelEncoder
df.LotFrontage = df.LotFrontage.fillna(0)# Fill missing values with 0
categorical_mask = (df.dtypes == object)# Create a boolean mask for categorical columns
categorical_columns = df.columns[categorical_mask].tolist()# Get list of categorical column names
print(df[categorical_columns].head())# Print the head of the categorical columns
le = LabelEncoder()# Create LabelEncoder object: le
df[categorical_columns] = df[categorical_columns].apply(lambda x: le.fit_transform(x))# Apply LabelEncoder to categorical columns
print(df[categorical_columns].head())
#OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=categorical_mask, sparse=False)# Create OneHotEncoder: ohe
df_encoded = ohe.fit_transform(df)# Apply OneHotEncoder to categorical columns 
print(df_encoded[:5, :])# Print first 5 rows of the resulting dataset
print(df.shape)# Print the shape of the original DataFrame
print(df_encoded.shape)# Print the shape of the transformed array
#DictVectorizer
from sklearn.feature_extraction import DictVectorizer
df_dict = df.to_dict("records")# Convert df into a dictionary: df_dict
dv = DictVectorizer(sparse=False)# Create the DictVectorizer object: dv
df_encoded = dv.fit_transform(df_dict)# Apply 
print(df_encoded[:5,:])# Print the resulting first five rows
print(dv.vocabulary_)# Print the vocabulary
#################################################
#Preprocessing within a pipeline
#################################################
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline  
X.LotFrontage = X.LotFrontage.fillna(0)# Fill LotFrontage missing values with 0

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor())]

xgb_pipeline = Pipeline(steps)# Create the pipeline: xgb_pipeline
xgb_pipeline.fit(X.to_dict("records"), y)# Fit the pipeline
#################################################
#Cross-validating your XGBoost model
#################################################
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
X.LotFrontage = X.LotFrontage.fillna(0)# Fill LotFrontage missing values with 0
# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor(max_depth=2, objective="reg:linear"))]
xgb_pipeline = Pipeline(steps)# Create the pipeline: xgb_pipeline
cross_val_scores = cross_val_score(xgb_pipeline, X.to_dict("records"), y, cv=10, scoring="neg_mean_squared_error")# Cross-validate the model
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))# Print the 10-fold RMSE
