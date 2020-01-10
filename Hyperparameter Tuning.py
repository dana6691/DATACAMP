################################################
#parameter 
    #logistic regression: top 3 coefficients
    #tree base, random forrest: no coefficients, but node decision(what features, what value to split)
################################################
#logistic regression
# Create a list of original variable names from the training DataFrame
original_variables = list(X_train.columns)

# Extract the coefficients of the logistic regression estimator
model_coefficients = log_reg_clf.coef_[0]

# Create a dataframe of the variables and coefficients & print it out
coefficient_df = pd.DataFrame({"Variable" : original_variables, "Coefficient": model_coefficients})
print(coefficient_df)

# Print out the top 3 positive variables
top_three_df = coefficient_df.sort_values(by='Coefficient', axis=0, ascending=False)[0:3]



#Extracting a Random Forest parameter
# Extract the 7th (index 6) tree from the random forest
chosen_tree = rf_clf.estimators_[6]

# Visualize the graph using the provided image
imgplot = plt.imshow(tree_viz)
plt.show()

# Extract the parameters and level of the top (index 0) node
split_column = chosen_tree.tree_.feature[0]
split_column_name = X_train.columns[split_column]
split_value = chosen_tree.tree_.threshold[0]

# Print out the feature and level
print("This node split on feature {}, at a value of {}".format(split_column_name, split_value))

#Hyperparameter: something you set before the modelling process, it also 'tune' your hyperpaprameters
    #important hyperparameter
        #Random forrest classifier:
            #n_jobs
            #random_state
            #verbose
        #Random forrest 
            #n_estimators
            #max_features
            #max_depth, min_sample_leaf
            #criterion

################################################
#Exploring Random Forest Hyperparameters
# Print out the old estimator, notice which hyperparameter is badly set
print(rf_clf_old)

# Get confusion matrix & accuracy for the old rf_model
print("Confusion Matrix: \n\n {} \n Accuracy Score: \n\n {}".format(confusion_matrix(y_test, rf_old_predictions),  accuracy_score(y_test, rf_old_predictions))) 

# Create a new random forest classifier with better hyperparamaters
rf_clf_new = RandomForestClassifier(n_estimators=500)

# Fit this to the data and obtain predictions
rf_new_predictions = rf_clf_new.fit(X_train, y_train).predict(X_test)

# Assess the new model
print("Confusion Matrix: \n\n", confusion_matrix(y_test,rf_new_predictions))
print("Accuracy Score: \n\n", accuracy_score(y_test,rf_new_predictions))
'''nice 5% accuracy boost just from changing the n_estimators'''
################################################
#Hyperparameters of KNN
################################################
#Manual Hyperparameter Choice
################################################
# Build a knn estimator for each value of n_neighbours
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_10 = KNeighborsClassifier(n_neighbors=10)
knn_20 = KNeighborsClassifier(n_neighbors=20)

# Fit each to the training data & produce predictions
knn_5_predictions = knn_5.fit(X_train,y_train).predict(X_test)
knn_10_predictions = knn_10.fit(X_train,y_train).predict(X_test)
knn_20_predictions = knn_20.fit(X_train,y_train).predict(X_test)

# Get an accuracy score for each of the models
knn_5_accuracy = accuracy_score(y_test, knn_5_predictions)
knn_10_accuracy = accuracy_score(y_test, knn_10_predictions)
knn_20_accuracy = accuracy_score(y_test, knn_20_predictions)
print("The accuracy of 5, 10, 20 neighbours was {}, {}, {}".format(knn_5_accuracy, knn_10_accuracy, knn_20_accuracy))
'''The accuracy of 5, 10, 20 neighbours was 0.7125, 0.765, 0.7825'''
'''it's good but too much time consuming'''
################################################
#Automating Hyperparameter Choice
################################################
# Set the learning rates & results storage
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2 , 0.5]
results_list = []

# Create the for loop to evaluate model predictions for each learning rate
for numb in learning_rates:
    model = GradientBoostingClassifier(learning_rate=numb)
    predictions = model.fit(X_train, y_train).predict(X_test)
    # Save the learning rate and accuracy score
    results_list.append([numb, accuracy_score(y_test, predictions)])

# Gather everything into a DataFrame
results_df = pd.DataFrame(results_list, columns=['learning_rate', 'accuracy'])
print(results_df)
################################################
#Building Learning Curves
################################################
# Set the learning rates & accuracies list
learn_rates = np.linspace(0.01, 2, num=30)
accuracies = []

# Create the for loop
for learn_rate in learn_rates:
  	# Create the model, predictions & save the accuracies as before
    model = GradientBoostingClassifier(learning_rate=learn_rate)
    predictions = model.fit(X_train, y_train).predict(X_test)
    accuracies.append(accuracy_score(y_test, predictions))

# Plot results    
plt.plot(learn_rates, accuracies)
plt.gca().set(xlabel='learning_rate', ylabel='Accuracy', title='Accuracy for different learning_rates')
plt.show()
'''r once the learning rate pushes much above 1.5, the accuracy starts to drop. You have learned and practiced a useful skill for visualizing large amounts of results for a single hyperparameter.'''
################################################
#Hyperparmeter conflicts
    #logistic: solver,penalty
    #random forest: no low number of trees
    #KNN: 1 neighbor and averagin 'votes' doesn't make sense
################################################

################################################
#More than 2 Hyperparameters: Using GBM(Gradient Boosting) algorithm
    #learn_rate
    #max_depth
#Automating 2 hyperparmaeters
#Grid Search    
    #Advan: don't ned to long line of ode
        #Find the best model within the grid
        #easy to explina
    #Disadvan: computationally expensive
        #uninformed; results of one model doesn't help to create next one
################################################
#Grid Search (two hyperparameters for the GBM algorithm)
# Create the function
def gbm_grid_search(learn_rate, max_depth):

	# Create the model
    model = GradientBoostingClassifier(learning_rate=learn_rate, max_depth=max_depth)
    
    # Use the model to make predictions
    predictions = model.fit(X_train, y_train).predict(X_test)
    
    # Return the hyperparameters and score
    return([learn_rate, max_depth, accuracy_score( y_test,predictions)])


# Create the relevant lists
results_list = []
learn_rate_list = [0.01, 0.1, 0.5]
max_depth_list = [2, 4, 6]

# Create the for loop
for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
        results_list.append(gbm_grid_search(learn_rate,max_depth))

# Print the results
print(results_list)  

################################################
#Extend the function(2 to 3)
def gbm_grid_search_extended(learn_rate, max_depth, subsample):

	# Extend the model creation section
    model = GradientBoostingClassifier(learning_rate=learn_rate, max_depth=max_depth, subsample=subsample)
    
    predictions = model.fit(X_train, y_train).predict(X_test)
    
    # Extend the return part
    return([learn_rate, max_depth, subsample, accuracy_score(y_test, predictions)])  

results_list = []

# Create the new list to test
subsample_list = [0.4 , 0.6]

for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
    
    	# Extend the for loop
        for subsample in subsample_list:
        	
            # Extend the results to include the new hyperparameter
            results_list.append(gbm_grid_search_extended(learn_rate, max_depth, subsample))
            
# Print results
print(results_list)    
################################################################################################
#Grid Search using Scikit Learn
    '''
    1) choose algorithm to tune
    2) defining which hyperparmeters to tune
    3) defining a range of values for each hyperparameter
    4) setting a cross-validation schem
    5) define a score function 
    6) extra function to use??'''
#GridSearchCV
    #estimator: only one algorithm
    #param_grid: parameter value
    #cv: cross_validation (k-fold)
    #scoring: which score to use to choose the best grid
    #refit:  fits the best hyperparameter to training data
    #n_jobs: allow multiple models to be created at the same time
    #return_train_score: useful for analyzing bias-variane tradeoff, but computation expense
        #will not assist in choosing best model, just for analysis
################################################################################################
# Model #1:
 GridSearchCV(cv=5, error_score='raise-deprecating',
       estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators='warn', n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False),
       fit_params=None, iid='warn', n_jobs=4,
       param_grid={'max_depth': [2, 4, 8, 15], 'max_features': ['auto', 'sqrt']},
       pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
       scoring='roc_auc', verbose=0) 


# Model #2:
 GridSearchCV(cv=10, error_score='raise-deprecating',
       estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=5, p=2,
           weights='uniform'),
       fit_params=None, iid='warn', n_jobs=8,
       param_grid={'n_neighbors': [5, 10, 20], 'algorithm': ['ball_tree', 'brute']},
       pre_dispatch='2*n_jobs', refit=False, return_train_score='warn',
       scoring='accuracy', verbose=0) 

#GridSearchCV with Scikit
# Create a Random Forest Classifier with specified criterion
rf_class = RandomForestClassifier(criterion='entropy')

# Create the parameter grid
param_grid = {'max_depth': [2, 4, 8, 15], 'max_features': ['auto' , 'sqrt']} 

# Create a GridSearchCV object
grid_rf_class = GridSearchCV(
    estimator=rf_class,
    param_grid=param_grid,
    scoring='roc_auc',
    n_jobs=4,
    cv=5,
    refit=True, return_train_score=True)
print(grid_rf_class)
################################################
#Grid Search
    #time_columns
    #params_ columns and the params column
    #test_score columns for each cv fold including the 
        #mean_test_score and 
        #std_test_score columns a 
    #rank_test_score column
    #train_score columns for each cv fold including the 
        #mean_train_score and 
        #std_train_score columns
################################################
# Read the cv_results property into a dataframe & print it out
cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
print(cv_results_df)

# Get and show the column with dictionaries of the hyperparameters used
column = cv_results_df.loc[:, ['params']]
print(column)

# Get and show the row that had the best mean test score
best_row = cv_results_df[cv_results_df['rank_test_score'] == 1 ]
print(best_row)

#Analyzing the best results
    #best_params_ – Which is a dictionary of the parameters that gave the best score.
    #best_score_ – The actual best score.
    #best_index_ – The index of the row in cv_results_ that was the best.
#Using the best results
# See what type of object the best_estimator_ property is
print(type(grid_rf_class.best_estimator_))

# Create an array of predictions directly using the best_estimator_ property
predictions = grid_rf_class.best_estimator_.predict(X_test)

# Take a look to confirm it worked, this should be an array of 1's and 0's
print(predictions[0:5])

# Now create a confusion matrix 
print("Confustion Matrix \n", confusion_matrix(y_test, predictions))

# Get the ROC-AUC score
predictions_proba = grid_rf_class.best_estimator_.predict_proba(X_test)[:,1]
print("ROC-AUC Score \n", roc_auc_score(y_test, predictions_proba))
################################################################################################
#Random Search
    #define an estimator
    #set a cross-validation and scoring function
    #randomly select grid
################################################################################################
#Randomly Sample Hyperparameters
# Create a list of values for the learning_rate hyperparameter
learn_rate_list = np.linspace(0.01,1.5,200)

# Create a list of values for the min_samples_leaf hyperparameter
min_samples_list = list(range(10,41))

# Combination list
combinations_list = [list(x) for x in product(learn_rate_list, min_samples_list)]

# Sample hyperparameter combinations for a random search.
random_combinations_index = np.random.choice(range(0, len(combinations_list)), 250, replace=False)
combinations_random_chosen = [combinations_list[x] for x in random_combinations_index]

# Print the result
print(combinations_random_chosen)



#Randomly Search with Random Forest
# Confirm how hyperparameter combinations & print
number_combs = len(combinations_list)
print(number_combs)

# Sample and visualise combinations
for x in [50, 500, 1500]:
    sample_hyperparameters(x)
    visualize_search()

# Sample all the hyperparameter combinations & visualise
sample_hyperparameters(number_combs)
visualize_search()
################################################
#GridSearchCV vs RandomizedSearchCV
    #Difference: random asks you for how many models to sample from the grid you set.: Decide how many samples to take

################################################
# Create the parameter grid
param_grid = {'learning_rate': np.linspace(0.1,2,150), 'min_samples_leaf': list(range(20,65))} 

# Create a random search object
random_GBM_class = RandomizedSearchCV(
    estimator = GradientBoostingClassifier(),
    param_distributions = param_grid,
    n_iter = 10,
    scoring='accuracy', n_jobs=4, cv = 5, refit=True, return_train_score = True)

# Fit to the training data
random_GBM_class.fit(X_train, y_train)

# Print the values used for both hyperparameters
print(random_GBM_class.cv_results_['param_learning_rate'])
print(random_GBM_class.cv_results_['param_min_samples_leaf'])





# Create the parameter grid
param_grid = {'max_depth':list(range(5,26)),
                'max_features':['auto' , 'sqrt']
} 

# Create a random search object
random_rf_class = RandomizedSearchCV(
    estimator = RandomForestClassifier(80),
    param_distributions = param_grid, 
    n_iter = 5,
    scoring='roc_auc', n_jobs=4, cv = 3, 
    refit=True, return_train_score = True)

# Fit to the training data
random_rf_class.fit(X_train, y_train)

# Print the values used for both hyperparameters
print(random_rf_class.cv_results_['param_max_depth'])
print(random_rf_class.cv_results_['param_max_features'])
################################################
#Compare Grid VS Random Grid
    #tried all combination vs Randomly select a subset of combination within sample space
    #No sampling method vs can select sampling method
    #more computation
    #guaranteed to find the best score
        #if data is large, random search is better
################################################
# Sample grid coordinates
grid_combinations_chosen = combinations_list[0:300]

# Print result
print(grid_combinations_chosen)

# Create a list of sample indexes
sample_indexes = list(range(0,len(combinations_list)))

# Randomly sample 300 indexes
random_indexes = np.random.choice(sample_indexes, 300, replace=False)

# Use indexes to create random sample
random_combinations_chosen = [combinations_list[index] for index in random_indexes]

# Call the function to produce the visualization
visualize_search(grid_combinations_chosen, random_combinations_chosen)
''' grid search will cover a small area completely whilst random search will cover a much larger area but not completely.''''

################################################################################################
#Informed Search vs Uninformed Search
    #Uninformed search: iternation of hyperparameter tuning does not learn from previous iterations
    #=parallelized our work

    #informed sesarch:
        '''1)random search
        2)find promising areas
        3)grid search in the smaller area
        4) continue until optimal score is obtained'''
        #utilize advan of grid and random
        #efficient computation
################################################################################################
# Confirm the size of the combinations_list
print(len(combinations_list))

# Sort the results_df by accuracy and print the top 10 rows
print(results_df.sort_values(by='accuracy', ascending=False).head(10))

# Confirm which hyperparameters were used in this search
print(results_df.columns)

# Call visualize_hyperparameter() with each hyperparameter in turn
visualize_hyperparameter('max_depth')
visualize_hyperparameter('min_samples_leaf')
visualize_hyperparameter('learn_rate')
'''Results clearly seem better when max_depth is below 20. learn_rates smaller than 1 seem to perform well too. There is not a strong trend for min_samples leaf though.'''



# Use the provided function to visualize the first results
visualize_first()

# Create some combinations lists & combine:
max_depth_list = list(range(1,21))
learn_rate_list = np.linspace(0.001,1,50)

# Call the function to visualize the second results
visualize_second()
''' appears to be a bump around max_depths between 5 and 10 as well as learn_rate less than 0.2 so perhaps there is even more room for improvement!'''
################################################
#uninformed method
	#1) Grid Search
	#2) Random Search
#Informed method
	#1) Coarse to Fine
	#2) Bayesian Hyperparameter tuning
	#3) Genetic algorithm
################################################

################################################
#Course to Fine
	‘’’1)Randome Search
	2)find promising areas
	3) Grid search in the smaller area
4)continue until optimal score obtained’’’
################################################
## Visualizing Coarse to Fine
# Confirm the size of the combinations_list
print(len(combinations_list))

# Sort the results_df by accuracy and print the top 10 rows
print(results_df.sort_values(by='accuracy', ascending=False).head(10))

# Confirm which hyperparameters were used in this search
print(results_df.columns)

# Call visualize_hyperparameter() with each hyperparameter in turn
visualize_hyperparameter('max_depth')
visualize_hyperparameter('min_samples_leaf')
visualize_hyperparameter('learn_rate')
#################################
#Bayesian Method
	#Bayes Rule: iteratively update our belief about our outcome in order to get a more accurate result.
	#P(A|B) = posterior probability
	#P(B|A)P(A) / P(B) =prior probability, already known fact
		#P(A|B) = probability that a predisposed person has a disease
		#P(A) = probability of person has a disease (outcome we already know)
		#P(B) = probability of person has pre-disposed (genetically) (outcome we want to know)
		#P(B|A) = probability that a person with the disease are pre-disposed
################################################
p_unhappy = 0.15# Assign probabilities to variables 
p_unhappy_close = 0.35
p_close = 0.07# Probabiliy someone will close

# Probability unhappy person will close
p_close_unhappy = (p_unhappy_close * p_close) / p_unhappy
print(p_close_unhappy)
‘’’ There's a 16.3% chance that a customer, given that they are unhappy, will close their account.
‘’’
################################################
# Bayesian Hyperparameter tuning with Hyperopt
	#1. Set the Domain: Our Grid (with a bit of a twist)
#2. Set the Optimization algorithm (use default TPE)
#3. Objective function to minimize: we will use 1-Accuracy
################################################
# Set up grid with specified hyperparameters
space = {'max_depth': hp.quniform('max_depth', 2, 10, 2),'learning_rate': hp.uniform('learning_rate', 0.001,0.9)}

# Set up objective function
def objective(params):
    params = {'max_depth': int(params['max_depth']),'learning_rate': params['learning_rate']}
    gbm_clf = GradientBoostingClassifier(n_estimators=100, **params) 
    best_score = cross_val_score(gbm_clf, X_train, y_train, scoring='accuracy', cv=2, n_jobs=4).mean()
    loss = 1 - best_score
    return loss

# Run the algorithm
best = fmin(fn=objective,space=space, max_evals=20, rstate=np.random.RandomState(42), algo=tpe.suggest)
print(best) #algorithm 20 evaluations
################################################
#Informed search: genetic algorithm
	#create model
	#pick the best model by scoring function
	#create new models that are similar to the best ones
	#add some randomness to avoid the local optimal
#allow us to learn from previous iteration like bayesian
#TPOT: automated python machine learning tool that optimizes machine learning pipelines using genetic programming
	#Generations: iteration number
	#Population_size: number of models to keep after each iteration
	$offspring_size: number of models to produce in each iteration
	#mutation_ratel: proportion of pipelines to apply randomness 
	#crossover_rate: proportion of pipelines to breed each iteration
	#scoring: function to determine the best model
	#cv: cross-validation strategy to use
################################################
## Genetic Hyperparameter Tuning with TPOT
# Assign the values outlined to the inputs
number_generations = 3
population_size = 4
offspring_size = 3
scoring_function = 'accuracy'

# Create the tpot classifier
tpot_clf = TPOTClassifier(generations=number_generations, population_size=population_size,
                          offspring_size=offspring_size, scoring=scoring_function,
                          verbosity=2, random_state=2, cv=2)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))


## Analysing TPOT's stability
‘’’try to run with different number of random_state (42,122,99). 
First time, chose Decision Tree as a Best pipeline, Second time chose KNeighborsClassifier and Third time was RandomForestClassifier. If we have low generations, population size and offspring, TPOT is not stable’’’
