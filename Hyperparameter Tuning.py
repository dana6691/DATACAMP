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

################################################
#Grid Search
################################################

################################################
#Random Search
################################################

################################################
#Informed Search
################################################
