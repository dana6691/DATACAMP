################################################
#Feature selection vs Feature extraction
    #reduce variables vs make  1 new variable using 3 variables(combine)
#Why do we reduce dimensionality?
    '''1)dataset is less complex
        2)less disk space
        3)less computation time
        4)lower chance of model overfitting'''
################################################
sns.pairplot(ansur_df_1, hue='Gender', diag_kind='hist')# Create a pairplot
plt.show()

reduced_df = ansur_df_1.drop('body_height', axis=1)# Remove one of the abundant(same) features
sns.pairplot(r`educed_df, hue='Gender')# Create a pairplot 
################################################
#t-SNE for high-dimensional data: doesn't work with non-numeric data
    #helps visually explore the patterns in a high dimensional dataset
    #high learning rate= algorithm to be more adventurous, low learning rate = more conservative
################################################
non_numeric = ['Branch', 'Gender', 'Component']# Non-numerical columns
df_numeric = df.drop(non_numeric, axis=1)# Drop the non-numerical
m = TSNE(learning_rate=50)# Create a t-SNE model
tsne_features = m.fit_transform(df_numeric)# Fit and transform the t-SNE model on the numeric dataset
print(tsne_features.shape)

sns.scatterplot(x="x", y="y", hue='Component', data=df)
plt.show()
sns.scatterplot(x="x", y="y", hue='Gender', data=df) #by gender
plt.show()
################################################################################################
#Feature Selection I
#high-dimenstional datset has problem of overfitting.
# How to detecting low quality features and remove them
    #give more observations on each city = generalization, remove overfitting
    #use SVC(support vector machine classifier)
    #accuracy test on testing set = model able to assign 82.6% of our unseen houses prices to the correct city
    #accuracy test on training set: If it is too higher than the testing accuracy, then overfitted
        #solution:adding features, increase observation
################################################
from sklearn.model_selection import train_test_split
y = ansur_df['Gender']
X = ansur_df.drop('Gender', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print("{} rows in test set vs. {} in training set. {} Features.".format(X_test.shape[0], X_train.shape[0], X_test.shape[1]))

#Fitting and testing the model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svc = SVC()#Support Vector Classification
svc.fit(X_train, y_train)# Fit the model
accuracy_train = accuracy_score(y_train, svc.predict(X_train))# Calculate accuracy scores 
accuracy_test = accuracy_score(y_test, svc.predict(X_test))
print("{0:.1%} accuracy on test set vs. {1:.1%} on training set".format(accuracy_test, accuracy_train))

#Accuracy after dimensionality reduction
X = ansur_df[['neckcircumferencebase']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
svc = SVC()
svc.fit(X_train, y_train)
accuracy_train = accuracy_score(y_train, svc.predict(X_train))
accuracy_test = accuracy_score(y_test, svc.predict(X_test))
print("{0:.1%} accuracy on test set vs. {1:.1%} on training set".format(accuracy_test, accuracy_train))
################################################
#Missing values or little variance
    #use minimal variance threshold 
    #normalize the variance before we do feature selection = each value / mean(value) and fit
        #then variance will be lower
    #drop features: if contains lot of missing value
################################################
#Finding a good variance threshold
normalized_df = head_df / np.mean(head_df) #normalized data
normalized_df.boxplot()
plt.show()
print(normalized_df.var()) #variance of normalized data
    #lowest two variance should be removed

#successfully removed the 2 low-variance features
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.001)# Create a VarianceThreshold feature selector
sel.fit(head_df / head_df.mean())# Fit the selector to normalized head_df
mask = sel.get_support()# Create a boolean mask
reduced_df = head_df.loc[:, mask]# Apply the mask to create a reduced dataframe

#Removing features with many missing values
df.isna().sum()#counting missing values
df.isna().sum()/len(df) #ratio of missing value
mask = df.isna().sum/len(df) <0.3
print(mask) #True or False
reduced_df = df.loc[:,mask] # Create a reduced dataset 
reduced_df.head()
################################################
#Pairwise correlation
    #measure strength of the correlation
################################################
corr = ansur_df.corr()# Create the correlation matrix
sns.heatmap(corr,  cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f")# Draw the heatmap
plt.show()
#Visualizing the correlation matrix
corr = ansur_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool)) #upper triangle
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f")
plt.show()
################################################
#Removing highly correlated features
    #High-correlation does not imply causation.
################################################
corr_matrix = ansur_df.corr().abs()# Calculate the correlation matrix
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))# Create a True/False mask 
tri_df = corr_matrix.mask(mask)

to_drop = [c for c in tri_df.columns if any(tri_df[c] >  0.95)]# List column names of highly correlated features (r > 0.95)
reduced_df = ansur_df.drop(to_drop, axis=1)# Drop the features
print("The reduced dataframe has {} columns.".format(reduced_df.shape[1]))
################################################################################################
#Feature Selection II on classification algorithm
    #select based on model performance
################################################
X_train_std = scaler.fit_transform(X_train)# Fit the scaler on the training 
lr.fit(X_train_std, y_train) 
X_test_std = scaler.transform(X_test)# Scale the test features
y_pred = lr.predict(X_test_std)# Predict on scaled test set

print("{0:.1%} accuracy on test set.".format(accuracy_score(y_test, y_pred))) #accuracy metrics and feature coefficients
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))

#Automatic Recursive Feature Elimination
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=3, verbose=1)#RFE with a LogisticRegression estimator and 3 features to select
rfe.fit(X_train, y_train)
print(dict(zip(X.columns, rfe.ranking_)))#features and their ranking (high = dropped early on)
print(X.columns[rfe.support_])#features that are not eliminated
acc = accuracy_score(y_test, rfe.predict(X_test))# Calculates the test set accuracy
print("{0:.1%} accuracy on test set.".format(acc)) 
################################################
#Feature Selection II on classification algorithm
    #Tree-based selection
        #Random forest classifier: ensemble model, different, random ,subset of features to number of decision trees
            #good for accurate and avoid overfitting
            #can do feature importance test, used for features selection, good part, no need scaled, due to its sum is equal to 1
################################################
#Building a random forest model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
rf = RandomForestClassifier(random_state=0)# Fit the random forest model
rf.fit(X_train, y_train)
acc = accuracy_score(y_test, rf.predict(X_test))# Calculate the test set accuracy
print(dict(zip(X.columns, rf.feature_importances_.round(2))))# Print the importances per feature
print("{0:.1%} accuracy on test set.".format(acc)) 

#Random forest for feature selection
mask = rf.feature_importances_ > 0.15# Create a mask for features importances above the threshold
reduced_X = X.loc[:,mask]
print(reduced_X.columns)

#Recursive Feature Elimination with random forests
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=2, verbose=1)
rfe.fit(X_train, y_train) #fit
mask = rfe.support_ >0.15 #create mask with rfe
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=2, step=2, verbose=1) #remove 2 features on each step
rfe.fit(X_train, y_train) #fit to training
mask = rfe.support_# Create a mask
################################################
#Feature Selection II on regression
    #linear regression: minimized loss function, MSE(mean square error) 
    #Regularized linear regression: simpler the model and make model accurate
################################################
#Creating a LASSO regressor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) #fit scaler and transform
X_train_std = scaler.fit_transform(X_train)
la = Lasso()# Create the Lasso model
la.fit(X_train_std,y_train) #fit to standardized 

#Lasso model results
X_test_std = scaler.transform(X_test)# Transform the test set
r_squared = la.score(X_test_std, y_test)# Calculate (R squared) on X_test_std
print("The model can predict {0:.1%} of the variance in the test set.".format(r_squared))

zero_coef = la.coef_ == 0# Create a list that has True values when coefficients equal 0
n_ignored = sum(zero_coef)# Calculate how many features have a zero coefficient
print("The model has ignored {} out of {} features.".format(n_ignored, len(la.coef_)))
    '''We can predict almost 85% of the variance in the BMI value using just 9 out of 91 of the features. 
       The R^2 could be higher though'''

#Adjusting the regularization strength
la = Lasso(alpha=0.1, random_state=0)# Find the highest alpha value with R-squared above 98% #alpah can be 0.1,0.01,0.5,1
la.fit(X_train_std, y_train)# Fits the model
r_squared = la.score(X_test_std, y_test) #calculates performance stats
n_ignored_features = sum(la.coef_ == 0)

print("The model can predict {0:.1%} of the variance in the test set.".format(r_squared))
print("{} out of {} features were ignored.".format(n_ignored_features, len(la.coef_)))
    '''more appropriate regularization strength we can predict 98% of the variance in the BMI value while ignoring 2/3 of the features'''
################################################
#Combining feature selectors(multiple)
################################################
#Creating LassoCV
from sklearn.linear_model import LassoCV
lcv = LassoCV()# Create and fit the LassoCV model
lcv.fit(X_train,y_train)
print('Optimal alpha = {0:.3f}'.format(lcv.alpha_))
r_squared = lcv.score(X_test,y_test)# Calculate R squared 
print('The model explains {0:.1%} of the test set variance'.format(r_squared))

lcv_mask = lcv.coef_!=0# Create a mask for coefficients not equal to zero
print('{} features out of {} selected'.format(sum(lcv_mask), len(lcv_mask)))
    '''LassoCV() model explains 88.2% of the test set variance 26 features out of 32 selected'''

#Creating ensemble model(GradientBoostingRegressor)
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor
rfe_gb = RFE(estimator=GradientBoostingRegressor(), 
             n_features_to_select=10, step=3, verbose=1)# Select 10 features with RFE , drop 3 features on each step
rfe_gb.fit(X_train, y_train)
r_squared = rfe_gb.score(X_test,y_test)#calculate R^2
print('The model can explain {0:.1%} of the variance in the test set'.format(r_squared))
    '''The model can explain 85.6% of the variance in the test set'''
rf_mask = rfe_rf.support_# Assign the support array to gb_mask

#Creating ensemble model(RandomForest)
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
rfe_rf = RFE(estimator=RandomForestRegressor(), 
             n_features_to_select=10, step=3, verbose=1)
rfe_rf.fit(X_train, y_train)
r_squared = rfe_rf.score(X_test, y_test)# Calculate the R squared
print('The model can explain {0:.1%} of the variance in the test set'.format(r_squared))
rf_mask = rfe_rf.support_# Assign the support array to gb_mask
    '''The model can explain 84.0% of the variance in the test set'''

# Sum the votes of the three models
votes = np.sum([lcv_mask, rf_mask, gb_mask], axis=0)
meta_mask = votes >= 3# Create a mask for features selected by all 3 models
X_reduced = X.loc[:, meta_mask]# Apply the dimensionality reduction on X
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=0)
lm.fit(scaler.fit_transform(X_train), y_train)# Plug the reduced dataset
r_squared = lm.score(scaler.transform(X_test), y_test)
print('The model can explain {0:.1%} of the variance in the test set using {1:} features.'.format(r_squared, len(lm.coef_)))
    '''The model can explain 86.8% of the variance in the test set using 7 features.'''
################################################################################################
#Feature Extraction: create new features with combination of original features
    #PCA: Scaled the data to make it easier to compare
#After standardizing the lower and upper arm lengths from the ANSUR dataset we've added two perpendicular vectors that are aligned with the main directions of variance. We can describe each point in the dataset as a combination of these two vectors multiplied with a value each. These values are then called principal components.


    '''more moremo re'''
################################################
#Manual feature extraction I
# Calculate the price from the quantity sold and revenue
sales_df['price'] = sales_df['revenue'] / sales_df['quantity']

# Drop the quantity and revenue features
reduced_df = sales_df.drop(['revenue','quantity'], axis=1)

print(reduced_df.head())

#Manual feature extraction II
# Calculate the mean height
height_df['height'] = height_df[['height_1', 'height_2', 'height_3']].mean(axis=1)

# Drop the 3 original height features
reduced_df = height_df.drop(['height_1', 'height_2', 'height_3'], axis=1)

print(reduced_df.head())
################################################
#Feature Extraction
#Principal Component Analysis
################################################
#Manual feature extraction I
sales_df['price'] = sales_df['revenue'] / sales_df['quantity']# Calculate the price from the quantity sold and revenue
reduced_df = sales_df.drop(['revenue','quantity'], axis=1)# Drop the quantity and revenue features
print(reduced_df.head())

#Manual feature extraction II
height_df['height'] = height_df[['height_1','height_2','height_3']].mean(axis=1)# Calculate the mean height
reduced_df = height_df.drop(['height_1','height_2','height_3'], axis=1)# Drop the 3 original height features
print(reduced_df.head())

#Calculating Principal Components
sns.pairplot(ansur_df) #pairplot of regular generalized data
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()# Create the scaler
ansur_std = scaler.fit_transform(ansur_df)
pca = PCA()# Create the PCA instance and fit and transform
pc = pca.fit_transform(ansur_std)
pc_df = pd.DataFrame(pc, columns=['PC 1', 'PC 2', 'PC 3', 'PC 4'])
sns.pairplot(pc_df)# Create a pairplot of the principal component dataframe
plt.show()

#PCA on larger datset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()# Scale the data
ansur_std = scaler.fit_transform(ansur_df)
pca = PCA()# Apply PCA
pca.fit(ansur_std)

#PCA explained variance
print(pca.explained_variance_ratio_) #variance ratio per component
    '''variance is explained by the 4th principal component? 3.77%'''
print(pca.explained_variance_ratio_.cumsum()) #cumulative sum of the explained variance ratio
    '''no more than 4 principal components we can explain more than 90% of the variance in the 13 feature dataset.'''
################################################
#PCA Application
    #PCA needs to decide how much of explained variance are willing to sacrifice. 
    #Downside: remaining components are hard to interpret
################################################
pipe = Pipeline([('scaler', StandardScaler()),
        		 ('reducer', PCA(n_components=2))])
pipe.fit(poke_df)# Fit it to the dataset and extract the component vectors
vectors = pipe.steps[1][1].components_.round(2)
print('PC 1 effects = ' + str(dict(zip(poke_df.columns, vectors[0]))))# Print feature effects
print('PC 2 effects = ' + str(dict(zip(poke_df.columns, vectors[1]))))
'''
    All features have a similar positive effect. PC 1 can be interpreted as a measure of overall quality (high stats).
    Defense has a strong positive effect on the second component and speed a strong negative one. This component quantifies an agility vs. armor & protection trade-off.
'''

#PCA for feature exploration
pipe = Pipeline([('scaler', StandardScaler()),
                 ('reducer', PCA(n_components=2))])
pc = pipe.fit_transform(poke_df)# Fit the pipeline to poke_df and transform the data
print(pc)

poke_cat_df['PC 1'] = pc[:, 0]# Add the 2 components to poke_cat_df
poke_cat_df['PC 2'] = pc[:, 1]

sns.scatterplot(data=poke_cat_df, 
                x='PC 1', y='PC 2', hue='Type')# Use the Type feature to color the PC 1 vs PC 2 scatterplot
plt.show()
sns.scatterplot(data=poke_cat_df, 
                x='PC 1', y='PC 2', hue='Legendary')# Use the Legendary feature to color the PC 1 vs PC 2 scatterplot
plt.show()

#PCA in a model pipeline
pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reducer', PCA(n_components=2)),
        ('classifier', RandomForestClassifier(random_state=0))]) #2 componentas extracted

pipe.fit(X_train,y_train)# Fit the pipeline to the training data
print(pipe.steps[1][1].explained_variance_ratio_)# Prints the explained variance ratio

accuracy = pipe.score(X_test,y_test)# Score the accuracy on the test set
print('{0:.1%} test set accuracy'.format(accuracy))
'''
    repeated the process with n_components =3, doesn't change accuracy
'''
################################################
#Principal Component Selection
    '''1)setting an explained variance threshold
        2)plot(principal component index vs Explained variance ratio)
        3)From above, use elblow method'''
    #1. pca.fit(X) --> pca.transform(X)
    #2. pca.fit_transform(X)  
    #3. pca.inverse_transform(pc) : imaging compression
################################################
#Selecting the proportion of variance to keep
################################################ 
# Pipe a scaler to PCA selecting 80% of the variance
pipe = Pipeline([('scaler', StandardScaler()),
        		 ('reducer', PCA(n_components=0.8))])
ipe.fit(ansur_df)# Fit the pipe to the data
print('{} components selected'.format(len(pipe.steps[1][1].components_)))

# Pipe a scaler to PCA selecting 90% of the variance
pipe = Pipeline([('scaler', StandardScaler()),
        		 ('reducer', PCA(n_components=0.9))])
pipe.fit(ansur_df)# Fit the pipe to the data
print('{} components selected'.format(len(pipe.steps[1][1].components_)))
################################################
#Choosing the number of components
################################################ 
pipe = Pipeline([('scaler', StandardScaler()),
        		 ('reducer', PCA(n_components=10))])#pca selecting 10 components
pipe.fit(ansur_df)# Fit
plt.plot(pipe.steps[1][1].explained_variance_ratio_)# Plot the explained variance ratio
plt.xlabel('Principal component index')
plt.ylabel('Explained variance ratio')
plt.show()
''' 'elbow' in the plot is at 3 components (the 3rd component has index 2).'''
################################################
#PCA for image compression
################################################
plot_digits(X_test)# Plot the MNIST sample data
pc = pipe.transform(X_test)# Transform the input data to principal components
print("X_test has {} features".format(X_test.shape[1]))# Prints the number of features per dataset
print("pc has {} features".format(pc.shape[1]))

X_rebuilt = pipe.inverse_transform(pc)# Inverse transform the components to original feature space
print("X_rebuilt has {} features".format(X_rebuilt.shape[1]))# Prints the number of features

plot_digits(X_rebuilt)# Plot the reconstructed data
    '''
    You've reduced the size of the data 10 fold but were able to reconstruct images with reasonable quality.
    '''
