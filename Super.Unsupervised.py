from sklearn import datasets
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
iris = datasets.load_iris()
type(iris)

print(iris.keys())

print(type(iris.data), type(iris.target))
print(iris.data.shape)

#EDA
X=iris.data
y=iris.target
df = pd.DataFrame(X,columns=iris.feature_names)
print(df.head())
print(df.describe())

#bar graph(Yes/No)
plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

#fit and predict the model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'],iris['target']) #fit
print(iris['data'].shape)
print(iris['target'].shape)

prediction = knn.predict(X_new)
X_new.shape
print('Prediction {}'.format(prediction))
################################################
#fit/predict kkn
################################################
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X,y)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))
################################################
#measuring performance
################################################
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_text = train_test_split(X,y,test_size=0.3,random_state=21, stratify=y)

################################################
# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)

# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
####################################################
#Model Evaluation
####################################################
# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Print the accuracy
print(knn.score(X_test, y_test))
###########################################
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
#################################################
#Regression
    #choose the line that minimize the error function
    #vertical distance between data and the line = residual
    #Minimize the sum of squares of residuals = Ordinary least square(OLS)
#################################################
boston=pd.read_csv('boston.csv')
print(boston.head())
X=boston.drop('MEDV',axis=1).values
y=boston["MEDV"].values
X_rooms=X[:,5]
type(X_rooms),type(y)
y=y.reshape(-1,1)
X_rooms=X_rooms.reshape(-1,1)
#plot
plt.scatter(X_rooms,y)
plt.ylabel("Value of house/1000 ($)")
plt.xlabel('Number of rooms')
plt.show()
import numpy as np
from sklearn import linear_model
reg=linear_model.LinearRegression()
reg.fit(X_rooms,y)
#################################################
#Regression2 (Gapminder data)
#################################################
# Import numpy and pandas
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X_train), max(X_train)).reshape(-1,1)

# Fit the regressor to the training data
reg_all.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))
################################################
#cross-validation
    #split the dataset into 5(4-training, 1-test)
    #fit on training, predict on test set, compute the matrix 
    #repeated 4 times
        #ex) 5 fold, 10 fold. k fold Cross validation(CV)
#################################################
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
reg = linear_model.LinearRegression()
cv_results=cross_val_score(reg,X,y, cv=5)
print(cv_results) #Print the 5-fold cross-validation scores
print("Average 5-Fold CV Score: {}".format(np.mean(cv_results)))
#################################################
#Regualarized Regression
    #Large coefficients lead to overfitting
    #Penalizeing large coefficients = regularization
    #Ridge Regression = sum of squares of residuals(OLS) * alpha
        #Hyperparameter tuning
        #Control model complexity
            # if alpha is too high: Underfitting
            # alpha = 0 (OLS) : overfitting
    #Lasso Regression = sum of absolute residuals * alpha
        #can be use for selecting important features of a dataset
        #shrink less important features's coefficients to exactly 0
#################################################
#Ridge Regression
from sklearn.model_selection import cross_val_score

alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

ridge = Ridge(normalize=True)

for alpha in alpha_space:
    ridge.alpha = alpha
    ridge_cv_scores = cross_val_score(ridge,X,y, cv=10)# Perform 10-fold CV
    ridge_scores.append(np.mean(ridge_cv_scores))# Append the mean of ridge_cv_scores
    ridge_scores_std.append(np.std(ridge_cv_scores))# Append the std of ridge_cv_scores

# Display the plot
display_plot(ridge_scores, ridge_scores_std)

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()
#Lasso Regression
from sklearn.linear_model import Lasso
names=boston.drop('MEDV',axis=1).columns
lasso=Lasso(alpha=0.1,normalize=True)
lasso_coef = lasso.fit(X,y).coef_ # Compute and print the coefficients
print(lasso_coef)
plt.plot(range(len(names)),lasso_coef)
plt.xticks(range(len(names)),names,rotation=60)
plt.ylabel('Coefficients')
plt.show()
##################################################################################################
#Tunning model: access model's performance, optimize your classification and regression model using 
    #Confusion Matrix(TruePositive, TrueNegative, FalsePositive, FalseNegative)
        #Precision = TP/TP+FP
        #Recall: TP/TP+FN
#################################################
#Tuning classification
# Import necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)
knn = KNeighborsClassifier(n_neighbors=6) #k-NN classifier
knn.fit(X_train,y_train) # fit
y_pred = knn.predict(X_test) # Predict 

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#################################################
#LogisticRegression: binary
    #outputs probabilities p given one features
    # p > 0.5, label 1
    # p <0.5, label 0
    #probability threshold = 0.5
#ROC curve: 
#################################################
# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train,y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#################################################
#AUC
#################################################
# Import necessary modules
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))
#################################################
#Hyperparameter Tuning
    #Linear Regresion:choosing parameters
    #Ridge/Lasso regression: Choosing alpha
    #k-Nearest Neighbors: choosing n_neighbors
    #Parameters like alpha and k:Hypoerparameters
#Hyperparameter: parameters should be specified before fitting a model, parameter cannot explicitly learned by fitting the model
    #how to find the correct hyperparameter
#################################################
#################################################
#Grid Search: cross-validation: try all combination
#################################################
from sklearn.model_selection import GridSearchCV
param_grid={'n_neighbors':np.arange(1,50)}
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn,param_grid, cv=5) #number of fold(cross-validation)
knn_cv.fit(X,y)
# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(knn_cv.best_params_)) 
print("Best score is {}".format(knn_cv.best_score_))
#################################################
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
c_space = np.logspace(-5, 8, 15)# Create the hyperparameter grid
param_grid = {'C':c_space, 'penalty':['l1', 'l2']}

logreg = LogisticRegression()#logistic regression classifier
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)
logreg_cv = GridSearchCV(logreg,param_grid,cv=5)#GridSearchCV
logreg_cv.fit(X_train,y_train)# Fit
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))
#################################################
#RandomizedSearchCV - Hyperparameter tuning
#################################################
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}
tree = DecisionTreeClassifier()# Decision Tree classifier
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
tree_cv.fit(X,y) #fit
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
#################################################
#Hold-out set: how to perform our modle in unseen data
#################################################
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=42)
l1_space = np.linspace(0, 1, 30) #hyperparameter grid
param_grid = {'l1_ratio': l1_space}
elastic_net = ElasticNet()#ElasticNet regressor
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)#GridSearchCV

gm_cv.fit(X_train,y_train) #fit
y_pred = gm_cv.predict(X_test) # Predict on the test set and compute metrics
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
##################################################################################################
#Preprocessing and pipelines
    #categorical variables --> create 'dummy variables'
    #scikit-learn:OneHoyEncoder()
    #pandas:get_dummies()
##################################################################################################
import pandas as pd
df = pd.read_csv('gapminder.csv')
df.boxplot('life','Region', rot=60) 
plt.show()

df_region = pd.get_dummies(df)# Create dummy variable
print(df_region.columns)

df_region =  pd.get_dummies(df,drop_first=True)# Create dummy variables with drop_first
print(df_region.columns)

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
ridge = Ridge(alpha=0.5, normalize=True)#ridge regressor
ridge_cv = cross_val_score(ridge,X,y,cv=5)#5-fold cross-validation
print(ridge_cv)
#################################################
#Missing value
    #encoded by zeros
    #question marks
    #negative ones
#1)droppping missing data --> but we need robust data
#2)imputing missing data --> guess what the missing values would be (etc. mean)
#3)imputing within a pipeline --> 
#################################################
df[df == '?'] = np.nan # Convert '?' to NaN
print(df.isnull().sum()) # Print the number of NaN
print("Shape of Original DataFrame: {}".format(df.shape)) #shape of original DataFrame
df = df.dropna()# Drop missing values
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))

#################################################
from sklearn.preprocessing import Imputer # Import the Imputer module
from sklearn.svm import SVC
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0) # Setup the Imputation transformer
clf = SVC() # Instantiate the SVC classifier: clf
steps = [('imputation', imp),
        ('SVM', clf)] # Setup the pipeline with the required steps: steps
#################################################
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())] # Setup the pipeline steps:
pipeline = Pipeline(steps)# Create the pipeline: pipeline
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred)) # Compute metrics
#################################################
#Normalizing(Centering and scaling):check the range of each feature by df.describe()
''' 1)standardization: subtract the mean and divide by variance
    2)sutract minimum and divided by the range
    3)minimum zero and maximum one
    4)#all features are centered around zero and have variance=1
    5)normalize to data range from -1 to 1'''
#################################################
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())] # Setup the pipeline
pipeline = Pipeline(steps)# Create the pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn_scaled = pipeline.fit(X_train, y_train)# Fit 
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train) #k-NN classifier to the unscaled data

print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))
#################################################
#pipline for classification
#################################################
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())] # Setup the pipeline steps

pipeline = Pipeline(steps) # Create the pipeline
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}# Specify the hyperparameter space

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=21)
cv = GridSearchCV(pipeline,param_grid=parameters,cv=3) #GridSearchCV
cv.fit(X_train,y_train)
y_pred = cv.predict(X_test)

print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))
##################################################################################################
#Unsupervised Learning
#K-means clustering
##################################################################################################
# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels =model.predict(new_points)

# Print cluster labels of new_points
print(labels)
#################################################
#scatter plot
import matplotlib.pyplot as plt 
xs = samples[:,0]
ys = samples[:,2]
plt.scatter(xs,ys,c=labels)
plt.show()
###################################################
# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs,ys,c=labels,alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x,centroids_y,marker='D',s=50)
plt.show()

###################################################
#k-means inertia graph
###################################################

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
###################################################
#cross_tab
###################################################
# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)
###################################################
#piedmont wines dataset
#data preprocessing types
    #standardized = mean 0, var 1
    #MaxAbsScaler
    #Normalizer
###################################################
# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,kmeans)

#StandardScaler VS KMeans
 #fit()/transform()
 #fit()/predict()
###################################################
#standardization and clustering pipeline 
#cross-tabulation
###################################################
# Import pandas
import pandas as pd

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels': labels, 'species': species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['species'])

# Display ct
print(ct)
##############################################
#Yahoo stock data- normalizer
    #StandardScaler() standardizes features (such as the features of the fish data from the previous exercise) by removing the mean and scaling to unit variance, 
    #Normalizer() rescales each sample - here, each company's stock price - independently of the other.
#################
# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer,kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)
#####################################################
# Import pandas
import pandas as pd

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))
#####################################################
#Hierarchical clustering - visualization
#####################################################
#Eurovision score data
from sklearn.preprocessing import normalize
normalized_movements = normalize(movements)# Normalize 
mergings = linkage(normalized_movements,method='complete')# Calculate the linkage
# Plot the dendrogram
dendrogram(mergings,
           labels=companies,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()
#####################################################
#Hierarchical clustering - cluster labels at intermediate stage can be recovered
    #for use in cross-tabulations
    #Heights on dendrogram = distance between mering clusters
        #calculate by "linkage method": 
            #'complete':distance between the furthest points of the clusters
            #'single': distance between clusters is the distance between the closest points of the clusters
#####################################################
# Perform the necessary imports
import pandas as pd
from scipy.cluster.hierarchy import fcluster

labels = fcluster(mergings,6, criterion='distance')# Use fcluster to extract labels
df = pd.DataFrame({'labels': labels, 'varieties': varieties})# Create a DataFrame 
ct = pd.crosstab(df['labels'] ,df['varieties'])
print(ct)
#####################################################
#t-SNE for 2-dimensional maps
    #t-SNE(t-distributed stochastic neighbor embedding)
    #maps samples to 2D space(or 3D)
    #fit and transform at the same time 'fit_transform'
#####################################################
from sklearn.manifold import TSNE
model = TSNE(learning_rate=200) # Create a TSNE instance
tsne_features = model.fit_transform(samples)
xs = tsne_features[:,0]
ys = tsne_features[:,1]
plt.scatter(xs,ys,c=variety_numbers) #scatter
plt.show()
#####################################################
from sklearn.manifold import TSNE
model = TSNE(learning_rate=50) # Create a TSNE instance
tsne_features = model.fit_transform(normalized_movements) #fit_transform to normalized_movements
xs = tsne_features[:,0]
ys = tsne_features[:,1]
plt.scatter(xs,ys,alpha=0.5) #scatter
for x, y, company in zip(xs, ys, companies): # Annotate the points
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()
#####################################################
#Dimensional Reduction: Remove less-informative"noise"features
    #(PCA):principal Component Analysis
    '''1)decorrelation: doesn't change dimension of data at all
       2)reduced dimension'''
        #roate the data samples to be aligned with axes = decorrelation
            #* PCA features are not linearly correlated --> measure by Pearson correlation test
        #shifts data samples so they have mean 0
        #fit/transform pattern
            #Row of transformed: samples
            #column: PCA features
#principal components: the directions along which the the data varies
#####################################################
#correlation
#####################################################
import matplotlib.pyplot as plt
from scipy.stats import pearsonr 

width = grains[:,0]
length = grains[:,1]
plt.scatter(width, length)
plt.axis('equal')
plt.show()
correlation, pvalue = pearsonr(width,length)# Calculate the Pearson correlation
print(correlation)
#####################################################
#Decorrelation (PCA)
#####################################################
from sklearn.decomposition import PCA
model = PCA()# Create PCA instance
pca_features = model.fit_transform(grains) # Apply the fit_transform method of model to grains
xs = pca_features[:,0]
ys = pca_features[:,1]

plt.scatter(xs, ys)
plt.axis('equal')
plt.show()
correlation, pvalue = pearsonr(xs, ys) # Calculate the Pearson correlation 
print(correlation)
#####################################################
#Intrinsic Dimension: how many features needed to approximate the dataset
    #detected with PCA
    #scatter plot is only work if samles are 2 or 3 features, So PCA needed
    #PCA shift the datset and Intrictic Dimension find 
#####################################################
plt.scatter(grains[:,0], grains[:,1])#untransformed points
model = PCA()# Create a PCA instance: model
model.fit(grains)
mean = model.mean_# Get the mean 
first_pc = model.components_[0,:]# Get the first principal component

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)
plt.axis('equal')# Keep axes on same scale
plt.show()
#####################################################
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
scaler = StandardScaler()# Create scaler: scaler
pca = PCA()# Create a PCA instance: pca
pipeline = make_pipeline(scaler,pca)# Create pipeline: pipeline
pipeline.fit(samples)# Fit the pipeline to 'samples'

features = range(pca.n_components_)# Plot the explained variances
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
#####################################################
#Dimension reduction(PCA)
#####################################################
# Import PCA
from sklearn.decomposition import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)
#####################################################
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer() 

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names() 

# Print words
print(words)
#####################################################
#Dimension reduction
    #less disk space
    #less complex
    #less computation time
    #lower chance of model overfitting

#feature selection: simply removed features
#feature extraction: make new feature with the combination of features
#####################################################
# Create a pairplot and color the points using the 'Gender' feature
sns.pairplot(ansur_df_1, hue='Gender', diag_kind='hist')
plt.show()
##########################################################################################################
#Missing values or little variance
#####################################################
# Create the boxplot
head_df.boxplot()

plt.show()
#####################################################
#linear classifier(Supervised Learning)###################################
#Logistic Regression
#####################################################
from sklearn.linear_model import LogisticRegression
import sklearn.datasets
wine= sklearn.datasets.load_wine()
lr=LogisticRegression()
lr.fit(wine.data,wine.target)
lr.score(wine.data,wine.target)

lr.predict_proba(wine.data[:1])
#####################################################
#linear classifier#
#Linear SVC(support vector)
#####################################################
import sklearn.datasets
wine= sklearn.datasets.load_wine()
from sklearn.svm import LinearSVC
svm = LinearSVC()
svm.fit(wine.data,wine.target)
svm.score(wine.data,wine.target)
#####################################################
#LogisticRegression, LinearSVC, SVC, KNeighborsClassifier
#####################################################
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# Define the classifiers
classifiers = [LogisticRegression(),LinearSVC(),SVC(),KNeighborsClassifier()]

# Fit the classifiers
for c in classifiers:
    c.fit(X,y)

# Plot the classifiers
plot_4_classifiers(X, y, classifiers)
plt.show()
#####################################################
#Sentimental Analysis
# Instantiate logistic regression and train
lr = LogisticRegression()
lr.fit(X,y)

# Predict sentiment for a glowing review
review1 = "LOVED IT! This movie was amazing. Top 10 this year."
review1_features = get_features(review1)
print("Review:", review1)
print("Probability of positive review:", lr.predict_proba(review1_features)[0,1])

# Predict sentiment for a poor review
review2 = "Total junk! I'll never watch a film by that director again, no matter how good the reviews."
review2_features = get_features(review2)
print("Review:", review2)
print("Probability of positive review:", lr.predict_proba(review2_features)[0,1])
#####################################################
#linear decision boundaries
#####################################################

#####################################################
#nonlinear decision boundaries - LogisticRegression
#####################################################

#####################################################
#Classification And Regression Tree(CART)
    #sequence of if-else about individual features
    #Goal: infer class labels
    #Able to capture non-linear relationship between features and labels
    #Don't require feature scaling(ex)standardization)
        #ex)Cancer(Y/N)
#####################################################
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
accuracy_score(y_test,y_pred)# Compute test set accuracy 
#####################################################
#Logistic Regreession VS Classification Tree
# Import LogisticRegression from sklearn.linear_model
from sklearn.linear_model import  LogisticRegression

# Instatiate logreg
logreg = LogisticRegression(random_state=1)

# Fit logreg to the training set
logreg.fit(X_train, y_train)

# Define a list called clfs containing the two classifiers logreg and dt
clfs = [logreg, dt]

# Review the decision regions of the two classifiers
plot_labeled_decision_regions(X_test, y_test, clfs)
#####################################################
#Using entropy as a criterion
# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# Instantiate dt_entropy, set 'entropy' as the information criterion
dt_entropy = DecisionTreeClassifier(max_depth=8,  criterion='entropy', random_state=1)

# Fit dt_entropy to the training set
dt_entropy.fit(X_train, y_train)
#####################################################
#Entropy vs Gini index
# Import accuracy_score from sklearn.metrics
from sklearn.metrics import accuracy_score

# Use dt_entropy to predict test set labels
y_pred= dt_entropy.predict(X_test)

# Evaluate accuracy_entropy
accuracy_entropy = accuracy_score(y_test, y_pred)

# Print accuracy_entropy
print('Accuracy achieved by using entropy: ', accuracy_entropy)

# Print accuracy_gini
print('Accuracy achieved by using the gini index: ', accuracy_gini)
#####################################################
#Regression Tree
# Import DecisionTreeRegressor from sklearn.tree
from sklearn.tree import DecisionTreeRegressor

# Instantiate dt
dt = DecisionTreeRegressor(max_depth=8,
             min_samples_leaf=0.13,
            random_state=3)

# Fit dt to the training set
dt.fit(X_train, y_train)
#####################################################
#MSE
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute y_pred
y_pred = dt.predict(X_test)

# Compute mse_dt
mse_dt = MSE(y_test, y_pred)

# Compute rmse_dt
rmse_dt = mse_dt**(1/2)

# Print rmse_dt
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))
#####################################################
#Linear regression vs regression tree
# Predict test set labels 
y_pred_lr = lr.predict(X_test)

# Compute mse_lr
mse_lr = MSE (y_pred_lr, y_test)

# Compute rmse_lr
rmse_lr = mse_lr**(1/2)

# Print rmse_lr
print('Linear Regression test set RMSE: {:.2f}'.format(rmse_lr))

# Print rmse_dt
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_dt))
#####################################################
#Bias-Variance Trafeoff
#####################################################

#####################################################
#Bagging and Random Forests
#####################################################

#####################################################
#Boosting
#####################################################