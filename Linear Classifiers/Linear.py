################################################
#Minimizeing a loss
    #loss: sum(target - predicted)^2
    #not good for classfication problem
################################################
from scipy.optimize import minimize
import numpy as np
minimize(np.square,0).x
print(minimize(np.square,2).x)
# The squared error, summed over training examples
def my_loss(w):
    s = 0
    for i in range(y.size):
        # Get the true and predicted target values for example 'i'
        y_i_true = y[i]
        y_i_pred = w@X[i]
        s = s + (y_i_true-y_i_pred)**2
    return s

# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)

# Compare with scikit-learn's LinearRegression coefficients
lr = LinearRegression(fit_intercept=False).fit(X,y)
print(lr)
################################################
#Loss function diagrams
################################################
# Mathematical functions for logistic and hinge losses
def log_loss(raw_model_output):
   return np.log(1+np.exp(-raw_model_output))
def hinge_loss(raw_model_output):
   return np.maximum(0,1-raw_model_output)

# Create a grid of values and plot
grid = np.linspace(-2,2,1000)
plt.plot(grid, log_loss(grid), label='logistic')
plt.plot(grid, hinge_loss(grid), label='hinge')
plt.legend()
plt.show()
################################################
#Loss function (logistic regressions)
################################################
# The logistic loss, summed over training examples
def my_loss(w):
    s = 0
    for i in range(len(y)):
        raw_model_output = w@X[i]
        s = s + log_loss(raw_model_output * y[i])
    return s

# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)

# Compare with scikit-learn's LogisticRegression
lr = LogisticRegression(fit_intercept=False, C=1000000).fit(X,y)
print(lr.coef_)
################################################
#Regularization
    #regularized loss = origianl loss + penalized large coefficient 
    #larger the penalized(smaller C value), reduce the goal of maximizing accuracy(train data)
        #but increase accuracy of test data, regularization --> less overfit
    #Linear regression regularization
        #L1(Lasso): feature selection, coefficienets close to zero
        #L2(Ridge): just shrink coefficients to smaller
################################################
lr_weak_reg= LinearRegression(C=100)
lr_strong_reg= LinearRegression(C=0.01)
lr_weak_reg.fit(X_train, y_train)
lr_strong_reg.fit(X_train, y_train)
lr_weak_reg.score(X_train,y_train)
lr_strong_reg.score(X_train,y_train)
################################################
#L1 and L2 Regularization
################################################
lr_L1= LogisticRegression(penalty='l1')
lr_L2= LogisticRegression(penalty='l2')
lr_L1.fit(_train, y_train)
lr_L2.fit(_train, y_train)
plt.plot(lr_L1.coef_.flatten())
plt.plot(lr_L2.coef_.flatten())

# Train and validaton errors initialized as empty list
train_errs = list()
valid_errs = list()

# Loop over values of C_value
for C_value in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    # Create LogisticRegression object and fit
    lr = LogisticRegression(C=C_value)
    lr.fit(X_train,y_train)
    
    # Evaluate error rates and append to lists
    train_errs.append( 1.0 - lr.score(X_train,y_train) )
    valid_errs.append( 1.0 - lr.score(X_valid,y_valid) )
    
# Plot results
plt.semilogx(C_values, train_errs, C_values, valid_errs)
plt.legend(("train", "validation"))
plt.show()
################################################
#Logistic regression and feature selection
################################################
# Specify L1 regularization
lr = LogisticRegression(penalty='l1')

# Instantiate the GridSearchCV object and run the search
searcher = GridSearchCV(lr, {'C':[0.001, 0.01, 0.1, 1, 10]})
searcher.fit(X_train, y_train)

# Report the best parameters
print("Best CV params", searcher.best_params_)

# Find the number of nonzero coefficients (selected features)
best_lr = searcher.best_estimator_
coefs = best_lr.coef_
print("Total number of features:", coefs.size)
print("Number of selected features:", np.count_nonzero(coefs))
################################################
#Regularization and probabilities
################################################
# Set the regularization strength
model = LogisticRegression(C=0.1)
model.fit(X,y)# Fit and plot
plot_classifier(X,y,model,proba=True)
prob = model.predict_proba(X)# Predict probabilities on training points
print("Maximum predicted probability", np.max(prob))

lr = LogisticRegression()
lr.fit(X,y)
proba = lr.predict_proba(X)# Get predicted probabilities
proba_inds = np.argsort(np.max(proba,axis=1))# Sort the example indices by their maximum probability
show_digit(proba_inds[-1], lr)# Show the most confident (least ambiguous) digit
show_digit(proba_inds[0], lr)# Show the least confident (most ambiguous) digit
################################################
#Multi-class logistic regression
################################################
lr0.fit(X,y==0) #0 versus rest
lr1.fit(X,y==1) #1 versus rest
lr2.fit(X,y==2)
lr0.decision_function(X)[0] #raw model output
lr1.decision_function(X)[0]
lr2.decision_function(X)[0]
lr.fit(X,y)
################################################
#One-vs-rest / Multinomial
    #binary classfier for each class/ single classfier for all clasess
    #predict with all, take largest output/ prediction directly outputs best class
    #simple modular/ more complicated
    #not direclty optimizing accuract/ problem directly
    # common for SVMs/ possible for SVMs, but less common
################################################
#One-vs-rest 
lr_ovr = LogisticRegression()
lr_ovr.fit(X_train, y_train)
lr_ovr.coef_.shape
lr_ovr.intercept_.shape
print("OVR training accuracy:", lr_ovr.score(X_train, y_train))
print("OVR test accuracy    :", lr_ovr.score(X_test, y_test))
#Multinomial
lr_mn = LogisticRegression(multi_class="multinomial",solver="lbfgs")
lr_mn.fit(X_train, y_train)
lr_mn.coef_.shape
lr_mn.intercept_.shape
print("Softmax training accuracy:", lr_mn.score(X_train, y_train))
print("Softmax test accuracy    :", lr_mn.score(X_test, y_test))
################################################
#Visualizing multi-class logistic regression
################################################
#logisticregression classifier
lr_class_1 = LogisticRegression(C=100)# Create the binary classifier (class 1 vs. rest)
lr_class_1.fit(X_train, y_train==1)
plot_classifier(X_train, y_train==1, lr_class_1)# Plot the binary classifier (class 1 vs. rest)
#SVC classifier
from sklearn.svm import SVC
svm_class_1 = SVC()# Create/plot the binary classifier (class 1 vs. rest)
svm_class_1.fit(X_train, y_train==1)
plot_classifier(X_train, y_train==1,svm_class_1)
################################################
#SVM(Support Vector): maximize the margin
    #part of linear regression, has hindge loss + L2 regularization
#Kernel SVM: faster to predict
################################################
svm = SVC(kernel="linear")# Train a linear SVM
svm.fit(X,y)
plot_classifier(X, y, svm, lims=(11,15,0,6))

print("Number of original examples", len(X))# Make a new data set keeping only the support vectors
print("Number of support vectors", len(svm.support_))
X_small = X[svm.support_]
y_small = y[svm.support_]

svm_small = SVC(kernel="linear")# Train a new SVM using only the support vectors
svm_small.fit(X_small,y_small)
plot_classifier(X_small, y_small, svm_small, lims=(11,15,0,6))
################################################
#Kernel SVM
    #gamma: control smoother the boundary(decrease --> smoother)
    #C: control shape of coundary
################################################# Instantiate an RBF SVM
# Instantiate an RBF SVM
svm = SVC()

# Instantiate the GridSearchCV object and run the search
parameters = {'C':[0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X_train,y_train)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)

# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test,y_test))
################################################
#Logistc Regression/ SVM
    #linear classfier/linear classfier
    #can use with kernels, but slow/ can and fast
    #output meaningful probabilities/no output probabilities
    #can extended to multi-class/same
    #all data points affect fit/onlt 'support vectors' affect fit
    #L2 or L1 regularization / just L2 regularization
################################################
#logistic regression
linear_model.LogisticRegression()
C, penalty, multi_class
#SVM
svm.LinearSVC
svm.SVC
C,kernel, gamma
#SGD-classifier(stochastic gradient descent)
SGDClassifier: scale well to large datasets
logreg=SGDClassifier(loss='log')
linsvm=SGDClassifier(loss='hinge')
alpha = 1/C
################################################
#SGD-classifier
################################################
# We set random_state=0 for reproducibility 
linear_classifier = SGDClassifier(random_state=0)

# Instantiate the GridSearchCV object and run the search
parameters = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 
             'loss':['hinge', 'log'], 'penalty':['l1','l2']}
searcher = GridSearchCV(linear_classifier, parameters, cv=10)
searcher.fit(X_train, y_train)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))
