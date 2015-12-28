import time as tm
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

#http://scikit-learn.org/stable/modules/generated/
from sklearn.svm import SVC
# sklearn.svm.SVC.html
from sklearn.ensemble import GradientBoostingClassifier
# sklearn.ensemble.GradientBoostingClassifier.html
from sklearn.naive_bayes import GaussianNB
# sklearn.naive_bayes.GaussianNB.html
'''
6) Establish your pipeline and parameters
    a. set up the classifiers
    b. plug Components into the pipeline
    c  Set up the grid search parameters and API
7) Cross Validation
X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=0.25, random_state=14)
X1, X2, y1, y2 = splitDataset(features_df, split_ratio, labels_df)
X_train, X_valid, y_train, y_valid = X1, X2, y1, y2

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp = Imputer(missing_values=0, strategy='mean', axis=0)
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(X_train)
imp.transform(X_train)

# Center and Normalize the data on just the training set, and then
# use the same transformation on any new testing sets.  You shouldn't
# take the test data into account when you calculate the mean because
# theoreticaly, you don't know what is in the test data.  Even afterward,
# when you have the test data, you should subtract by the original mean.

normalizer = preprocessing.Normalizer().fit(X)
normalizer.transform(X_train)
normalizer.transform(X_test)

def standard_normalize(X):
  #3.a1 Subtract all values by the mean value
  X -= np.mean(X, axis = 0)
  # for images in particular, perform once for each RGB color channel
  #3.a2 Divide all values by the L2 norm or standard deviation
  # use STDev if the transformation should take all data into account
  Xstandardize = X / np.std(X, axis = 0)
  # use L2 norm if you want to perform this operation per array instead
  Xnormalize = X / np.sqrt(np.sum(X**2) + 1e-5)
  # In case of images, the relative scales of pixels are already approximately
  # equal (range from 0 to 255), so not strictly necessary to perform this step
# for use with sparse vectors, when maintaing sparsity is important
# max_abs_scaler = preprocessing.MaxAbsScaler()
# 3.b Perform PCA and whitening in one step
pca = RandomizedPCA(n_components=100, whiten=True).fit(X_train)
def pca_and_whiten(X):
  #3.b1 Decorrelate data by projecting all values onto the principal eigenbasis
  cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix
  U,S,V = np.linalg.svd(cov) #compute the SVD
  Xrotated = np.dot(X, U) # rotate by multiplying by the basis vectors
  # the result is that the matrix now lies on a principal eigenvector
  Xrot_reduced = np.dot(X, U[:,:100]) # optionally, perform PCA as well
  #3.b2 Divide every dimension in the eigenbasis by the eigenvalues
  Xwhiten = Xrot_reduced / np.sqrt(S + 1e-5)
  # recall that the eigenvalues are square-root of the singluar values

#5.b1 Select top percentile of features
select_top_10_filter = SelectPercentile(chi2)  # 10% is the default
select_top_20_filter = SelectPercentile(chi2, percentile=20)
select_top_50_filter = SelectPercentile(chi2, percentile=50)
select_top_90_filter = SelectPercentile(chi2, percentile=90)
#100% just selects everything, which makes it quite useless
'''

'''

#6.a Set up your three classifiers
support_vector_classifier = SVC()
# boosted_tree_classifier = GradientBoostingClassifier()
# naive_bayes_classifier = GaussianNB()

#6.b Plug pieces into your pipeline
support_vector_pipeline = Pipeline([
  ('select_top_features', select_top_50_filter),
  ('normalize', normalizer),
  ('svm', support_vector_classifier)
])
'''
boosted_tree_pipeline = Pipeline([
  ('select_top_features', select_top_50_filter),
  ('normalize', normalizer),
  ('tree', boosted_tree_classifier)
])
naive_bayes_pipeline = Pipeline([
  ('select_top_features', select_top_50_filter),
  ('normalize', normalizer),
  ('svm', naive_bayes_classifier)
])
'''
X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=0.25, random_state=14)

support_vector_pipeline.fit(X_train, y_train)
# boosted_tree_pipeline.fit(X_train, y_train)
# naive_bayes_pipeline.fit(X_train, y_train)

# 6.c Set up grid search and final pieces
support_vector_parameters = {
  'feature_selection__percentile' : (10, 20, 50, 90),
  'gamma': [0.001, 0.01, 0.1, 1.],  # Go from 1e-5 to 1e2
  'C' : (0.1, 1.0, 10., 100.),  #regularization, too high => overfitting
  'kernel' : ('linear', 'rbf', 'poly'),
  'degree' : (2, 3, 5)
} # http://pyml.sourceforge.net/doc/howto.pdf
'''
boosted_tree_parameters = {
  'learning_rate' : (0.05, 0.1, 0.2, 0.5),
  'n_estimators' : (50, 100, 200),
  'subsample' : (0.5, 1.0, 2.0),
  'max_depth' : (2, 3, 5, 10)
}
naive_bayes_parameters = {
  'clf__bayes1' : (1,3,10),
  'clf__bayes2' : (100, 1000, 10000),
  'clf__bayes3' : (10, 20, 30, 40, 50)
}
'''
# n_jobs=-1 means run as many jobs as possible, rather than default=1
svm_grid_search = GridSearchCV(support_vector_pipeline,
  support_vector_parameters, n_jobs=-1, verbose=1, error_score=0)
# tree_grid_search = GridSearchCV(boosted_tree_pipeline,
#   boosted_tree_parameters, n_jobs=-1, verbose=1, error_score='raise')
# bayes_grid_search = GridSearchCV(naive_bayes_pipeline,
#   naive_bayes_parameters, n_jobs=-1, verbose=1, error_score='raise')

#7 Show me the money!
print("Performing grid search...")
t0 = time()
svm_grid_search.fit(X_train, y_train)
# tree_grid_search.fit(X_train, y_train)
# bayes_grid_search.fit(X_train, y_train)

svm_y_prediction = svm_grid_search.predict(X_test)
svm_report = sklearn.metrics.classification_report( y_test, svm_y_prediction )
svm_best_params = svm_grid_search.best_params_

print("done in %0.3fs" % (time() - t0))
print "Support vector machine score: %0.3f" % svm_grid_search.best_score_
print("Best parameters set:")
for p_name in svm_best_params:
    print("\t%s: %r" % (p_name, svm_best_params[p_name]))
print "SVM Classification Report:"
print svm_report


# print "gradient boosted tree score: %s" % tree_grid_search.best_score_
# print("Best parameters set:")
# tree_best_parameters = tree_grid_search.best_estimator_.get_params()
# for param_name in sorted(boosted_tree_parameters.keys()):
#     print("\t%s: %r" % (param_name, tree_best_parameters[param_name]))

# print "optimized naive bayes score: %s" % bayes_grid_search.best_score_
# print("Best parameters set:")
# bayes_best_parameters = bayes_grid_search.best_estimator_.get_params()
# for param_name in sorted(naive_bayes_parameters.keys()):
#     print("\t%s: %r" % (param_name, bayes_best_parameters[param_name]))


