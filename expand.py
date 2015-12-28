import time as tm
import numpy as np
import pandas as pd
import establish as est

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold

X, y = est.getData()
# X_test = est.getTest()
modules = ['bayes','tree','svm']
env = 'scoring' # searching, scoring, sending
default_scores = [0,0,0]

support_vector_parameters = {
  'select__k' : [50, 100, 'all'],
  'svm__gamma': [0.01],  # Go from 0.00001 to 10
  'svm__C' : [0.1, 1.0, 10., 100.], # regularization, too high => overfitting
  'svm__kernel' : ['rbf', 'linear', 'poly'],
  'svm__degree' : [2, 3, 5]
} # http://pyml.sourceforge.net/doc/howto.pdf
boosted_tree_parameters = {
  'feature_selection__percentile' : [10, 20, 50, 90],
  'select__k' : [50, 100, 'all'],
  'tree__learning_rate' : [0.05, 0.1, 0.2, 0.5],
  'tree__n_estimators' : [50, 100, 200],
  'tree__max_depth' : [2, 3, 5]
}

'''
I. For Hand-captured Features
  1) Remove features that have low impact on the end result
      a. VarianceThreshold - features that do not vary among the data points
      http://scikit-learn.org/stable/auto_examples/missing_values.html#example-missing-values-py
      b. impute values, which is to replace with mean or median
      c. Perform the test/train split
  2) Set up all the pre-processing components of the pipeline
      a. SelectPercentile - selects highest scoring percentage of features
      b. Select K Best - just pick the top K features
      c. Center - shift data so all resuls have zero mean
      d. Standardize - scale so the data has unit variance from -1 to 1
      e. Normalize - for features where it is important to maintain zeros
  3) Expand on previous work by building a pipeline and basic reporting
      a. set up the classifiers and estimators
      b. plug all the pieces into the pipeline
      c  try numerous combinations of parameters with grid search
      d. generate a classification report and print results
II. For Image Data
  1) Just remove IDs or based on instructions
      a. max-pooling will take care of irregularities and outliers
  2) Not needed since pixel values are all numbers
  3) Don't replace zero values, those are just values of pure white
  4) Data already comparable since pixels all lie in the same range of 0 to 255
  5) Feature reduction
      a. PCA - rotate the data so it lies along the main axis
      b. Whiten - squeeze the values together to make a sphere
III. Troubleshooting
  1) Learning Curve: you suspect you need more data
     http://scikit-learn.org/stable/modules/learning_curve.html
  2) Turn off VarianceThreshold to use more data
  3) Raise Percentile to 50 or 90, or use SelectKBest instead
'''

def print_finals(default_scores):
  bayes, tree, svm = default_scores
  print "Final Scores:"
  print "Gaussian Naive Bayes: %s" % np.round(bayes, 3)
  print "Gradient Boosted Tree: %s" % np.round(tree, 3)
  print "Support Vector Machine: %s" % np.round(svm, 3)
  print ""

def run_report(clf, X_test, y_test):
  print "Classification Report:"
  y_prediction = clf.predict(X_test)
  report = classification_report( y_test, y_prediction, digits=3)
  print report

# 1) Limit the number of features to just those at the top
print "Dimensions of X before anything:"   # %d, %d" %  X.shape
print X[0:5]

variance_filter = VarianceThreshold()
# X = variance_filter.fit_transform(X)
variance_filter.fit(X)
X = variance_filter.transform(X)
# X_test = variance_filter.transform(X_test)
# print "Dimensions of X after filtering: %d, %d" %  X.shape
print "Dimensions of X after filtering:"
print X[0:5]

selector = SelectKBest(k=5)
selector.fit(X, y)
X = selector.transform(X)
# X_test = selector.transform(X_test)
print "Dimensions of X after selecting:"
print X[0:5]
checkpoint = tm.time()

# 2) Set up all the pre-processing components of the pipeline
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
print "Dimensions of X after scaling:"
print X[0:10]

# X_test = scaler.transform(X_test)

# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
print "------------- Scaling completed in %0.2fs ---------------" % (tm.time() - checkpoint)

X1, X2, y1, y2 = train_test_split(X, y, test_size=0.25, random_state=14)
X_train, X_test, y_train, y_test = X1, X2, y1, y2

# 3.b) Plug pieces into your pipeline
if (env == "searching"):
  support_vector_pipeline = Pipeline([
    ('scale', scaler),
    ('select', selector),
    ('svm', SVC())
  ])
  support_vector_pipeline.fit(X_train, y_train)

  boosted_tree_pipeline = Pipeline([
    ('scale', scaler),
    ('select', selector),
    ('tree', tree_clf)
  ])
  boosted_tree_pipeline.fit(X_train, y_train)

  seconds = (tm.time() - checkpoint)
  m, s = divmod(seconds, 60)
  h, m = divmod(m, 60)
  print "----------- Finished fitting in %d:%02d:%02d -------------" % (h, m, s)
  checkpoint = tm.time()

# 3.c) Set up grid search and final pieces

  # n_jobs=-1 means run as many jobs as possible, rather than default=1
  # you can put in how much cross validation you want..
  svm_grid_search = GridSearchCV(support_vector_pipeline,
    support_vector_parameters, verbose=3, cv=2, n_jobs=-1, error_score=0)
  tree_grid_search = GridSearchCV(boosted_tree_pipeline,
    boosted_tree_parameters, verbose=3)

# 3.d) Run everything and generate the report
  print("Performing grid search...")
  svm_grid_search.fit(X_train, y_train)
  tree_grid_search.fit(X_train, y_train)
  seconds = (tm.time() - checkpoint)
  m, s = divmod(seconds, 60)
  h, m = divmod(m, 60)
  print "---- Grid search completed in %d:%02d:%02d -------------" % (h, m, s)

  print "Best parameters set:"
  best_params = tree_grid_search.best_params_
  for p_name in best_params:
      print("\t%s: %r" % (p_name, best_params[p_name]))

if ('bayes' in modules) and (env == 'scoring'):
  checkpoint = tm.time()
  bayes_clf = GaussianNB()
  bayes_clf.fit(X_train, y_train)
  bayes_score = cross_val_score( bayes_clf, X_test, y_test)
  default_scores[0] = np.round(bayes_score, 3)
  print "--- Bayes portion completed in %0.2fs ---" % (tm.time() - checkpoint)

if ('tree' in modules) and (env == 'scoring'):
  checkpoint = tm.time()
  tree_clf = GradientBoostingClassifier()
  tree_clf.fit(X_train, y_train)
  tree_score = cross_val_score( tree_clf, X_test, y_test )
  default_scores[1] = np.round(tree_score, 3)
  print "--- Tree portion completed in %0.2fs ---" % (tm.time() - checkpoint)

if ('svm' in modules) and (env == 'scoring'):
  checkpoint = tm.time()
  svm_clf = SVC()
  svm_clf.fit(X_train, y_train)
  svm_score = cross_val_score( svm_clf, X_test, y_test )
  default_scores[2] = np.round(svm_score, 3)
  print "--- SVM portion completed in %0.2fs ---" % (tm.time() - checkpoint)

if (env == 'scoring'):
  print_finals(default_scores)
  run_report(bayes_clf, X_test, y_test)

def getData():
  return X_train, X_test, y_train, y_test

def pullTest():
  return X_test