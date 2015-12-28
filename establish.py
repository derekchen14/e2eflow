import time as tm
import numpy as np
import pandas as pd
import personal_utils

verbose = False
np.random.seed(14)
modules_to_run = [2,3,4,5]

input_csv = "sample_data.csv"
output_csv = False
columns_to_drop = ['Removed']
cols_with_text = ['Gender']
cols_with_nan = ['Blanks']

'''
0) Load the data into Pandas
    a. read from CSV, assumes there is a header row by default
    b. find unique values
1) Get a general grasp of the data
    a. df.describe(), df.head()
    b. (examples, features) = df.shape
    c. print out the results
2) Initial cleaning on features
    a. remove features that are obviously redundant or ineffectual
    b. replace nan with "unknown", "other", or zeros
    c. change categories to integers with get_dummies
    d. do you need to drop outliers?
3) Initial cleaning on labels
    a. drop extra columns if needed
    b. change text labels into integers
4) Change pandas object types to make sklearn happy
    a. convert the features dataframe into a matrix
    b. convert the labels dataframe into an array
5) Establish a baseline
    a. create a Dummy Classifier that checks without fitting
    a. run the classifier to see that you have valid numbers
'''

def feature_label_split(dataframe):
  in_df = dataframe.iloc[:,:-1]
  out_df = dataframe.iloc[:,-1:]

  permuted_rows = np.random.permutation(in_df.index)
  in_df = in_df.reindex(permuted_rows)
  out_df = out_df.reindex(permuted_rows)

  return (in_df, out_df)

def findUniqueValues(pandas_object):
  if isinstance(pandas_object, pd.DataFrame):
    column_headers = list(pandas_object.columns.values)
    if len(column_headers) > 2:
      unique_values = manyUniqueValues(pandas_object)
    else:
      column_header = column_headers[-1]
      unique_values = set( pandas_object[column_header].tolist() )
  elif isinstance(pandas_object, pd.Series):
    unique_values = set( pandas_object.tolist() )
  else:
    unique_values = "Warning: neither a DataFrame nor a Series"
  return unique_values

def manyUniqueValues(dataframe):
  unique_values = []
  head_count = len(dataframe.columns)
  for column in dataframe:
    if head_count > 100:
      print column
    uniques = set( dataframe[column].tolist() )
    uniques = list(uniques)
    if (len(uniques) > 7):
      remaining = len(uniques) - 6
      description = "column has %d remaining values" % remaining
      truncated = uniques[0:6]
      truncated.append(description)
      result = {column: truncated}
    else:
      result = {column: uniques}
    unique_values.append(result)
  return unique_values

def fill_not_a_number(dataframe, columns):
  for col in columns:
    dataframe[col] = dataframe[col].fillna(0)
    # dataframe[col] = dataframe[col].fillna('missing')

def console_out(unique_features):
  for feature in unique_features:
    print feature

def label_mapping(dataframe, unique_classes):
  class_dictionary = {}
  for ii, value in enumerate(unique_classes):
    class_dictionary[value] = ii
  print class_dictionary
  name = dataframe.columns.values[0]
  def labeler(row):
    return class_dictionary[row]
  dataframe[name] = dataframe[name].apply( labeler )
  return dataframe

def find_column_difference(features_df, test_df):
  f_col = set(features_df.columns.values)
  t_col = set(test_df.columns.values)
  remain = list(t_col - f_col)
  print remain
  print "%d, %d, %d" % (len(f_col), len(t_col), len(remain))

  print "Dimensions of X after double drop: %d, %d" %  features_df.shape
  print "Dimensions of Z after double drop: %d, %d" %  test_df.shape

# 0) Load the data into Pandas
print "Loading data ..."
initial_time = tm.time()
if output_csv:
  features_df = pd.read_csv(input_csv, index_col=0, parse_dates=True)
  labels_df = pd.read_csv(output_csv, index_col=0, parse_dates=True)
else:
  full_df = pd.read_csv(input_csv) # , header=None)
  features_df, labels_df = feature_label_split(full_df)
unique_features = findUniqueValues(features_df)
unique_labels = findUniqueValues(labels_df)
# test_df = pd.read_csv(test_csv, index_col=0, parse_dates=True)
# fill_not_a_number(test_df, cols_with_nan)
print "----------------- Loaded in %0.2fs -----------------" % (tm.time() - initial_time)

# 1) Get a grasp of the data
if 1 in modules_to_run:
  print "FEATURE PREVIEW:"
  print features_df.head() if verbose else console_out(unique_features)
  print "Number of examples: %d rows" % features_df.shape[0]
  print "Number of features: %d columns" % features_df.shape[1]
  print "---------------------------------"

  print "LABELS PREVIEW:"
  print labels_df.head() if verbose else unique_labels
  print "Unique Classes: %d" % len(unique_labels)
  print "Number of labels: %d" % len(labels_df)
  print "---------------------------------"

  print "DATA DISTRIBUTION:"
  print np.round(features_df.describe(), 2)
  print "---------------------------------"

# 2) Initial cleaning on features
if 2 in modules_to_run:
  features_df.drop(columns_to_drop, axis=1, inplace=True)
  # test_df.drop(columns_to_drop, axis=1, inplace=True)
  fill_not_a_number(features_df, cols_with_nan)
  checkpoint = tm.time()
  features_df = pd.get_dummies(features_df, prefix='new', columns=cols_with_text)
  # test_df = pd.get_dummies(test_df, prefix='new', columns=cols_with_text)
  print "------------- Hot Encoding completed in %0.2fs ----------------" % (tm.time() - checkpoint)
  print "Dimensions of X after encoding: %d, %d" %  features_df.shape

# 3) Initial cleaning on labels
if 3 in modules_to_run:
  labels_df = label_mapping(labels_df, unique_labels)
  if verbose:
    print features_df.head()
    print labels_df.head()

# 4) Convert to matrices and arrays
if 4 in modules_to_run:
  checkpoint = tm.time()
  X = features_df.as_matrix()
  X_test = test_df.as_matrix()
  y = labels_df[labels_df.columns.values[0]].tolist()
  print "------------- Converting took about %0.2fs ---------------" % (tm.time() - checkpoint)

# 5) Establish a baseline
if 5 in modules_to_run:
  checkpoint = tm.time()
  from sklearn.dummy import DummyClassifier
  dummy_clf = DummyClassifier(strategy='most_frequent',random_state=14)
  dummy_clf.fit(X, y)
  score = dummy_clf.score(X, y) * 100
  print "Baseline score: %0.1f" % score
  print "------------- Module Five completed in %0.2fs ---------------" % (tm.time() - checkpoint)

def getData():
  return X, y

def classCount():
  return len(unique_labels)