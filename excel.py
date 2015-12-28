import time as tm
import numpy as np
import pandas as pd
import expand as xpa

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils, generic_utils
from sklearn.metrics import classification_report

X_train, X_valid, y_train, y_valid = xpa.getData()
y_train, y_valid = [np_utils.to_categorical(x) for x in (y_train, y_valid)]
X_test = xpa.pullTest()

sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
adagrad = keras.optimizers.Adagrad()
adam = keras.optimizers.Adam(beta_1=0.85, beta_2=0.99)

sgd.title = 'sgd'
adagrad.title = 'adagrad'
adam.title = 'adam'
# optms = [adam, dama]
units = [100]
# layrs = [1,2,3,5]
actvs = ['relu']

def getIniter(activation):
  if activation == 'tanh':
    return 'glorot_normal'
  else:
    return 'he_normal'

# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=14)
y_train = np_utils.to_categorical(y_train)
# y_valid = np_utils.to_categorical(y_valid)

# for actv in actvs:
#   initer = getIniter(actv)
#   for optm in optms:
#     for unit in units:
checkpoint = tm.time()
actv = 'relu'
initer = getIniter(actv)
optm = 'adam'
unit = 100

model = Sequential()
model.add(Dense(unit, input_dim=41, init=initer))
model.add(Activation(actv))
model.add(Dropout(0.5))
model.add(Dense(unit, init=initer))
model.add(Activation(actv))
model.add(Dropout(0.5))
# if layr > 1:
#   model.add(Dense(unit, init=initer))
#   model.add(Activation(actv))
#   model.add(Dropout(0.5))
model.add(Dense(12, init=initer))
model.add(Activation('softmax'))

model.compile(loss='mean_squared_error', optimizer=optm)
model.fit(X_train, y_train, nb_epoch=20, batch_size=16, show_accuracy=True, verbose=0)
# score = model.evaluate(X_valid, y_valid, batch_size=32)
# score = np.round(1-score, 3)
# print "Activation function: %s, Optimizer: %s, Number of layers: %s" % (actv, optm.title, unit)
# print "Final score: %s" % score
print "--- Model creating and compiled in %0.2fs ---" % (tm.time() - checkpoint)

model.fit(X_train, y_train, nb_epoch=20, batch_size=32, show_accuracy=True, verbose=2)
# score = model.evaluate(X_valid, y_valid, batch_size=64)
# score = np.round(1-score, 3)
# print "Final score: %s" % score

def answer_transformer(y):
  class_dictionary = {
    0: 'functional needs repair',
    1: 'functional',
    2: 'non functional'
  }
  return class_dictionary[y]

def set_ids(y, X):
  X['amount_tsh'] = y
  print X.shape
  X.columns = ['status_group']
  return X

def write_preds(answer, fname):
  pred_df = answer.applymap(answer_transformer)
  pred_df.to_csv(fname, index=True, header=True)

print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=1)
print preds[0:4]

new_df = pd.DataFrame.from_csv("sample_sub.csv")

desired_ids = new_df.iloc[:,0:1].copy()
answer = set_ids(preds, desired_ids)

print answer[0:5]

write_preds(answer, "well_preds.csv")

# y_prediction = model.predict(X_valid, batch_size=64)
# print classification_report( y_valid, y_prediction, digits=3)



