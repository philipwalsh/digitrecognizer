# Philip Walsh
# philipwalsh.ds@gmail.com
# 12/31/2019 - fighting with conda environments at the moment
# appears to be running with a cryptic message 
# WARNING:tensorflow:Falling back from v2 loop because of error: Failed to find data adapter that can handle input: <class 'pandas.core.frame.DataFrame'>, <class 'NoneType'>
# i converted all train and test, x and y to np.asarray and the worning is no more
import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split

my_epochs = 25
my_dropout = 0.20
my_test_size = 0.20
my_learning_rate =.000875
my_activation = 'relu'
my_optimizer = 'adam'

# 'relu', 'elu', 'tanh'
# adam, rmsprop?

##########################################################################
##### TEST SIZE 0.20
#  5 epochs, 0.20 dropout: loss= 0.0946, accuracy=0.9685, kaggle=0.96685 
# 10 epochs, 0.20 dropout: loss= 0.0562, accuracy=0.9710, kaggle=0.97614
# 25 epochs, 0.20 dropout: loss= 0.0639, accuracy=0.9730, kaggle=0.97857 *
# 50 epochs, 0.20 dropout: loss= 0.0764, accuracy=0.9760, kaggle=0.97371
#
#  5 epochs, 0.30 dropout: loss= 0.0723, accuracy=0.9668, kaggle=0.????? 
# 10 epochs, 0.30 dropout: loss= 0.0587, accuracy=0.9718, kaggle=0.?????
# 25 epochs, 0.30 dropout: loss= 0.0523, accuracy=0.9750, kaggle=0.97514 *
# 50 epochs, 0.30 dropout: loss= 0.0695, accuracy=0.9766, kaggle=0.?????
#####
##########################################################################
# re running my best, with a larger data set
# all rows shifted left 5 px and right 5 px, then combined
# 25 epochs, 0.20 dropout: loss= 0.1134, accuracy=0.9726, kaggle=0.97142 , not an improvement

# [0.9788] epochs=25, dropout=.2, test_size=.2, learning_rate=.001
# [0.9753] epochs=25, dropout=.2, test_size=.2, learning_rate=.002


# now with new augmented data, shifted left and de-bolded, shifted right and bolded
# [0.9685] my_epochs = 5, my_dropout= 0.25, my_test_size = 0.25, my_learning_rate=.0005
# [0.9744] my_epochs = 25, my_dropout= 0.25, my_test_size = 0.25, my_learning_rate=.0005
# [0.9767] my_epochs = 25, my_dropout= 0.15, my_test_size = 0.20, my_learning_rate=.00125, kaggle=0.97685
# [0.9765] my_epochs = 35, my_dropout= 0.25, my_test_size = 0.10, my_learning_rate=.00125, kaggle=0.97385


# [0.9706] my_epochs = 20, my_dropout = 0.10, my_test_size = 0.10, my_learning_rate = .01, my_activation = 'relu', my_optimizer = 'adam'

# nn with 
# [0.9774] my_epochs = 30, my_dropout = 0.30, my_test_size = 0.20, my_learning_rate =.005, my_activation = 'relu', my_optimizer = 'adam', kaggle=


# 0.9729 my_epochs = 15, my_dropout = 0.20, my_test_size = 0.20, my_learning_rate =.000875, my_activation = 'relu', my_optimizer = 'adam'
# 0.9737 my_epochs = 20, my_dropout = 0.20, my_test_size = 0.20, my_learning_rate =.000875, my_activation = 'relu', my_optimizer = 'adam'


import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


print('tf version', tf.__version__)


X = pd.read_csv("excluded/combined_train.csv")
y = X['label']
X.drop('label',axis=1, inplace=True)
# nromalize each cell
X = X/255

X_train, X_ho, y_train, y_ho = train_test_split(X, y, test_size=my_test_size, random_state=9261774)
print(X_train.shape)
print(y_train.shape)


# TF may want an array, not a dataFrame, so these next 2 lines may fix that issue 
y_train = np.asarray(y_train)
y_ho = np.asarray(y_ho)

# lets do the same with the X
X_train = np.asarray(X_train)
X_ho = np.asarray(X_ho)


#print(1/0)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=188, activation=my_activation, input_shape=(784, )))
model.add(tf.keras.layers.Dropout(my_dropout))
model.add(tf.keras.layers.Dense(units=94, activation=my_activation, input_shape=(784, )))
model.add(tf.keras.layers.Dropout(my_dropout))
model.add(tf.keras.layers.Dense(units=28, activation=my_activation, input_shape=(784, )))
model.add(tf.keras.layers.Dropout(my_dropout))

model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model.compile(optimizer=my_optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'], learning_rate=my_learning_rate)
model.summary()
model.fit(X_train, y_train, epochs=my_epochs, verbose=False)
test_loss, test_accuracy = model.evaluate(X_ho, y_ho)
print("Test accuracy: {}".format(test_accuracy))
#save the model
model_json = model.to_json()
with open("excluded/digit_recognizer_model.json", "w") as json_file:
    json_file.write(model_json)
#save the weights
model.save_weights("excluded/digit_recognizer_weights.h5")


#print(1/0)
X_test = pd.read_csv("excluded/test.csv")
X_test = np.asarray(X_test)
#X_test = X_test/255
y_pred=pd.DataFrame(model.predict(X_test))
y_pred.to_csv('excluded/nn-predictions.csv', index=True)
print('\n***')
print('*** complete')
print('***')