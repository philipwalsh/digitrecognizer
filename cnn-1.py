#################################################
# title   : digit recognizer
# from    : kaggle.com
# file    : cnn-1.py
#         : philip walsh
#         : philipwalsh.ds@gmail.com
#         : 2019-12-23
# 
#  machines status
# lenovo laptop running keras/tensorflow apprently ok on first look.
# custom pc with dual gpus cannot load keras without mkl error
# asus laptop ?

from keras.models import Sequential, load_model
#from keras.models import load_model
#from tensorflow.keras.models import Sequential, load_model



from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

def init_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Valid', activation='relu', input_shape=(28,82,1)))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64, kernel_size=(5,5), padding='Valid', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])

    annealer = ReduceLROnPlateau(monitor='val_acc', patience=1, verbose=2, factor=0.5, min_lr=0.0000001)
    #annealer = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=2, factor=0.5, min_lr=0.0000001)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False
        #, preprocessing_function=random_add_or_erase_spot)
    )
    return model, annealer, datagen




import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from os.path import isfile, join
from sklearn.model_selection import ShuffleSplit
#import lib.model as md
import lib

from sklearn.metrics import accuracy_score, confusion_matrix
from keras.utils.np_utils import to_categorical

#hide the tensoflow depreciated warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import time

start_time = time.time()
current_script = os.path.basename(__file__)
log_prefix = os.path.splitext(current_script)[0].replace('_','-')
bVerbose = False


working_dir=os.getcwd()
excluded_dir = os.path.join(working_dir, 'excluded') # working_dir + '\excluded'

print('\n')
print('*****')
print('***** start of script: ', log_prefix)
print('*****')
print('\n')

if bVerbose:
    print('\nworking dir   :', working_dir)


import winsound
def alert_me(num_beeps):
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 500  # Set Duration To 1000 ms == 1 second

    for n in range(1,num_beeps+1):
        winsound.Beep(frequency, duration)



#print('saving xxx ...', sendtofile(excluded_dir,'xxx.csv',xxx))
def sendtofile(outdir, filename, df, verbose=False):
    script_name = log_prefix + '_'
    out_file = os.path.join(outdir, script_name + filename)
    if verbose:
        print("saving file :", out_file)
    df.to_csv(out_file, index=False)
    return out_file




##
## MAIN SCRIPT START HERE
##

# load the train_data
train_data = pd.read_csv('excluded/train-min.csv', low_memory=False)
print('\ntrain_data loaded')
print('train_data.shape   :', train_data.shape)

# load the test_data
#test_data = pd.read_csv('excluded/test.csv', low_memory=False)
#print('\ntest_data loaded')
#print('test_data.shape    :', test_data.shape)



y = train_data['label']
X = train_data.drop('label', axis=1)
print(y.value_counts().to_dict())


y = to_categorical(y, num_classes=10)
del train_data
X = X / 255.0
X = X.values.reshape(-1,28,28,1)

seed = 9261774

train_index, holdout_index = ShuffleSplit(n_splits=1, train_size=.8, test_size=None, random_state=seed).split(X).__next__()
print(len(train_index))


X_train = X[train_index]
y_train = y[train_index]
X_holdout = X[holdout_index]
y_holdout = y[holdout_index]

# Parameters
epochs = 30
batch_size = 64
validation_steps = 10000

# initialize Model, Annealer and Datagen
model, annealer, datagen = init_model()

train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
holdout_generator = datagen.flow(X_holdout, y_holdout, batch_size=batch_size)


print('\n')
print('*****')
print('***** end of script: ', log_prefix)
print('*****')
print('\n')