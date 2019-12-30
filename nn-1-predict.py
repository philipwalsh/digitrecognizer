import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split

print('tf version', tf.__version__)


X = pd.read_csv("excluded/train.csv")
y = X['label']
X.drop('label',axis=1, inplace=True)
# nromalize each cell
X = X/255

X_train, X_ho, y_train, y_ho = train_test_split(X, y, test_size=.20, random_state=9261774)
print(X_train.shape)
print(y_train.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784, )))
model.add(tf.keras.layers.Dropout(0.20))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=5)
test_loss, test_accuracy = model.evaluate(X_ho, y_ho)
print("Test accuracy: {}".format(test_accuracy))
#save the model
model_json = model.to_json()
with open("excluded/digit_recognizer_model.json", "w") as json_file:
    json_file.write(model_json)
#save the weights
model.save_weights("excluded/digit_recognizer_weights.h5")

X_test = pd.read_csv("excluded/test.csv")
y_pred=pd.DataFrame(model.predict(X_test))
y_pred.to_csv('excluded/nn-predictions.csv', index=True)
print('\n***')
print('*** complete')
print('***')