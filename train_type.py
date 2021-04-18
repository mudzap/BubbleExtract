from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental import preprocessing

import cv2
import tensorflow as tf
import numpy as np
import glob
import os

from tensorflow import keras
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.metrics import confusion_matrix

import argparse

parser = argparse.ArgumentParser(description='Process training parameters.')
parser.add_argument('--epochs', help="N. of epochs.", default=10, type=int)
args = parser.parse_args()
EPOCHS = args.epochs


CLASSES = 3
IM_SHAPE = (50, 50, 3)

#Helper function for reading data
def add_file_to_train_set(x_train, y_train, filename, class_num, verbose = False):
    for file in glob.glob(filename):
        data = cv2.imread(file)
        data = cv2.resize(data, IM_SHAPE[:2]) #Too much data loss?
        x_train.append(data)
        y_train.append(class_num)
        if verbose:
            print("File: " + file + ", Class: " + str(class_num))

#Builds datasets
x_train = []
y_train = []
add_file_to_train_set(x_train, y_train, "output/train/no-text/*.png", 0)
add_file_to_train_set(x_train, y_train, "output/train/vert-text/*.png", 1)
add_file_to_train_set(x_train, y_train, "output/new-vert/*.png", 1)
add_file_to_train_set(x_train, y_train, "output/new-hor/*.png", 2)
add_file_to_train_set(x_train, y_train, "output/train/hor-text/*.png", 2)
x_train = np.array(x_train)/255
y_train = np.array(y_train)

x_test = []
y_test = []
add_file_to_train_set(x_test, y_test, "output/test/no-text/*.png", 0)
add_file_to_train_set(x_test, y_test, "output/test/vert-text/*.png", 1)
add_file_to_train_set(x_test, y_test, "output/test/hor-text/*.png", 2)
x_test = np.array(x_test)/255
y_test = np.array(y_test)


#random_state = 1: Initial Seeding
x_train, x_validate, y_train, y_validate = train_test_split(
    x_train, y_train,
    test_size=0.3,
    random_state = 1
)


text_model = Sequential([
    preprocessing.RandomZoom(height_factor=0.05, width_factor=0.05, fill_mode='constant'),
    preprocessing.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode='constant'),
    # It is expected to get high frequency components when dealing with
    # text-like characteristsica, to ensure identification of these and
    # for performance reasons (for now), a single convolutional layer with
    # a set of kernels of size 7x7 will be utilized.
    Conv2D(filters=32, kernel_size=7, activation='relu', input_shape=IM_SHAPE),
    MaxPooling2D(pool_size=2),
    # Typical dropout value
    Dropout(0.5),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(24, activation='relu'),
    Dense(CLASSES, activation='softmax')
])

# Training params
text_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(lr=0.001),
    metrics=['accuracy']
)

# Training
text_model.fit(
    x_train, 
    y_train,
    batch_size=512,
    epochs=EPOCHS, 
    verbose=1,
    # Since most of the data obtained is not actually text, but
    # rather 'speech bubbles candidates', classes will be weighted
    # as to prevent bias
    class_weight= {0:1., 1:2., 2:2.},
    validation_data=(x_validate, y_validate)
)

# Evaluation
score = text_model.evaluate(x_test, y_test, verbose=False)
print(score)

x_test = np.array(x_test, np.float32)
y_pred = text_model.predict(x_test)
label = np.argmax(y_pred, axis=1)
#print(label)
mat = confusion_matrix(y_test, label)
print(mat)

text_model.save("models/trained_model_" + str(EPOCHS))
print("Save resulting model to dir: models/trained_model_" + str(EPOCHS))
print("EOP")
