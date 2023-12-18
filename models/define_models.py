# -----------------------------------------------------------------------------
# Define models 
# -----------------------------------------------------------------------------
# Author: Daniel Oliveira
# https://github.com/danielsousaoliveira
# -----------------------------------------------------------------------------
# Script to define models for training

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation

def define_alexnet_model():

    model = Sequential()

    model.add(Convolution2D(32, 11, strides = (4, 4), padding = 'valid', input_shape=(192,192,3)))
    model.add(BatchNormalization())
    model.add(Activation ('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))

    model.add(Convolution2D(64, 11, strides = (1, 1), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation ('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))

    model.add(Convolution2D(128, 3, strides = (1, 1), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation ('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))

    model.add(Flatten())

    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dropout(0.4))

    model.add(Dense(units = 3, activation = 'softmax'))

    return model

def define_mobilenet_model():

    base_model=tf.keras.applications.MobileNetV3Small(
        input_shape=(224,224,3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=False
    )

    base_model.trainable = False

    x = base_model.output
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.Model(base_model.input, x)

    return model

def define_vgg16_model():

    base_model=tf.keras.applications.VGG16(
        input_shape=(192.192,3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    x = base_model.output
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.Model(base_model.input, x)

    return model

