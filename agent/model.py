import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from keras.utils.vis_utils import plot_model
from keras.metrics import binary_accuracy
from keras.callbacks import EarlyStopping


import os
#from workers import trainer
from src import config


def load_models(colNames, hPars, normalized_train_features, test_labels_encoded, normalized_test_features, train_labels_encoded):
    #Create inputs and following dense layers following the 2D structure of column names. Column names will be used to segment data for input
    inLayer = []
    dense1 = []
    for i in range((len(colNames))):
        inLayer.append(keras.Input(shape = (len(colNames[i],))))
        if hPars['activation'] == 'lrelu':
            dense1 = layers.Dense(units = 1)(inLayer[i])
            dense1.append(layers.LeakyReLU(alpha=0.1)(dense1))
        else:
            dense1 = layers.Dense(units = 1, activation = hPars['activation'])(inLayer[i])

    concat = layers.Concatenate()(dense1)

    if hPars['activation'] == 'lrelu':
        outLayer = layers.Dense(units = 11)(concat)
        outLayer = layers.LeakyReLU(alpha=0.1)(concat)
    else:
        outLayer = layers.Dense(units = 11, activation = hPars['activation'])(concat)

    if hPars['activation'] == 'lrelu':
        outLayer = layers.Dense(units = 5)(outLayer)
        outLayer = layers.LeakyReLU(alpha=0.1)(outLayer)
    else:
        outLayer = layers.Dense(units = 5, activation = hPars['activation'])(outLayer)

    outLayer = layers.Dense(units = 1, activation = 'binary_crossentropy')(outLayer)

    funcModel = keras.Model(inputs = inLayer, outputs = outLayer, name = "KOIDetectionModel")

    funcModel.compile(optimizer=hPars['optimizer'], 
                  loss=hPars['losses'], 
                  metrics=[binary_accuracy])

    earlyStop = EarlyStopping(monitor= 'val_loss', mode='moderate')

    history = funcModel.fit(normalized_train_features, test_labels_encoded, normalized_test_features, train_labels_encoded, validation_data = [normalized_validation_features, validation_labels_encoded], optimizer = hPars['optimizer'], callbacks = [earlyStop])
    
    print("models loaded succes")
    return history, funcModel

if __name__ == "__main__":
    print ("model test complete")