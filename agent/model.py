import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from keras.utils.vis_utils import plot_model
from keras.metrics import binary_accuracy
from keras.callbacks import EarlyStopping

import os
from src import config
import logging

log = logging.getLogger(__name__)
def load_models(normalized_train_features, test_labels_encoded, 
                normalized_test_features, train_labels_encoded, hPars):
    #Create inputs and following dense layers following the 2D structure of column names. Column names will be used to segment data for input
    inLayer = []
    dense1 = []
    colNames = hPars['col_names'][0]
    experiment_dir = hPars['experiment_dir'][0]

    inLayer = tf.keras.Input(shape = len(colNames))

    if hPars['activation'] == 'lrelu':
        dense1 = layers.Dense(units = len(colNames))(inLayer)
        dense1 = layers.LeakyReLU(alpha=0.1)(dense1)
    else:
        dense1 = layers.Dense(units = len(colNames), activation = hPars['activation'])(inLayer)


    if hPars['activation'] == 'lrelu':
        outLayer = layers.Dense(units = 11)(dense1)
        outLayer = layers.LeakyReLU(alpha=0.1)(dense1)
    else:
        outLayer = layers.Dense(units = 11, activation = hPars['activation'])(dense1)

    if hPars['activation'] == 'lrelu':
        outLayer = layers.Dense(units = 5)(outLayer)
        outLayer = layers.LeakyReLU(alpha=0.1)(outLayer)
    else:
        outLayer = layers.Dense(units = 5, activation = hPars['activation'])(outLayer)

    outLayer = layers.Dense(units = 1, activation = hPars['activation'])(outLayer)

    funcModel = tf.keras.Model(inputs = inLayer, outputs = outLayer, name = "KOIDetectionModel")
    plot_model(funcModel, to_file= os.path.join(experiment_dir, 'model_struct.png'), show_shapes=True, show_layer_names=True)   

    funcModel.compile(optimizer=hPars['optimizer'], 
                  loss=hPars['loss'], 
                  metrics=[binary_accuracy])
    
    earlyStop = EarlyStopping(monitor= 'val_loss', mode='moderate')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=experiment_dir)

    history = funcModel.fit(normalized_train_features, 
                            test_labels_encoded, 
                            validation_data=[normalized_test_features,train_labels_encoded],
                            callbacks = [earlyStop, tensorboard_callback])
    
    log.info("models loaded succes")
    return history, funcModel

if __name__ == "__main__":
    print ("model test complete")