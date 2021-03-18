import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from keras.utils.vis_utils import plot_model
from keras.metrics import binary_accuracy
from keras.callbacks import EarlyStopping
from datetime import datetime

import os
#from src import config
import logging

log = logging.getLogger(__name__)
def load_models(normalized_train_features, train_labels_encoded, 
                normalized_test_features, test_labels_encoded, hPars):
    #Create inputs and following dense layers following the 2D structure of column names. Column names will be used to segment data for input
    in_layer = []
    dense1 = []
    col_names = hPars['col_names']
    experiment_dir = hPars['experiment_dir']

    ######## Model ########

    ######## Input Layer ########
    in_layer = tf.keras.Input(shape = (len(col_names)))

    ######## Dense Layer 1 ########
    if hPars['activation'] == 'lrelu':
        dense1 = layers.Dense(units = len(col_names))(in_layer)
        dense1 = layers.LeakyReLU(alpha=0.1)(dense1)
    else:
        dense1 = layers.Dense(units = len(col_names), activation = hPars['activation'])(in_layer)

    ######## Dense Layer 2 ########
    if hPars['activation'] == 'lrelu':
        out_layer = layers.Dense(units = 12)(dense1)
        out_layer = layers.LeakyReLU(alpha=0.1)(dense1)
    else:
        out_layer = layers.Dense(units = 12, activation = hPars['activation'])(dense1)

    ######## Dense Layer 3 ########
    if hPars['activation'] == 'lrelu':
        out_layer = layers.Dense(units = 6)(out_layer)
        out_layer = layers.LeakyReLU(alpha=0.1)(out_layer)
    else:
        out_layer = layers.Dense(units = 6, activation = hPars['activation'])(out_layer)

    ######## Output Layer ########
    out_layer = layers.Dense(units = 1, activation = 'sigmoid')(out_layer)

    ######## Define Model Object from Layers ########
    funcModel = tf.keras.Model(inputs = in_layer, outputs = out_layer, name = "KOIDetectionModel")
    funcModel.compile(optimizer=hPars['optimizer'], 
                  loss= hPars['loss'], 
                  metrics=['binary_accuracy'])

    plot_model(funcModel, to_file= os.path.join(experiment_dir, 'model_struct.png'), show_shapes=True, show_layer_names=True)
    
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=experiment_dir)

    history = funcModel.fit(normalized_train_features, 
                            train_labels_encoded, 
                            validation_split = 0.2,
                            batch_size= 5,
                            callbacks = [tensorboard_callback],
                            verbose = 1,
                            epochs = 10)
    #model_num = 
    #funcModel.save(os.path.join(experiment_dir, 'Model_Weight_#%s', datetime.now(tz=None)))
    log.info("Model Weights Saved")

    funcModel.predict(normalized_test_features, verbose = 1)
    log.info("Model Validation Saved")
    return history, funcModel

#model = load_models()

if __name__ == "__main__":
    print ("model test complete")