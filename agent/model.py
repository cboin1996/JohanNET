import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from keras.utils.vis_utils import plot_model
from keras.metrics import binary_accuracy
from keras.callbacks import EarlyStopping
import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix as cfm

import os
#from src import config
import logging

log = logging.getLogger(__name__)
def load_models(normalized_train_features, train_labels_encoded, 
                normalized_test_features, test_labels_encoded, h_pars):
    #Create inputs and following dense layers following the 2D structure of column names. Column names will be used to segment data for input
    in_layer = []
    dense1 = []
    experiment_dir = h_pars['experiment_dir']
    time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    fp = os.path.join(experiment_dir, f'Model_Weight_#{time_stamp}')

    ######## Model ########

    ######## Input Layer ########
    

    if h_pars['layer_type'] == 'dense':
        col_names = h_pars['col_dense']
        in_layer = tf.keras.Input(shape = (len(col_names)))
        ######## Dense Layer 1 ########
        if h_pars['activation'] == 'lrelu':
            dense1 = layers.Dense(units = len(col_names))(in_layer)
            dense1 = layers.LeakyReLU(alpha=0.1)(dense1)
        else:
            dense1 = layers.Dense(units = len(col_names), activation = h_pars['activation'])(in_layer)

        dense1 = layers.Dropout(h_pars['dropout'])(dense1)
        ######## Dense Layer 2 ########
        if h_pars['activation'] == 'lrelu':
            out_layer = layers.Dense(units = 12)(dense1)
            out_layer = layers.LeakyReLU(alpha=0.1)(dense1)
        else:
            out_layer = layers.Dense(units = 12, activation = h_pars['activation'])(dense1)
        
        out_layer = layers.Dropout(h_pars['dropout'])(out_layer)

        ######## Dense Layer 3 ########
        if h_pars['activation'] == 'lrelu':
            out_layer = layers.Dense(units = 6)(out_layer)
            out_layer = layers.LeakyReLU(alpha=0.1)(out_layer)
        else:
            out_layer = layers.Dense(units = 6, activation = h_pars['activation'])(out_layer)
        
        out_layer = layers.Dropout(h_pars['dropout'])(out_layer)

        ######## Output Layer ########
        out_layer = layers.Dense(units = 1, activation = 'sigmoid')(out_layer)
    
    elif h_pars['layer_type'] == 'conv':
        col_names = h_pars['col_conv']
        in_layer = tf.keras.Input(shape = (len(col_names[0][:]), len(col_names[:])))
        reshaped = layers.Reshape((3, 12)) (in_layer)
        
        ######## Conv Layer 1 ########
        if h_pars['activation'] == 'lrelu':
            conv1 = layers.Conv1D(filters = 12, kernel_size = 3, strides = 1, padding = 'same')(reshaped)
            out_layer = layers.LeakyReLU(alpha=0.1)(conv1)
        else:
            out_layer = layers.Conv1D(filters = 2, kernel_size = 3, padding = 'same', activation = h_pars['activation'])(reshaped)

        ######## Conv Layer 2 ########
        #if h_pars['activation'] == 'lrelu':
        #    out_layer = layers.Conv1D(filters = 2, kernel_size = 3, padding = 'valid')(out_layer)
        #    out_layer = layers.LeakyReLU(alpha=0.1)(dense1)
        #else:
        #    out_layer = layers.Conv1D(filters = 12, kernel_size = 3, padding = 'valid', activation = h_pars['activation'])(out_layer)

    out_layer = layers.Dense(units = 1, activation = 'sigmoid')(out_layer)
   
    ######## Output Layer ########

    ######## Define Model Object from Layers ########
    funcModel = tf.keras.Model(inputs = in_layer, outputs = out_layer, name = "KOIDetectionModel")
    funcModel.compile(optimizer=h_pars['optimizer'], 
                  loss= h_pars['loss'], 
                  metrics=['binary_accuracy'])
  
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=experiment_dir)

    history = funcModel.fit(normalized_train_features, 
                            train_labels_encoded, 
                            validation_split = 0.2,
                            batch_size= 5,
                            callbacks = [tensorboard_callback],
                            verbose = 1,
                            epochs = 10)
    #model_num = 
    funcModel.save(fp)
    plot_model(funcModel, to_file = os.path.join(fp, 'model_struct.png') , show_shapes=True, show_layer_names=True)
    log.info("Model Weights Saved")

    #Calculating Confusion Table
    raw_preds = pd.DataFrame(funcModel.predict(normalized_test_features, verbose = 1))
    thresh3 = np.where(raw_preds> 0.3, 1, 0)
    thresh4 = np.where(raw_preds> 0.4, 1, 0)
    thresh5 = np.where(raw_preds> 0.5, 1, 0)
    thresh6 = np.where(raw_preds> 0.6, 1, 0)
    thresh7 = np.where(raw_preds> 0.7, 1, 0)


    results = pd.DataFrame(raw_preds)
    results['03'] = thresh3
    results['04'] = thresh4
    results['05'] = thresh5
    results['06'] = thresh6
    results['07'] = thresh7

    cfm3 = pd.DataFrame(cfm(test_labels_encoded, thresh3))
    cfm5 = pd.DataFrame(cfm(test_labels_encoded, thresh5))
    cfm7 = pd.DataFrame(cfm(test_labels_encoded, thresh7))

    results.to_csv(os.path.join(fp, 'validation_results.csv'))
    cfm3.to_csv(os.path.join(fp, 'confusion_matrix_T3.csv'))
    cfm5.to_csv(os.path.join(fp, 'confusion_matrix_T5.csv'))
    cfm7.to_csv(os.path.join(fp, 'confusion_matrix_T7.csv'))
    #cfm5.to_csv(os.path.join(fp, 'confusion_matrix_T5.csv'))
    #cfm7.to_csv(os.path.join(fp, 'confusion_matrix_T7.csv'))
    
    log.info("Model Validation Saved")
    return history, funcModel

#model = load_models()

if __name__ == "__main__":
    print ("model test complete")