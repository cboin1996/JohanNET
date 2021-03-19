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
        shaped_train_features = normalized_train_features
        shaped_test_features = normalized_test_features
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
            out_layer = layers.Dense(units = h_pars['layer2_units'])(dense1)
            out_layer = layers.LeakyReLU(alpha=0.1)(dense1)
        else:
            out_layer = layers.Dense(units = h_pars['layer2_units'], activation = h_pars['activation'])(dense1)
        
        out_layer = layers.Dropout(h_pars['dropout'])(out_layer)

        ######## Dense Layer 3 ########
        if h_pars['activation'] == 'lrelu':
            out_layer = layers.Dense(units = 6)(out_layer)
            out_layer = layers.LeakyReLU(alpha=0.1)(out_layer)
        else:
            out_layer = layers.Dense(units = h_pars['layer3_units'], activation = h_pars['activation'])(out_layer)
        
        out_layer = layers.Dropout(h_pars['dropout'])(out_layer)

        ######## Output Layer ########
        out_layer = layers.Dense(units = 1, activation = 'sigmoid')(out_layer)
    """
    elif h_pars['layer_type'] == 'conv':
        
        #reshape data
        print("Model plot saved================================================================================================================")
        print(normalized_train_features.shape)
        shaped_train_features = np.reshape(normalized_train_features, (-1, 12,3))
        print("Model plot saved================================================================================================================")
        print(shaped_train_features.shape)
        shaped_test_features =  np.reshape(normalized_test_features, (-1, 12,3))

        col_names = h_pars['col_conv']
        in_layer = tf.keras.Input(shape = (len(col_names[:]), len(col_names[0][:])))
        #reshaped = layers.Reshape((12, 3)) (in_layer)
        
        ######## Conv Layer 1 ########
        if h_pars['activation'] == 'lrelu':
            conv1 = layers.Conv1D(filters = 32, kernel_size = 3, padding = 'same')(in_layer)
            out_layer = layers.LeakyReLU(alpha=0.1)(conv1)
        else:
            out_layer = layers.Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation = h_pars['activation'])(in_layer)
"""
    out_layer = layers.Dense(units = 1, activation = 'sigmoid')(out_layer)
   
    ######## Output Layer ########

    ######## Define Model Object from Layers ########
    funcModel = tf.keras.Model(inputs = in_layer, outputs = out_layer, name = "KOIDetectionModel")
    funcModel.compile(optimizer=h_pars['optimizer'], 
                  loss= h_pars['loss'], 
                  metrics=['binary_accuracy'])
  
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=experiment_dir)

    history = funcModel.fit(shaped_train_features, 
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

    tp3, fn3, flp3, tn3 = cfm(test_labels_encoded, thresh3).ravel()
    tp5, fn5, flp5, tn5 = cfm(test_labels_encoded, thresh5).ravel()
    tp7, fn7, flp7, tn7 = cfm(test_labels_encoded, thresh7).ravel()

    ###PUT CFM CALCULATIONS HERE###
    cfm_metrics = pd.DataFrame()
    accuracy = [(tp3 + flp3)/(tp3 + flp3 + tn3 + fn3),
                (tp5 + flp5)/(tp5 + flp5 + tn5 + fn5),
                (tp7 + flp7)/(tp7 + flp7 + tn7 + fn7)]
    cfm_metrics['accuracy'] = accuracy

    precision = [(tp3)/(tp3 + flp3),
                (tp5)/(tp5 + flp5),
                (tp7)/(tp7 + flp7)]
    cfm_metrics['precision'] = precision

    recall = [(tp3)/(tp3 + fn3),
                (tp5)/(tp5 + fn5),
                (tp7)/(tp7 + fn7)]
    cfm_metrics['recall'] = recall


    #f1 = pd.DataFrame(2*((precision.iloc[0] * recall.iloc[0])/ (precision.iloc[0] + recall.iloc[0])))
    #print (f1)
    
    hpar_file = pd.DataFrame([['layer_type', h_pars['layer_type']], 
                            ['layer2_units', h_pars['layer2_units']], 
                            ['layer3_units', h_pars['layer3_units']], 
                            ['activation', h_pars['activation']], 
                            ['optimizer', h_pars['optimizer']], 
                            ['loss', h_pars['loss']], 
                            ['dropout', h_pars['dropout']]])

    hpar_file.to_csv(os.path.join(fp, 'hyper_params.csv'))
    results.to_csv(os.path.join(fp, 'validation_results.csv'))

    #accuracy.to_csv(os.path.join(fp, 'confusion_matrix_accuracy.csv'))
    #precision.to_csv(os.path.join(fp, 'confusion_matrix_precision.csv'))
    #recall.to_csv(os.path.join(fp, 'confusion_matrix_recall.csv'))
    #f1.to_csv(os.path.join(fp, 'confusion_matrix_f1.csv'))
    cfm_metrics.to_csv(os.path.join(fp, 'confusion_matrix.csv'))
    #cfm5.to_csv(os.path.join(fp, 'confusion_matrix_T5.csv'))
    #cfm7.to_csv(os.path.join(fp, 'confusion_matrix_T7.csv'))

    
    log.info("Model Validation Saved")
    return history, funcModel

if __name__ == "__main__":
    print ("model test complete")