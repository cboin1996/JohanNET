import logging
import os
from agent import model as mod
from sklearn.preprocessing import LabelEncoder
from keras.utils.vis_utils import plot_model
from talos import Scan
from keras.activations import relu, sigmoid
from sklearn.preprocessing import StandardScaler

from src import config, util
log = logging.getLogger(__name__)
def run(experiment_dir, root_dir):
    """Launches the trainer worker.

    Args:
        experiment_dir (str): the absolute path to the folder used to output training results
    """
    log.info("Trainer launched successfully.")
    conf = config.Default()
    # setup file paths for experiment
    fp_raw_data       = os.path.join(root_dir, conf.data_dirname, conf.raw_data_fname)
    fp_train_features = os.path.join(experiment_dir, conf.train_feat_fname)
    fp_train_labels   = os.path.join(experiment_dir, conf.train_label_fname)
    fp_valid_features = os.path.join(experiment_dir, conf.val_feat_fname)
    fp_valid_labels   = os.path.join(experiment_dir, conf.val_label_fname)
    fp_test_features  = os.path.join(experiment_dir, conf.test_feat_fname)
    fp_test_labels   = os.path.join(experiment_dir, conf.test_label_fname)

    # separate the data for training, validation, and testing
    train_features, train_labels, test_features, test_labels, validation_features, validation_labels = util.collect_processed_data(fp_raw_data, 
                                                                                                                                fp_train_features,
                                                                                                                                fp_train_labels,
                                                                                                                                fp_test_features,
                                                                                                                                fp_test_labels,
                                                                                                                                conf.data_feature_colnames,
                                                                                                                                conf.data_label_colnames,
                                                                                                                                fp_valid_features_out=fp_valid_features,
                                                                                                                                fp_valid_labels_out=fp_valid_labels,
                                                                                                                                index_col='rowid',
                                                                                                                                drop_cols=conf.raw_data_drop_cols,
                                                                                                                                train_data_split_factor = conf.train_data_split_factor,
                                                                                                                                valid_data_split_factor = conf.valid_data_split_factor,
                                                                                                                                seed=conf.random_seed)
    
    # normalize the training data before training
    scaler = StandardScaler()
    normalized_train_features = scaler.fit_transform(train_features)

    # training loop here

    # normalize the validation before validation
    normalized_validation_features = scaler.transform(validation_features)

    # normalize the test data before evaluation
    normalized_test_features = scaler.transform(test_features)

    #----------------------------------------------------------------------------------
    #Use label encoder to transform
    enc = LabelEncoder()
    enc.fit(["FALSE POSITIVE", "CANDIDATE"])
    test_labels_encoded = enc.transform(train_labels[:,1])
    train_labels_encoded = enc.transform(test_labels[:,1])
    validation_labels_encoded = enc.transform(validation_labels[:,1])

    colNames = [['koi_period' ,'koi_period_err1' ,'koi_period_err2'],
            ['koi_time0bk','koi_time0bk_err1','koi_time0bk_err2'],
            ['koi_impact','koi_impact_err1' ,'koi_impact_err2'],
            ['koi_duration','koi_duration_err1','koi_duration_err2'],
            ['koi_depth','koi_depth_err1','koi_depth_err2'],
            ['koi_prad','koi_prad_err1','koi_prad_err2'],
            ['koi_insol','koi_insol_err1','koi_insol_err2'],
            ['koi_steff','koi_steff_err1','koi_steff_err2'],
            ['koi_slogg','koi_slogg_err1','koi_slogg_err2'],
            ['koi_srad','koi_srad_err1','koi_srad_err2'],
            ['ra','dec','koi_kepmag', 'koi_teq','koi_model_snr','koi_tce_plnt_num']]

    hPars ={
    'activation': ['relu', 'sigmoid'],
    'optimizer': ['Adam', 'RMSprop'],
    'loss': ['binary_crossentropy', 'logcosh']
    }

    #model = mod.load_models(colNames, hPars, normalized_train_features, test_labels_encoded, normalized_test_features, train_labels_encoded)
    #plot_model(model, to_file= os.path.join(experiment_dir, 'model_struct.png'), show_shapes=True, show_layer_names=True)
    
    h = Scan(x = normalized_train_features, y = test_labels_encoded, x_val = normalized_validation_features, y_val = validation_labels_encoded, params = hPars, model = mod.load_models, print_params = True)