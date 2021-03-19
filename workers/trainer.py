import logging
import os
from agent import model as mod
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils.vis_utils import plot_model
from talos import Scan
from keras.activations import relu, sigmoid
from sklearn.preprocessing import StandardScaler
import glob

from src import config, util
log = logging.getLogger(__name__)
def run(experiment_dir, root_dir, relative_experiment_path):
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
    #normalized_validation_features = scaler.transform(validation_features)

    # normalize the test data before evaluation
    normalized_test_features = scaler.transform(test_features)

    #----------------------------------------------------------------------------------

    util.write_csv(pd.DataFrame(train_labels), os.path.join(experiment_dir, 'nom_feats.csv'))

    hPars = dict(conf.h_pars)
    hPars['experiment_dir'] = [experiment_dir]
    hPars['col_dense'] =  [conf.data_feature_colnames]
    hPars['col_conv'] =  [conf.colNames]
    #print("Model plot saved================================================================================================================")
    h = Scan(x = normalized_train_features, 
             y = train_labels, 
             params = hPars, 
             model = mod.load_models,
             experiment_name=relative_experiment_path, 
             x_val = normalized_test_features, 
             y_val = test_labels, 
             print_params = True)



    all_filenames = [i for i in glob.glob(os.path.join(experiment_dir, 'Model_Weight_#*', 'confusion_matrix.csv'))]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    combined_csv.to_csv(os.path.join(experiment_dir, 'combined_metrics.csv'))

    
    