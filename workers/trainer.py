import logging
import os
from agent import model

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
