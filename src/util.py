import argparse
import h5py
import json
import tensorflow as tf
from types import SimpleNamespace
import numpy as np
import random
import os, sys
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

def get_cmdl_args(args: list, choices): 
    """Simple command line parser

    Args:
        args (list): the input arguments from command prompt
        return (list) : the list of parsed arguments
    """
    parser = argparse.ArgumentParser(description="Predict planets, its chill.")
    parser.add_argument("mode",
                        choices=choices,
                        help="What mode should I run?")

    return parser.parse_args(args)


def save_file(fpath, txt):
    with open(fpath, 'w') as f:
        log.info(f"Saving {txt} to : {fpath}")
        f.write(txt)

def config_writer(fpath, obj):
    with open(fpath, 'w') as f:
        log.info(f"Saving configuration Config.py as json: outfile -> {fpath}.")
        json.dump(obj.__dict__, f)

def config_loader(fpath):
    with open(fpath, 'r') as f:
        return json.load(f, object_hook=lambda d: SimpleNamespace(**d))

def load_json(fpath):
    with open(fpath, 'r') as f:
        return json.load(f)

def latexify(s):
    return s.replace('_', '\_').replace('%', '\%')

def print_dct(dct):
    for k, v in dct.items():
        log.info(f"{latexify(k)} & {v} \\\\")

def inititialize_dirs(dirs, root_dir):
    for directory in dirs:
        dir_path = os.path.join(root_dir, directory)
        if not os.path.exists(dir_path):
            log.info(f"Making dir {dir_path}")
            os.mkdir(dir_path)

def read_csv(fp, index_col=None):
    return pd.read_csv(fp, index_col=index_col)

def write_csv(df, fp):
    log.info(f"Generating file : {fp}")
    df.to_csv(fp)

def collect_processed_data(fp_in, fp_train_features_out, fp_train_labels_out, fp_test_features_out, 
                           fp_test_labels_out, feature_cols, label_cols, fp_valid_features_out=None, fp_valid_labels_out=None, 
                           index_col=None, drop_cols=None,
                           train_data_split_factor=0.75, valid_data_split_factor=0,
                           seed=None):
    """Load's a dataframe at fp_in, performs processing, and writes out to files

    Args:
        fp_in (str): file path of the dataframe to process
        index_col (int): the column containing the indices of the raw df
        fp_train_out (str): file path for the training data
        fp_valid_out (str): file path for the validation data
        fp_test_out (str) : file path for the test data
        feature_cols (list) : the list of the names of the feature columns in the raw dataframe
        label_cols   (list) : the list of the names of the label columns in the raw dataframe
        index_col    (str)  : the name of the inded columns (if any)
        train_data_split_factor : the percent of data to use for training
        valid_data_split_factor : the percent of data to use for validation (tuning hyperparameters)
    Returns:
        (tuple) : training feature dataset, training label dataset, test feature dataset, test label dataset, validation feature dataset, validation label dataset
    """


    log.info("Beginning Data Preprocessing:")
    test_data_split_factor = 1 - train_data_split_factor
    initial_output = '\n'.join([
        f"feature_cols: {feature_cols}",
        f"label cols: {label_cols}",
        f"removing columns: {drop_cols}",
        f"Splits: {train_data_split_factor} for training, {valid_data_split_factor} for validation, {test_data_split_factor}"
    ])
    log.info(initial_output)
    raw_data = read_csv(fp_in, index_col=index_col)
    

    proc_df = raw_data.drop(columns=drop_cols)
    proc_df = proc_df.dropna()

    feature_df = proc_df.drop(columns=label_cols)
    label_df = proc_df.drop(columns=feature_cols)
    enc = LabelEncoder()
    label_df_enc = enc.fit_transform(label_df['koi_pdisposition'])
    label_df = label_df.drop(columns='koi_pdisposition')
    print(label_df.shape)
    label_df['koi_pdisposition'] = label_df_enc

    # perform split into training data, and remaining data 'intermediate'
    train_features, intermediate_features, train_labels, intermediate_labels = model_selection.train_test_split(feature_df,
                                                                                                                label_df,
                                                                                                                train_size=train_data_split_factor,
                                                                                                                random_state=seed)
    # split data further if validation split factor is non zero
    if valid_data_split_factor != 0:
        test_data_split_from_remaining = 1 - (valid_data_split_factor / (1 - train_data_split_factor)) # calculates split factor scaled to the remaining data size after first split
        test_features, validation_features, test_labels, validation_labels = model_selection.train_test_split(intermediate_features,
                                                                                                              intermediate_labels,
                                                                                                              train_size=test_data_split_from_remaining,
                                                                                                              random_state=seed)
        write_csv(validation_features, fp_valid_features_out)
        write_csv(validation_features, fp_valid_labels_out)
    else:   
        test_features = intermediate_features
        test_labels   = intermediate_labels
        validation_features = pd.DataFrame()
        validation_labels = pd.DataFrame()

    write_csv(train_features, fp_train_features_out)
    write_csv(train_labels, fp_train_labels_out)
    write_csv(test_features, fp_test_features_out)
    write_csv(test_labels, fp_test_labels_out)
    summary = '\n'.join([
       "Split comlete. Data lengths summarized below: ",
       f"Total data : {len(proc_df)}",
       f"Train : {len(train_features)}",
       f"Test : {len(test_features)}",
       f"Validation features : {len(validation_features)}"])
    log.info(summary)
    
    return train_features.to_numpy(), train_labels.to_numpy(), test_features.to_numpy(), test_labels.to_numpy(), validation_features.to_numpy(), validation_labels.to_numpy()

    

