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
    df.to_csv(fp)

def initialize_dataframe(fp_in, fp_out, index_col=None, drop_cols=None):
    """Load's a dataframe at fp_in, performs processing, and writes out to file, returning the df in the process

    Args:
        fp_in (str): file path of the dataframe to process
        index_col (int): the column containing the indices of the raw df
        fp_out (str): file path of the destination dataframe, including filename
    """
    raw_df = read_csv(fp_in, index_col)
     # Perform the data preprocessing here
    proc_df = raw_df.drop(columns=drop_cols)
    write_csv(proc_df, fp_out)
    return proc_df