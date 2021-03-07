import logging
import datetime

from src import config, util
from workers import trainer 

import numpy as np
import tensorflow as tf
import os, sys
import random

logger = logging.getLogger(__name__)

def setup_global_logging_stream(conf: config.Default):
    """sets up the logging stream to stdout

    Args:
        config (config.Default): the config.py file
    """
    console = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(conf.log_format)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def start(args):
    conf = config.Default()
    root_dir = os.path.dirname(args[0])
    cmdl_args = args[1:]
    parsed_args = util.get_cmdl_args(cmdl_args, conf.cmdl_choices)

    util.inititialize_dirs(conf.dirs, root_dir)
    """ Set the seed generators for repeatable experiments """
    np.random.seed(conf.random_seed)
    tf.random.set_seed(conf.random_seed)
    os.environ['PYTHONHASHSEED']=str(conf.random_seed)
    random.seed(conf.random_seed)


    if parsed_args.mode == conf.param_tr:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        experiment_dir = os.path.join(root_dir, conf.output_dirname, timestamp+f'_seed{conf.random_seed}')
        os.mkdir(experiment_dir)

        """ Setup logging to file and console """
        logging.basicConfig(level=logging.INFO,
                            format=conf.log_format,
                            datefmt=conf.log_date_fmt,
                            filename=os.path.join(experiment_dir, "out.log"),
                            filemode='w')

        setup_global_logging_stream(conf)
        trainer.run(experiment_dir, root_dir)

    elif parsed_args.mode == conf.param_pred:
        setup_global_logging_stream(conf)


if __name__=="__main__":
    start(sys.argv)