import logging
import datetime

from src import config, util
from workers import trainer, reporter

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
    root_dir = sys.path[0]
    cmdl_args = args[1:]
    parsed_args = util.get_cmdl_args(cmdl_args, conf.cmdl_choices)

    util.inititialize_dirs(conf.dirs, root_dir)
    """ Set the seed generators for repeatable experiments """
    np.random.seed(conf.random_seed)
    tf.random.set_seed(conf.random_seed)
    os.environ['PYTHONHASHSEED']=str(conf.random_seed)
    random.seed(conf.random_seed)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    if parsed_args.mode == conf.param_tr:
        relative_experiment_path = os.path.join(conf.output_dirname, timestamp+f'_seed{conf.random_seed}')
        experiment_dir = os.path.join(root_dir, relative_experiment_path)
        os.mkdir(experiment_dir)

        """ Setup logging to file and console """
        logging.basicConfig(level=logging.INFO,
                            format=conf.log_format,
                            datefmt=conf.log_date_fmt,
                            filename=os.path.join(experiment_dir, "out.log"),
                            filemode='w')

        setup_global_logging_stream(conf)
        trainer.run(experiment_dir, root_dir, relative_experiment_path)

    
    elif parsed_args.mode == conf.param_latex:
        setup_global_logging_stream(conf)
        report_root = os.path.join(root_dir, conf.report_dirname)
        res_dir = os.path.join(root_dir, conf.output_dirname)
        list_of_exp_paths = util.find_files(os.path.join(res_dir, '*'))
        fig_params = [{"name" : conf.model_struct_fname,
                    "width" : 0.4,
                    "caption" : "Network model for %s %s",
                    "preface" : "Figure %s presents the model structure for experiment %s and model %s.\n"}
        ]

        reporter.generate_latex_report(res_dir, 
                                        report_root, 
                                        list_of_exp_paths, 
                                        "Model_Weight_#*", 
                                        conf,
                                        parsed_args.n, False, timestamp, fig_params)


if __name__=="__main__":
    start(sys.argv)