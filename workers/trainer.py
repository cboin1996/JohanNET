import logging
import os
from agent import model

from src import config, util
log = logging.getLogger(__name__)
def run(experiment_dir, root_dir):
    """Launches the trainer worker.

    Args:
        experiment_dir (str): the absolute path to the folder used to output training results
    """
    log.info("Trainer launched successfully.")
    
    conf = config.Default()
    processed_df = util.initialize_dataframe(os.path.join(root_dir, conf.rel_path_to_raw_data), 
                                            os.path.join(root_dir, conf.rel_path_to_proc_data),
                                            index_col='rowid')
    print(processed_df.head())
    