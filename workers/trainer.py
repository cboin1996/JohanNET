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
                                            index_col='rowid',
                                            drop_cols=conf.raw_data_drop_cols)
    print(processed_df.head())
<<<<<<< HEAD
    
=======
    
    
>>>>>>> 9d61a3cc1be85527995eab311d6495b6cc81d9a7
