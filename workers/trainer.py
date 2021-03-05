import logging
import os
log = logging.getLogger(__name__)
def run(experiment_dir, root_dir):
    """Launches the trainer worker.

    Args:
        experiment_dir (str): the absolute path to the folder used to output training results
    """
    log.info("Trainer launched successfully.")
    
    conf = config.Default()
    raw_data_path = os.path.join(root_dir, conf.data_fname)

    
