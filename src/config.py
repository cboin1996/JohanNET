import os 
class Default:
    output_dirname = os.path.join('.outputs')
    config_json_fname = "conf.json"

    def __init__(self):
        """IO configuration"""
        self.output_dirname = self.output_dirname
        self.data_dirname = 'data'
        self.dirs = [self.output_dirname, self.data_dirname]

        self.config_json_fname = self.config_json_fname
        self.raw_data_fname = "raw_data.csv"
        self.proc_data_fname = "processed_data.csv"
        self.rel_path_to_raw_data = os.path.join(self.data_dirname, self.raw_data_fname)
        self.rel_path_to_proc_data = os.path.join(self.data_dirname, self.proc_data_fname)

        """Argument Parsing"""
        self.param_tr   = 'tr'
        self.param_pred = 'pred'
        self.cmdl_choices = [self.param_tr, 
                             self.param_pred]       
        """Model Hyperparameters"""
        self.random_seed = 1


        """Logging"""
        self.log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
        self.log_date_fmt = "%y-%m-%d %H:%M:%S"


