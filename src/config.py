import os 
class Default:
    output_dirname = os.path.join('.outputs')
    config_json_fname = "conf.json"

    def __init__(self):
        """IO configuration"""
        self.output_dirname = self.output_dirname
        self.config_json_fname = self.config_json_fname
        self.dirs = [self.output_dirname]
        self.data_fname = "cumulative.csv"

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


