import os 
class Default:
    output_dirname = os.path.join('.outputs')
    config_json_fname = "conf.json"

    def __init__(self):
        """IO configuration"""
        self.output_dirname = self.output_dirname
        self.report_dirname = "reports"
        self.data_dirname = 'data'
        self.dirs = [self.output_dirname, self.data_dirname, self.report_dirname]

        self.config_json_fname = self.config_json_fname
        self.raw_data_fname = "raw_data.csv"
        self.proc_data_fname = "processed_data.csv"

        self.model_struct_fname = "model_struct.png"
        self.conf_matrix_fname = "confusion_matrix.csv"
        self.h_pars_fname = "hyper_params.csv"
        self.rel_path_to_raw_data = os.path.join(self.data_dirname, self.raw_data_fname)
        self.rel_path_to_proc_data = os.path.join(self.data_dirname, self.proc_data_fname)

        self.train_feat_fname = 'train_features.csv'
        self.train_label_fname = 'train_labels.csv'
        self.val_feat_fname  = 'validation_features.csv'
        self.val_label_fname  = 'validation_labels.csv'
        self.test_feat_fname = 'test_features.csv'
        self.test_label_fname = 'test_labels.csv'

        """Data"""
        self.train_data_split_factor = 0.8
        self.valid_data_split_factor = 0
        self.raw_data_drop_cols = ['kepoi_name', 'kepler_name', 'koi_teq_err1', 'koi_teq_err2', 'koi_tce_delivname','koi_disposition', 'koi_fpflag_nt','koi_score','koi_fpflag_ss','koi_fpflag_co','koi_fpflag_ec', 'kepid']

        self.data_label_colnames = ['koi_pdisposition']
        self.data_feature_colnames = ['koi_period' ,'koi_period_err1' ,'koi_period_err2',
                                       'koi_time0bk','koi_time0bk_err1','koi_time0bk_err2',
                                       'koi_impact','koi_impact_err1' ,'koi_impact_err2',
                                      'koi_duration','koi_duration_err1','koi_duration_err2',
                                      'koi_depth','koi_depth_err1','koi_depth_err2',
                                      'koi_prad','koi_prad_err1','koi_prad_err2',
                                      'koi_teq',
                                      'koi_insol','koi_insol_err1','koi_insol_err2',
                                      'koi_model_snr','koi_tce_plnt_num',
                                      'koi_steff','koi_steff_err1','koi_steff_err2',
                                      'koi_slogg','koi_slogg_err1','koi_slogg_err2',
                                      'koi_srad','koi_srad_err1','koi_srad_err2','ra','dec','koi_kepmag']
        # used to define input layer structure in the model
        self.colNames = [['koi_period' ,'koi_period_err1' ,'koi_period_err2'],
                    ['koi_time0bk','koi_time0bk_err1','koi_time0bk_err2'],
                    ['koi_impact','koi_impact_err1' ,'koi_impact_err2'],
                    ['koi_duration','koi_duration_err1','koi_duration_err2'],
                    ['koi_depth','koi_depth_err1','koi_depth_err2'],
                    ['koi_prad','koi_prad_err1','koi_prad_err2'],
                    ['koi_insol','koi_insol_err1','koi_insol_err2'],
                    ['koi_steff','koi_steff_err1','koi_steff_err2'],
                    ['koi_slogg','koi_slogg_err1','koi_slogg_err2'],
                    ['koi_srad','koi_srad_err1','koi_srad_err2'],
                    ['ra','dec','koi_kepmag'],
                    ['koi_teq','koi_model_snr','koi_tce_plnt_num']]

        """Argument Parsing"""
        self.param_tr   = 'tr'
        self.param_pred = 'pred'
        self.param_latex = 'latex'
        self.cmdl_choices = [self.param_tr, 
                             self.param_pred,
                             self.param_latex]       
        """Model Hyperparameters"""
        self.random_seed = 1
            
        # hyperparameters for tuning with talos
        self.h_pars ={
            'layer_type':   ['dense'],
            'layer2_units': [36], #, 24, 12
            'layer3_units': [12], #, 6, 3
            'activation':   ['relu'], #, 'lrelu', 'sigmoid'
            'optimizer':    ['Adam'],
            'loss':         ['mean_squared_error', 'binary_crossentropy'],
            'dropout':      [0, 0.1, 0.2],

            #Passing config variables
            'experiment_dir' : [],
            'col_dense' : [],
            'col_conv' : [],
        }

        self.conf_matrix_acc_col = 'accuracy'
        self.conf_matrix_prec_col = 'precision'
        self.conf_matrix_rec_col = 'recall'
        self.conf_matrix_model_col = 'model_no'
        self.conf_matrix_f1_col = 'f1 score'
        self.conf_matrix_drop_cols = ['Unnamed: 0']
        self.conf_matrix_index = [0.3, 0.5, 0.7]
        self.conf_matrix_index_name = 'threshold'

        self.h_pars_descriptions = {
            'layer_type': "the type of DNN layer used in the experiment",
            'layer2_units': "number of nodes in the 1st hidden layer",
            'layer3_units': "number of nodes in the 2nd hidden layer",
            'activation':   "activation function used for layers in the model",
            'optimizer':    "optimizer used while training",
            'loss':         "the loss metric used in training",
            'dropout':      "the dropout used in training"
        }

        self.hyp_drop_cols = ["Unnamed: 0"]
        self.h_pars_header = ["Hyperparameter", "Value"]
        """Logging"""
        self.log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
        self.log_date_fmt = "%y-%m-%d %H:%M:%S"


