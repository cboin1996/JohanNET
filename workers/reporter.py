from src import util
import pandas as pd
import os
import shutil
import glob

def calc_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

def calc_f1_for_matrix(fpath, prec_col, rec_col, f1_col, drop_cols, index=None, index_name=None):
    conf_matrix = util.read_csv(fpath)
    if index is not None:
        conf_matrix = conf_matrix.set_index(pd.Index(index))
        conf_matrix.index.name = index_name
    conf_matrix = conf_matrix.drop(columns=drop_cols)
    conf_matrix[f1_col] = calc_f1(conf_matrix[prec_col], conf_matrix[rec_col])
    return conf_matrix


def query_models_by_f1(ascending, limit, data_paths, metric, acc_col, prec_col, rec_col, model_col, f1_col, drop_cols=None, index=None, index_name=None):
    """Sort results in folders by a value within a csv in ascending or descending order

    Args:
        ascending (bool): true means sort by ascending values
        limit (int): number of results to return
        data_paths (list) : paths to the csv files
        metric (str) : the metric for evaluating a confusion matrix's success
        acc_col (str) : the column of the confusion matrix's accuracy
        prec_col (str) : the column of the confusion matrix's precision
        rec_col (str) : the column of the confusion matrix's recall
        model_col (str) : the column of the model name
        drop_cols (str) : the columns to drop from the csv
    
    Returns:
        pd.DataFrame, list : the dataframe of the query results, the list of model names in order specified 'ascending == True/False'
    """
    for i, fpath in enumerate(data_paths):
        if i == 0:
            df = calc_f1_for_matrix(fpath, prec_col, rec_col, f1_col, drop_cols, index= index, index_name=index_name)
        else:
            df_new = calc_f1_for_matrix(fpath, prec_col, rec_col, f1_col, drop_cols, index= index, index_name=index_name)
            df = df.append(df_new)
    
    df = df.set_index(model_col)
    if metric == 'average':
        df = df.groupby([model_col]).mean()
    
    else:
        raise ValueError('only average performance is implemented for "metric"')

    df = df.sort_values(by=[f1_col], ascending=ascending)[:limit] # slice the df selecting the values constrained by the limit

    return df, list(df.index)


def get_figure_str(fig_width, fig_path, fig_label, fig_caption):
    figure =    """
        \\begin{figure}
        \caption{%s}
        \centering
            \includegraphics[width=%s\linewidth]{%s}
        \label{%s}
        \end{figure}
        """ % (fig_caption, fig_width, fig_path, fig_label)
    return figure

def generate_latex_report(results_dir, output_root, list_of_exp_paths, model_templ, conf, 
                          num_to_report, ascending, report_timestamp, fig_params):
    """Generates a latex report body

    Args:
        output_root (str): The string of the rroto reporting folder
        list_of_exp_paths (str): List of experiments (directories) to scrape info from
        model_templ (str) : the template 'globable' name for the model folder
        conf (config.Default): configuration class
        num_to_report (str) : the number of results to include in the report
        ascending (bool) : True, False ascending or descending
        report_timestamp (str): timestampe for the report folder
        fig_params (dict): paramaters for formatting figures
    """
    conf_matrix_acc_col = conf.conf_matrix_acc_col
    conf_matrix_prec_col = conf.conf_matrix_prec_col
    conf_matrix_rec_col = conf.conf_matrix_rec_col
    conf_matrix_model_col = conf.conf_matrix_model_col
    conf_matrix_drop_cols = conf.conf_matrix_drop_cols
    conf_matrix_fname = conf.conf_matrix_fname
    conf_matrix_index_name = conf.conf_matrix_index_name
    conf_matrix_index = conf.conf_matrix_index
    conf_matrix_f1_col = conf.conf_matrix_f1_col
    h_pars_descriptions = conf.h_pars_descriptions
    h_pars_header = conf.h_pars_header
    h_pars_fname = conf.h_pars_fname
    hyp_drop_cols = conf.hyp_drop_cols

    report_dir = os.path.join(output_root, report_timestamp)
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)
    report_txt_fname = 'latex_results.tex'
    report_path = os.path.join(report_dir, report_txt_fname)
    with open(report_path, 'a') as f:
        f.write('\section{Results}\n')
        f.write('\subsection{Parameter Descriptions}\n')
        f.write("Table \\ref{tab:hyplegend} presents the hyperparameters that were tuned during the validation training of the model.")
        f.write(pd.DataFrame({'Hyperparameter' : h_pars_descriptions.keys(), "Description" : h_pars_descriptions.values()}).to_latex(caption="Hyperparameter Legend", label="tab:hyplegend"))
        all_model_metric_paths = util.find_files(os.path.join(results_dir, '*', model_templ, conf_matrix_fname))

        conf_result_df, queried_models = query_models_by_f1(ascending, num_to_report, all_model_metric_paths, 'average', conf_matrix_acc_col, 
                                                        conf_matrix_prec_col, conf_matrix_rec_col, conf_matrix_model_col, conf_matrix_f1_col, conf_matrix_drop_cols,
                                                        conf_matrix_index, conf_matrix_index_name)
        f.write('\subsection{Summary of Analysis}\n')
        if len(list_of_exp_paths) == 1:
            f.write("An experiment was conducted including the training and validation of many models using a single random seed value. For the experiment ")
        else:
            f.write("An experiment was conducted including the training and validation of many models using a variety of random seeds. For all experiments ")

        f.write("each model was compared based on the top average f1 score calculated from its confusion matrix across three thresholds: 0.3, 0.5 and 0.7. ")
        f.write("The 5 models with the highest average f1 score were selected as the `best'. The averaged confusion matrices for the top 5 models are presented in Table \\ref{tab:summary_confmatrix}, below.")
        f.write(conf_result_df.to_latex(caption=f"Top 5 Model Confusion Matrices Selected by Highest Average f1 Score Across Thresholds {conf_matrix_index}", label=f"tab:summary_confmatrix"))

        for dir_ in list_of_exp_paths: # iterate experiments
            dir_name = os.path.basename(os.path.normpath(dir_))
            latexify_dir_name = util.latexify(dir_name)
            f.write('\subsection{Experiment %s}\n' % (latexify_dir_name))
            report_exp_dir = os.path.join(report_dir, dir_name)
            os.mkdir(report_exp_dir)

            for model_name in queried_models: # iterate the models in each experiment
                rep_exp_model_dir = os.path.join(report_exp_dir, util.strip_illegal_chars(model_name))
                os.mkdir(rep_exp_model_dir)
                rep_exp_model_name = dir_name + model_name
                model_fpath = os.path.join(dir_, model_name)
                latexify_model_name = util.latexify(model_name)
                f.write('\subsubsection{Model %s}\n' % (latexify_model_name))
                # generate figures
                for fig_map in fig_params: 
                    fig_name = fig_map['name']
                    fig_src = os.path.join(model_fpath, fig_name)
                    fig_dest = os.path.join(rep_exp_model_dir, fig_name)
                    fig_relative_path = dir_name + "/" + util.strip_illegal_chars(model_name) + "/" + fig_name
                    shutil.copy(fig_src, fig_dest)
                    f.write(fig_map['preface'] % ("\\ref{fig:" + util.strip_illegal_chars(fig_relative_path) + "}", latexify_dir_name, latexify_model_name))
                    f.write(get_figure_str(fig_map['width'], fig_relative_path, f"fig:{util.strip_illegal_chars(fig_relative_path)}", fig_map["caption"] % (latexify_dir_name, latexify_model_name)))

                # generate confusion matrix table
                model_conf_matrix_fpath = os.path.join(model_fpath, conf_matrix_fname)
                model_conf_matrix = calc_f1_for_matrix(model_conf_matrix_fpath, conf_matrix_prec_col, conf_matrix_rec_col, conf_matrix_f1_col, conf_matrix_drop_cols, 
                                                        index=conf_matrix_index, index_name=conf_matrix_index_name)
                model_conf_matrix = model_conf_matrix.drop(columns=[conf_matrix_model_col])
                f.write("Table \\ref{tab:conf_matr%s} presents the confusion matrix for experiment %s and model %s.\n" % (util.strip_illegal_chars(rep_exp_model_name), latexify_dir_name, latexify_model_name))
                f.write(model_conf_matrix.to_latex(caption=f"Confusion Matrix for {latexify_dir_name} {latexify_model_name}", label=f"tab:conf_matr{util.strip_illegal_chars(rep_exp_model_name)}"))

                # output hyperparameters
                hyp_df = util.read_csv(os.path.join(model_fpath, h_pars_fname))
                hyp_df = hyp_df.drop(columns = hyp_drop_cols)
                hyp_df.columns = h_pars_header
                f.write("Table \\ref{tab:hyp%s} presents the hyperparameters used for experiment %s and model %s.\n" % (util.strip_illegal_chars(rep_exp_model_name), latexify_dir_name, latexify_model_name))
                f.write(hyp_df.to_latex(caption=f"Hyperparameter's for {latexify_dir_name} {latexify_model_name}", label=f"tab:hyp{util.strip_illegal_chars(rep_exp_model_name)}"))
        
    print(f"Completed generating your report in {report_path}")