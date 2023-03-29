import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import warnings
from matplotlib.ticker import FormatStrFormatter
from activation_utils import compute_mean_digit_separations_domainIL, compute_mean_digit_separation_classIL, get_distances

def preprocess_performance_data(results, results_bp, results_ewc, results_si, results_l2, CL_MODE, MODELS, EVAL_METHOD):

    interesting_cols = [
        'task_test_accu',
        'task_train_accu',
        'lr',
        'use_recurrent_weights',
        'lr_rec',
        'layer_max_sparsities',
        'size_hidden',
        'freeze_suppressed_neuron_weights',
        'maintain_total_activity',
        'rec_grad_normalization',
        'fw_grad_normalization',
        'rec_learning_neurons',
        'random_seed',
        'permanent_sparsification',
        'rec_inference_iterations',
        'stop_early_at_accu',
        'sparsity_level_function',
        'turn_off_rec_norm_normalization',
        'hebbian_fw_learning',
        'hidden_activation',
        'from_ff_learning',
        'frac_rec_deciding_sparsity',
        'network_type',
        'reg_coef'
    ]
    interesting_cols_normal = results.columns[results.columns.isin(interesting_cols)]
    if not results_bp is None:
        interesting_cols_bp = results_bp.columns[results_bp.columns.isin(interesting_cols)]
    if not results_ewc is None:
        interesting_cols_ewc = results_ewc.columns[results_ewc.columns.isin(interesting_cols)]
    if not results_si is None:
        interesting_cols_si = results_si.columns[results_si.columns.isin(interesting_cols)]
    if not results_l2 is None:
        interesting_cols_l2 = results_l2.columns[results_l2.columns.isin(interesting_cols)]

    results = results.sort_values(by='task_test_accu_last', ascending=False)[interesting_cols_normal]
    if not results_bp is None:
        results_bp = results_bp.sort_values(by='task_test_accu_last', ascending=False)[interesting_cols_bp]
    if not results_ewc is None:
        results_ewc = results_ewc.sort_values(by='task_test_accu_last', ascending=False)[interesting_cols_ewc]
    if not results_si is None:
        results_si = results_si.sort_values(by='task_test_accu_last', ascending=False)[interesting_cols_si]
    if not results_l2 is None:
        results_l2 = results_l2.sort_values(by='task_test_accu_last', ascending=False)[interesting_cols_l2]

    no_sparsity = re.sub('[^0,]', '', results.iloc[0]['layer_max_sparsities'])


    ### Create column for comparisons
    results['mode'] = ""
    results['mode'] = results['mode'].astype(str)

    # Non-sparse, non-recurrent runs
    mask = (~results['use_recurrent_weights']) & (results['layer_max_sparsities'] == no_sparsity)
    results.loc[mask, 'mode'] = 'dfc-standard'

    # Sparse, non-recurrent runs
    mask = (~results['use_recurrent_weights']) & (results['layer_max_sparsities'] != no_sparsity)
    results.loc[mask, 'mode'] = 'dfc-sparse'

    # Non-sparse, recurrent runs
    mask = (results['use_recurrent_weights']) & (results['layer_max_sparsities'] == no_sparsity)
    results.loc[mask, 'mode'] = 'dfc-rec'

    # Sparse and recurrent runs
    mask = (results['use_recurrent_weights']) & (results['layer_max_sparsities'] != no_sparsity)
    results.loc[mask, 'mode'] = 'dfc-sparse-rec'

    if 'permanent_sparsification' in results.columns:
        mask = results['permanent_sparsification']
    else:
        mask = (results['size_hidden'] == "20,4")
    results.loc[mask, 'mode'] = 'permanent-sparsification'



    # BP
    if not results_bp is None:
        results_bp['mode'] = 'bp'
        results = results.append(results_bp)

    # EWC
    if not results_ewc is None:
        results_ewc['mode'] = 'ewc-' + results_ewc['reg_coef'].astype(str)
        results = results.append(results_ewc)

    # L2
    if not results_l2 is None:
        results_l2['mode'] = 'l2-' + results_l2['reg_coef'].astype(str)
        results = results.append(results_l2)


    # SI
    if not results_si is None:
        results_si['mode'] = 'si-' + results_si['reg_coef'].astype(str)
        results = results.append(results_si)

    return results, interesting_cols


def capitalize_label(name):
    name_list = name.split('-')
    name_list[0] = name_list[0].upper()
    return '-'.join(name_list)

def plot_sparsity_mix(ax, lrs, set_ylabel=False, title='', cl_mode='domain', frac_rec_deciding_sparsity=1, eval_set='test', eval_idx=-1):

    line_fmt='-o'
    markersize=5
    capsize=5
    if cl_mode == 'domain':
        results_mix = (pd.read_csv(os.getcwd() + '/../out/hpsearches-final/hpconfig_domain-mnist-sparse-rec-sparsity-deciding-mix/search_results.csv', delimiter=';'))
    else:
        results_mix = (pd.read_csv(os.getcwd() + '/../out/hpsearches-final/hpconfig_class-mnist-sparse-rec-sparsity-deciding-mix/search_results.csv', delimiter=';'))


    interesting_cols_mix = [
        'task_test_accu',
        'task_train_accu',
        'lr',
        'use_recurrent_weights',
        'lr_rec',
        'layer_max_sparsities',
        'size_hidden',
        'freeze_suppressed_neuron_weights',
        'maintain_total_activity',
        'rec_grad_normalization',
        'fw_grad_normalization',
        'rec_learning_neurons',
        'rec_adaptation',
        'random_seed',
        'permanent_sparsification',
        'block_non_sparsified_neurons',
        'rec_inference_iterations',
        'frac_fb_deciding_sparsity',
        'frac_rec_deciding_sparsity'
        
    ]
    interesting_cols_mix = results_mix.columns[results_mix.columns.isin(interesting_cols_mix)]
    results_mix = results_mix.sort_values(by='task_test_accu_last', ascending=False)[interesting_cols_mix]
    size = results_mix.iloc[0]['size_hidden']

    results_mix['eval_metric'] = results_mix[f'task_{eval_set}_accu'].apply(lambda x: float(eval(x).split(',')[eval_idx]))

    not_groupby_cols_mix = ['task_test_accu', 'random_seed', 'lr_rec', 'task_train_accu',
                        'rec_learning_neurons', 'rec_grad_normalization', 'permanent-sparsification',
                        'rec_inference_iterations']


    groupby_cols_mix = [x for x in interesting_cols_mix if x not in not_groupby_cols_mix]
    colors = plt.cm.Greys(np.linspace(0,1,7))
    for idx, lr_exp in enumerate(lrs):
        for use_recurrent_weights in [True]:
            for frac_rec_deciding_sparsity in [frac_rec_deciding_sparsity]:

                selected_results = results_mix[
                        (results_mix['size_hidden'] == size) &
                        (results_mix['frac_rec_deciding_sparsity'] == frac_rec_deciding_sparsity) &
                        #(results['use_recurrent_weights'] == use_recurrent_weights) &
                        (np.abs(results_mix['lr'] - 1 / (10 ** lr_exp)) < 1e-8)
                    ]

                grouped = selected_results.groupby(by=groupby_cols_mix)
                means = grouped.mean().reset_index()
                stds = grouped.std().reset_index()
                ax.errorbar(means['frac_fb_deciding_sparsity'], means[f'eval_metric'],
                            yerr=stds['eval_metric'], label=f"1e-{lr_exp}", color=colors[idx+2],
                            fmt=line_fmt, capsize=capsize, markersize=markersize)

    ax.legend(title='Learning rate')
    if set_ylabel:
        ax.set_ylabel('Accuracy')
    ax.set_xlabel('fraction of activity obtained from feedback')
    ax.set_title(title)
    ax.set_xticks([0, 0.5, 1], [0, 0.5, 1])

def plot_lr_sweep(ax, results, results_bp, results_ewc, results_si, results_l2,
                  interesting_cols, modes_to_plot, CL_MODE, MODELS, EVAL_METHOD,
                  line_fmt='-o', markersize=5, capsize=5, red_mode='dfc-sparse-rec',
                  title='', fontsize=14, display_legend=False, legend_loc=None,
                  yticks=None, xticks=None, eval_set='test', eval_idx=-1):
    
    not_groupby_cols = ['task_test_accu', 'random_seed', 'lr_rec', 'task_train_accu',
                        'rec_learning_neurons', 'rec_grad_normalization', 'permanent-sparsification',
                        'rec_inference_iterations', 'fw_grad_normalization', 'turn_off_rec_norm_normalization',
                        'hebbian_fw_learning', 'sparsity_level_function', 'frac_rec_deciding_sparsity', 'from_ff_learning',
                       'block_non_sparsified_neurons', 'network_type', 'reg_coef']

    x_axis_value = 'lr' if EVAL_METHOD == 'LR' else 'stop_early_at_accu'

    if EVAL_METHOD == 'LR':
        not_groupby_cols.append('stop_early_at_accu')

    groupby_cols = [x for x in interesting_cols if x not in not_groupby_cols]

    plotted_curves = []
    for mode in modes_to_plot:

        selected_results = results[
                (results['mode'] == mode)
            ]

        selected_results['lr'] = selected_results['lr'].astype(float)
        selected_results['eval_metric'] = selected_results[f'task_{eval_set}_accu'].apply(lambda x: float(eval(x).split(',')[eval_idx]))


        if mode.startswith('bp'):
            group_by_cols_curr = [x for x in groupby_cols if x in results_bp.columns]
        elif mode.startswith('ewc'):
            group_by_cols_curr = [x for x in groupby_cols if x in results_ewc.columns]
        elif mode.startswith('si'):
            group_by_cols_curr = [x for x in groupby_cols if x in results_si.columns]
        elif mode.startswith('l2'):
            group_by_cols_curr = [x for x in groupby_cols if x in results_l2.columns]
        else:
            group_by_cols_curr = groupby_cols

        grouped = selected_results.groupby(by=group_by_cols_curr)
        means = grouped.mean().reset_index()

        stds = grouped.std().reset_index()
        ax.errorbar(means[x_axis_value], means['eval_metric'],
                     yerr=stds['eval_metric'], label=capitalize_label(mode), fmt=line_fmt, markersize=markersize, capsize=capsize,
                            color=('red' if mode==red_mode else None))

        plotted_curves.append((means['eval_metric'], stds['eval_metric'], mode))
    
    if display_legend:
        ax.legend(prop={'size': fontsize}, loc=legend_loc)

    ax.set_ylabel(f'Accuracy ({CL_MODE.capitalize()}-IL)', fontsize=fontsize)
    ax.set_title(title)

    if x_axis_value == 'lr':
        plt.xscale('log')
        plt.xlabel('Learning Rate (4 Epochs)', fontsize=fontsize)

        if CL_MODE == 'class' and EVAL_METHOD == 'LR':
            ax.set_xlim(1 / 10**(5.6),1 / 10**(0.9))
        elif CL_MODE == 'domain' and EVAL_METHOD == 'LR':
            ax.set_xlim(1 / 10**(5.1),1 / 10**(0.9))
    else:
        ax.set_xlabel('Early Stop Accuracy', fontsize=fontsize)
        
    if yticks is not None:
        ax.set_yticks(yticks, yticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if xticks is not None:
        ax.set_xticks(xticks, xticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
    return plotted_curves

def plot_peak_aligned_lr_sweep(ax, plotted_curves, CL_MODE, title='',
                              line_fmt='-o', markersize=5, capsize=5, red_mode='dfc-sparse-rec',
                               min_accu=None, max_length=999, fontsize=14):
    """
    Plots performance across LRs. The curves are aligned such that they start with either
    peak performance, or the first value which surpasses `min_accu`, if provided.
    """
    plt.style.use('grayscale')
    plt.gcf().patch.set_facecolor('white')

    min_length = max_length
    for means, stds, mode in plotted_curves:
        if min_accu:
            max_idx = np.where(np.array(means) > min_accu)[0][0]
        else:
            max_idx = np.argmax(means)
        len_after_peak = len(means[max_idx:])
        ax.errorbar(range(len_after_peak), means[max_idx:],
                     yerr=stds[max_idx:], label=capitalize_label(mode), fmt=line_fmt,
                     markersize=markersize, capsize=capsize, color=('red' if mode==red_mode else None))
        min_length = min(len_after_peak, min_length)

    for means, stds, mode in plotted_curves:
        if min_accu:
            max_idx = np.where(np.array(means) > min_accu)[0][0]
        else:
            max_idx = np.argmax(means)
        print(mode, 'mean of selected range:', np.mean(means[max_idx:max_idx+min_length]))
        
    ax.set_xlim(-0.1, min_length - 0.9)
    if min_accu:
        ax.set_xlabel(f'#LR steps after accu={min_accu}', fontsize=fontsize)
    else:
        ax.set_xlabel('# LR steps after peak performance', fontsize=fontsize)
    ax.set_title(title)
    ax.set_xticks(range(min_length),  # ensure int x axis
                  range(min_length))
    ax.set_ylabel(f'Accuracy ({CL_MODE.capitalize()}-IL)', fontsize=fontsize)
    ax.legend(prop={'size': fontsize})

def load_performance_data(CL_MODE, MODELS, EVAL_METHOD, subdir=''):
    """
    Loads performance data for given configuration.
    """
    results = None
    results_bp = None
    results_ewc = None
    results_si = None
    results_l2 = None

    subdir_str = '.' if subdir == '' else subdir

    if CL_MODE == 'domain' and MODELS == 'CLB' and EVAL_METHOD == 'LR':
        modes_to_plot = ['bp', 'ewc-100', 'si-1', 'dfc-sparse-rec']
        results = pd.read_csv(os.getcwd() + f'/../out/hpsearches-final/{subdir_str}/hpconfig_domain-mnist-sparse-rec/search_results.csv', delimiter=';')
        results_bp = pd.read_csv(os.getcwd() + f'/../out/hpsearches-final/{subdir_str}/hpconfig_domain-split-mnist-bp/search_results.csv', delimiter=';')
        results_ewc = pd.read_csv(os.getcwd() + f'/../out/hpsearches-final/{subdir_str}/hpconfig_domain-split-mnist-ewc/search_results.csv', delimiter=';')
        results_si = pd.read_csv(os.getcwd() + f'/../out/hpsearches-final/{subdir_str}/hpconfig_domain-split-mnist-si/search_results.csv', delimiter=';')
    elif CL_MODE == 'domain' and MODELS == 'DFC' and EVAL_METHOD == 'LR':
        modes_to_plot = ['dfc-standard', 'dfc-rec', 'dfc-sparse', 'dfc-sparse-rec']
        results = pd.read_csv(os.getcwd() + f'/../out/hpsearches-final/{subdir_str}/hpconfig_domain-mnist-sparse-rec-all-variants/search_results.csv', delimiter=';')
    elif CL_MODE == 'domain' and MODELS == 'CLB' and EVAL_METHOD == 'MIN_ACCU':
        modes_to_plot = ['bp', 'ewc-100', 'si-10', 'dfc-sparse-rec']
        results = pd.read_csv(os.getcwd() + f'/../out/hpsearches-final/{subdir_str}/hpconfig_domain-mnist-sparse-rec-min-accu/search_results.csv', delimiter=';')
        results_bp = pd.read_csv(os.getcwd() + f'/../out/hpsearches-final/{subdir_str}/hpconfig_domain-split-mnist-bp-min-accu/search_results.csv', delimiter=';')
        results_ewc = pd.read_csv(os.getcwd() + f'/../out/hpsearches-final/{subdir_str}/hpconfig_domain-split-mnist-ewc-min-accu/search_results.csv', delimiter=';')
        results_si = pd.read_csv(os.getcwd() + f'/../out/hpsearches-final/{subdir_str}/hpconfig_domain-split-mnist-si-min-accu/search_results.csv', delimiter=';')
    elif CL_MODE == 'class' and MODELS == 'CLB' and EVAL_METHOD == 'LR':
        modes_to_plot = ['bp', 'ewc-10000', 'si-100', 'dfc-sparse-rec']
        results = pd.read_csv(os.getcwd() + f'/../out/hpsearches-final/{subdir_str}/hpconfig_class-mnist-sparse-rec/search_results.csv', delimiter=';')
        results_bp = pd.read_csv(os.getcwd() + f'/../out/hpsearches-final/{subdir_str}/hpconfig_class-split-mnist-bp/search_results.csv', delimiter=';')
        results_ewc = pd.read_csv(os.getcwd() + f'/../out/hpsearches-final/{subdir_str}/hpconfig_class-split-mnist-ewc/search_results.csv', delimiter=';')
        results_si = pd.read_csv(os.getcwd() + f'/../out/hpsearches-final/{subdir_str}/hpconfig_class-split-mnist-si/search_results.csv', delimiter=';')
    elif CL_MODE == 'class' and MODELS == 'DFC' and EVAL_METHOD == 'LR':
        modes_to_plot = ['dfc-standard', 'dfc-rec', 'dfc-sparse', 'dfc-sparse-rec']
        results = pd.read_csv(os.getcwd() + f'/../out/hpsearches-final/{subdir_str}/hpconfig_class-mnist-sparse-rec-all-variants/search_results.csv', delimiter=';')
    elif CL_MODE == 'class' and MODELS == 'CLB' and EVAL_METHOD == 'MIN_ACCU':
        modes_to_plot = ['bp', 'ewc-100000', 'si-100', 'dfc-sparse-rec']
        results = pd.read_csv(os.getcwd() + f'/../out/hpsearches-final/{subdir_str}/hpconfig_class-mnist-sparse-rec-min-accu/search_results.csv', delimiter=';')
        results_bp = pd.read_csv(os.getcwd() + f'/../out/hpsearches-final/{subdir_str}/hpconfig_class-split-mnist-bp-min-accu/search_results.csv', delimiter=';')
        results_ewc = pd.read_csv(os.getcwd() + f'/../out/hpsearches-final/{subdir_str}/hpconfig_class-split-mnist-ewc-min-accu/search_results.csv', delimiter=';')
        results_si = pd.read_csv(os.getcwd() + f'/../out/hpsearches-final/{subdir_str}/hpconfig_class-split-mnist-si-min-accu/search_results.csv', delimiter=';')
        
    return results, results_bp, results_ewc, results_si, results_l2, modes_to_plot

def plot_performance(CL_MODE, MODELS, EVAL_METHOD, FIG_DIR, FIG_SIZE, ylim=None, fontsize=14, display_legend=False, legend_loc=None, xticks=None, yticks=None, subdir='', eval_set='test', eval_idx=-1):
    """
    Plots performance results across LRs or minimum accuracies, according to argument configuration.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results, results_bp, results_ewc, results_si, results_l2, modes_to_plot = load_performance_data(CL_MODE, MODELS, EVAL_METHOD, subdir=subdir)
        results, interesting_cols = preprocess_performance_data(results, results_bp, results_ewc, results_si, results_l2, CL_MODE, MODELS, EVAL_METHOD)

        plt.rcParams["figure.figsize"] = (FIG_SIZE[0], FIG_SIZE[1])
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.patch.set_facecolor('white')
        plotted_curves = plot_lr_sweep(ax, results, results_bp, results_ewc, results_si, results_l2,
                                       interesting_cols, modes_to_plot, CL_MODE, MODELS, EVAL_METHOD,
                                       title='', fontsize=fontsize, display_legend=display_legend,
                                       legend_loc=legend_loc, xticks=xticks, yticks=yticks, eval_set=eval_set,
                                       eval_idx=eval_idx)
        if ylim:
            plt.ylim(ylim)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        fig.savefig(f'{FIG_DIR}{CL_MODE=}-{MODELS=}-{EVAL_METHOD=}.svg', format='svg', bbox_inches = "tight")
        
        # Aligned plot
        if EVAL_METHOD == 'LR':
            fig, ax = plt.subplots(nrows=1, ncols=1)
            plot_peak_aligned_lr_sweep(ax, plotted_curves, CL_MODE, title='', min_accu=0.75 if CL_MODE == 'domain' else 0.3, max_length=6)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            fig.savefig(f'{FIG_DIR}{CL_MODE=}-{MODELS=}-{EVAL_METHOD=}-LR_aligned.svg', format='svg', bbox_inches = "tight")

def plot_normalized_separation_domainIL(name, ax, num_seeds, tgt_overlap_vals, task_overlap_vals, lrs, red_mode='dfc-sparse-rec'):
    """
    Plots difference between inter- and intra-label separation across LRs, averaged over random seeds.
    """
    
    separation_mean_diffs, _, _ = compute_mean_digit_separations_domainIL(name, num_seeds, tgt_overlap_vals, task_overlap_vals)
    separation_diff_means_means = np.mean(separation_mean_diffs, axis=0)
    separation_diff_means_stds = np.std(separation_mean_diffs, axis=0)

    ax.errorbar(lrs, separation_diff_means_means, yerr=separation_diff_means_stds, label=capitalize_label(name),
                fmt='-o', markersize=5, capsize=5, color=('red' if name==red_mode else None))
    return np.array(separation_mean_diffs)

def plot_digit_separation_classIL(name, ax, lrs, all_overlap_vals, red_mode='dfc-sparse-rec'):
    """
    Inter-digit representational separation across LRs, averaged over random seeds.
    """
    
    all_overlap_means = compute_mean_digit_separation_classIL(name, 5, all_overlap_vals)
    
    all_overlap_means_means = np.mean(all_overlap_means, axis=0)
    all_overlap_means_stds = np.std(all_overlap_means, axis=0)

    ax.errorbar(lrs, all_overlap_means_means, yerr=all_overlap_means_stds, label=capitalize_label(name),
                fmt='-o', markersize=5, capsize=5,
                color=('red' if name==red_mode else None))


def plot_hyperplane_results(names, distance_means, distance_stds, red_mode='dfc-sparse-rec', ylabel='accuracy', fontsize=13):
    plt.rcParams["figure.figsize"] = (6,4)
    plt.gcf().patch.set_facecolor('white')
    
    chosen_lrs = [1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
    lrs = [10 ** (-neg_exp) for neg_exp in chosen_lrs]

    for name in names:
        plt.errorbar(lrs, distance_means[name], yerr=distance_stds[name], label=capitalize_label(name),
                    fmt='-o', markersize=5, capsize=5,
                    color=('red' if (name==red_mode) else None))

    plt.legend(prop={'size': fontsize})
    plt.gcf().patch.set_facecolor('white')
    plt.xscale('log')
    plt.xlabel('Learning Rate (4 Epochs)', fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)