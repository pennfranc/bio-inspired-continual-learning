import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import entropy
from sklearn.utils.extmath import randomized_svd
import torch
import os

def mat_to_singular_vals(R, normalize=False, sigmoid=False, n_components=None):
    """
    Returns the vector of singular values of matrix R.
    Optionally matrix R is normalized, or passed through sigmoid before SVD.
    These options are generally not used.
    """
    if normalize:
        R = R / np.linalg.norm(R)
    else:
        R = R
    if sigmoid:
        R = torch.sigmoid(R)
    
    if n_components is None:
        n_components = R.shape[0]

    _, Sigma, _ = randomized_svd(R,
                                 n_components=n_components,
                                 n_iter=5,
                                 random_state=None)
    return Sigma

def effective_dimensionality(R, normalize=False, sigmoid=False, n_components=None):
    """
    Computes the effective dimensionality of matrix R according to 
    https://ieeexplore.ieee.org/abstract/document/7098875.
    """
    Sigma = mat_to_singular_vals(R, normalize, sigmoid, n_components)
    return np.exp(entropy(Sigma))

def cross_linear_separability(arr1, arr2, arr_hyp=None, tgt1=None, tgt2=None):
    """
    Fraction of data points that can be linearly separated in arr2
    by a support vector machine classifier trained on arr1.
    """
    reg = LogisticRegression(penalty='l1', solver='liblinear').fit(arr1, tgt1)
    return reg.score(arr2, tgt2)

def movement_towards_hyperplane_normalized(arr1, arr2, arr_hyp=None, tgt1=None, tgt2=None):
    """
    Computes the movment of activations from arr1 to arr2 in the direction of the hyperplane
    dividing activations arr_hyp according to labels tgt1. 
    
    The way this function is used in our domain-IL movement towards hyperplane
    analysis, arr1 and arr2 represent feedforward activations of when a specific task is 
    first learned and feedforward activations of the same task but after training has
    finished, respectively. arr_hyp represents target activations of when a specific task is
    learned (i.e. they correspond to arr1 activation with recurrent gating and controller effects).

    """
    reg = LogisticRegression(penalty='l1', solver='liblinear').fit(arr_hyp, tgt1)

    # compute hyperplane normal
    normal = reg.coef_ / np.linalg.norm(reg.coef_)

    # compute movement of activations up to end of training
    act_diff_0 = arr1[tgt1 == 0] - arr2[tgt1 == 0]
    act_diff_1 = arr1[tgt2 == 1] - arr2[tgt2 == 1]

    # project movements onto hyperplane normal
    act_diff_proj_0 = np.matmul(normal, np.transpose(act_diff_0))
    act_diff_proj_1 = np.matmul(normal, np.transpose(act_diff_1))

    # average movement towards hyperplane
    act_diff_proj = np.concatenate([-act_diff_proj_0.flatten(), act_diff_proj_1.flatten()])
    mean_movement_towards_hyperplane = np.mean(np.clip(act_diff_proj, 0, None))

    # comptue total movement in any direction
    act_diff_total = abs(arr1 - arr2).mean() / 2 
    mean_movement_total = act_diff_total

    # return normalized movement towards hyperplane
    return mean_movement_towards_hyperplane / mean_movement_total

def get_inter_task_results_arr(name,
                               dist_func=movement_towards_hyperplane_normalized,
                               layer=1,
                               seed=1,
                               combined_activations=None,
                               combined_activations_hyp=None,
                               num_tasks=5,
                               start_task=0,
                               task=None,
                               tgt=None):
    """
    Computes the distance measure given by dist_func between pairs of task-wise activations
    given by combined_activations. In case dist_func requires a hyperplane, combined_activations_hyp
    contains the activations needed to comptue the hyperplane.
    """

    # retrieve relevant activations
    activations = combined_activations[name][layer][seed]
    if combined_activations_hyp:
        activations_hyp = combined_activations_hyp[name][layer][seed]
    else: 
        activations_hyp = activations

    # iterate through pairs of tasks
    result_arr = np.zeros((num_tasks, num_tasks))
    for i in range(start_task, num_tasks):
        for j in range(i, num_tasks):

            # for movement towards hyperplane metric, the distance between a task with itself is 0
            if (i == j) and (dist_func == movement_towards_hyperplane_normalized):
                result_arr[i, j] = 0
                continue

            # compute the distance metric between task i and j activations
            result_arr[i, j] = dist_func(
                activations[(task == i)],
                activations[(task == j)],
                arr_hyp=activations_hyp[(task == i)],
                tgt1=tgt[task == i], tgt2=tgt[task == j])
    return result_arr

def get_distances(name,
                  dist_func=movement_towards_hyperplane_normalized,
                  num_tasks=5,
                  layer=1,
                  num_seeds=5,
                  runs=None,
                  size=None,
                  sparsity=None,
                  lr_rec=None,
                  num_classes_per_task=None,
                  lr=None,
                  activation_type_hyp=None,
                  activation_type='-feedforward',
                  mode='across_time',
                  pure_ff_sparsity=False):

    """
    Computes the distance measure given by dist_func between pairs of task-wise activations
    with respect to different start_task values. start_task determines the 'reference frame'
    that was used to comptue the activations considered. Usually it determines which task
    the network was last trained on, but depending on the `mode` parameter, the meaning can vary.
    See the `load_data()` function for further informations.
    """
    distances = []

    # iterate over reference tasks
    for start_task in range(num_tasks):

        # the last task is usually skipped as a reference task for distance measures
        # this is because we can't compare the distance of the last task to a subsequent task
        if mode in ['across_task_before_learned', 'across_task_first_learned', 'across_time'] and start_task == 4:
            break
        
        # retrieve relevant activations, as well as arrays indicating which tasks and labels (tgt) activations belong to
        _, _, combined_activations, _, _, _, task, tgt, _ = (
            load_data(start_task, mode=mode, layer=layer, runs=runs, size=size, num_seeds=num_seeds, sparsity=sparsity, lr_rec=lr_rec, num_classes_per_task=num_classes_per_task, num_tasks=num_tasks, lr=lr, record_str=activation_type, pure_ff_sparsity=pure_ff_sparsity)
        )

        # retrieve activations required to comptue hyperplane for dist_func that require a hyperplane
        if activation_type_hyp:
            _, _, combined_activations_hyp, _, _, _, task, tgt, _ = (
                load_data(start_task, mode=mode, layer=layer, runs=runs, size=size, num_seeds=num_seeds, sparsity=sparsity, lr_rec=lr_rec, num_classes_per_task=num_classes_per_task, num_tasks=num_tasks, lr=lr, record_str=activation_type_hyp, pure_ff_sparsity=pure_ff_sparsity)
            )
        else:
            combined_activations_hyp = combined_activations
        
        # compute distance measures for every random seed
        seed_vals = []
        for seed in range(num_seeds):
            arr = get_inter_task_results_arr(name, dist_func=dist_func, layer=layer, seed=seed,
                                                combined_activations_hyp=combined_activations_hyp, combined_activations=combined_activations,
                                                num_tasks=num_tasks,
                                                start_task=start_task, task=task, tgt=tgt)
            readout_idx = start_task + 1 if mode in ['across_task_before_learned', 'across_task_first_learned'] else num_tasks - 1
            seed_vals.append(arr[start_task,readout_idx])
        

        distances.append(seed_vals)

    return np.array(distances)

def compute_overlap_all_activations(combined_activations, task, tgt, name, layer, chosen_task_0, chosen_task_1, chosen_tgt_0, chosen_tgt_1, seed, abs_activity=True):
    """
    Computes the overlap between two sets of activations that correspond to different digits.
    """

    # select sets of digit representations (activations) to compare based on task and target (label) indices
    chosen_task_0 = task if chosen_task_0 == -1 else chosen_task_0
    chosen_task_1 = task if chosen_task_1 == -1 else chosen_task_1
    chosen_tgt_0 = tgt if chosen_tgt_0 == -1 else chosen_tgt_0
    chosen_tgt_1 = tgt if chosen_tgt_1 == -1 else chosen_tgt_1
    chosen_activations_0 = combined_activations[name][layer][seed][(task == chosen_task_0) & (tgt == chosen_tgt_0)]
    chosen_activations_1 = combined_activations[name][layer][seed][(task == chosen_task_1) & (tgt == chosen_tgt_1)]

    # if selected to (usually turned on), take the absolute values of all activities
    if abs_activity:
        chosen_activations_0 = np.abs(chosen_activations_0)
        chosen_activations_1 = np.abs(chosen_activations_1)

    # Compute the sum across different input samples
    act_vec_0 = (chosen_activations_0).sum(axis=0)
    act_vec_1 = (chosen_activations_1).sum(axis=0)

    # Normalize sums to unit length
    act_vec_0 /= np.linalg.norm(act_vec_0)
    act_vec_1 /= np.linalg.norm(act_vec_1)

    # Compute dot product
    dot_product = np.sum(act_vec_0 * act_vec_1)

    return dot_product


def load_data(start_task,
              mode='across_time',
              layer=1,
              runs=None,
              size=None,
              num_seeds=None,
              sparsity=None,
              lr_rec=None,
              num_classes_per_task=None,
              num_tasks=None,
              lr=None,
              two_task=False,
              limited_batch_nr=2,
              task_under_consideration=0,
              record_str='-feedforward',
              pure_ff_sparsity=False):
    """
    Loads activation recordings. Arrays of activation recordings of length up to num_tasks
    are returned. The types of activations contained in the array depends on the 'mode' and
    'start_task' arguments.
    """

    # prepare dictionaries for saving data
    activations_dict = {}
    targets_dict = {}

    # iterate over learning algorithms
    for name in runs:
        activations_dict[name] = {}
        targets_dict[name] = {}
        used_record_str = record_str if 'dfc' in name else '-feedforward'

        # iterate through network layers
        for layer in range(len(size.split(","))):
            activations_dict[name][layer] = {}
            targets_dict[name][layer] = {}

            # iterate through random seeds
            for seed in range(num_seeds):
                
                # construct file name based on arguments
                two_task_string = '2task-' if two_task else ''
                limited_batch_nr_string = f"-{limited_batch_nr=}" if two_task else ''
                pure_ff_sparsity_str = "_pure-ff-sparsity" if pure_ff_sparsity else ""
                dir_name = os.getcwd() + f'/../out/activations/{two_task_string}size={size}-{lr=}-sparsity={sparsity}-{lr_rec=}-{num_classes_per_task=}{limited_batch_nr_string}{pure_ff_sparsity_str}'

                activations_dict[name][layer][seed] = []
                targets_dict[name][layer][seed] = []


                def load_data_helper():
                    activations = np.load(path + f"/{train_idx=}{test_idx=}{layer=}.npy")
                    targets = np.load(dir_name + "/" + name + f"-seed={seed+1}" + f"/activations-feedforward" + f"/targets-{train_idx=}{test_idx=}.npy")
                    activations_dict[name][layer][seed].append(activations)
                    targets_dict[name][layer][seed].append(np.argmax(targets, axis=1))

                # load activations based on 'mode' and 'start_task'
                path = dir_name + "/" + name + f"-seed={seed+1}" + f"/activations{used_record_str}"
                if mode == 'across_time':
                    test_idx = task_under_consideration if two_task else start_task
                    for train_idx in range(start_task, num_tasks):
                        load_data_helper()
                elif mode == 'across_task':
                    train_idx = num_tasks - 1
                    for test_idx in range(num_tasks):
                        load_data_helper()
                elif mode == 'across_task_first_learned':
                    for test_idx in range(start_task, num_tasks):
                        train_idx = test_idx
                        load_data_helper()
                elif mode == 'across_task_before_learned':
                    train_idx = start_task
                    for test_idx in range(train_idx, num_tasks):
                        load_data_helper()
                else:
                    raise ValueError()


    combined_activations = {}
    combined_targets = {}
    combined_activations_list = {}
    for layer in range(len(size.split(","))):
        combined_activations_list[layer] = {0: [], 1: [], 2: [], 3: [], 4: []}

    for name, activations_all_seeds in activations_dict.items():
        combined_activations[name] = {}
        combined_targets[name] = {}
        for layer in range(len(size.split(","))):
            combined_activations[name][layer] = {}
            combined_targets[name][layer] = {}
            for seed in range(num_seeds):
                activations = activations_all_seeds[layer][seed]
                concatenated_act = np.concatenate(activations, axis=0)
                concatenated_tgt = np.concatenate(targets_dict[name][layer][seed], axis=0)
                combined_activations[name][layer][seed] = concatenated_act
                combined_targets[name][layer][seed] = concatenated_tgt
                combined_activations_list[layer][seed].append(concatenated_act)

    combined_activations['all'] = {}
    for layer in range(len(size.split(","))):
        combined_activations['all'][layer] = {}
        for seed in range(num_seeds):
            combined_activations['all'][layer][seed] = np.concatenate(combined_activations_list[layer][seed], axis=0)

    task = np.array([j for j in range(start_task, num_tasks) for i in range(512)])
    tgt = combined_targets[runs[0]][0][0]
    number = task * 2 + tgt
    
    return activations_dict, targets_dict, combined_activations, combined_targets, concatenated_act, concatenated_tgt, task, tgt, number


def compute_activation_overlaps(
    cl_mode,
    lrs,
    abs_activity=True,
    overlap_func=compute_overlap_all_activations,
    start_task=0,
    num_tasks=5,
    num_numbers=10,
    num_seeds=5,
    record_str='-feedforward',
    mode='across_task',
    layer=1,
    lr_rec=40, 
    num_classes_per_task=2,
):
    """
    Computes overlaps between pairs of digit representations according to the dot product of
    the sums of absolute activity vectors corresponding to different digits.
    For domain-IL, we distinguish between intra-label (intra-target) and inter-label (inter-target)
    digit pairs. In class-IL this distinction is not necessary.
    """

    # infer network variables from CL paradigm
    sparsity = "0.4,0.8,0.5" if cl_mode == 'domain' else "0.2,0.8,0"
    size = "20,20" if cl_mode == 'domain' else "200,200"

    tgt_overlap_vals = {}  # inter-label overlaps
    task_overlap_vals = {}  # intra-label overlaps
    all_overlap_vals = {}  # overlaps between all pairs of digits

    # iterate over learning lgorithms
    for name in ['dfc', 'dfc-rec', 'dfc-sparse', 'dfc-sparse-rec']:
        tgt_overlap_vals[name] = []
        task_overlap_vals[name] = []
        all_overlap_vals[name] = []
        
        # iterate over LRs
        for lr in lrs:
            tgt_overlap_vals[name].append(np.zeros((num_tasks, num_tasks, num_seeds)))
            task_overlap_vals[name].append(np.zeros((num_tasks, num_tasks, num_seeds)))
            all_overlap_vals[name].append(np.zeros((num_numbers, num_numbers, num_seeds)))

            _, _, combined_activations, _, _, _, task, tgt, _ = (
                load_data(start_task=start_task, mode=mode, layer=layer, runs=[name], size=size, num_seeds=num_seeds, sparsity=sparsity, lr_rec=lr_rec, num_classes_per_task=num_classes_per_task, num_tasks=5, lr=lr, record_str=record_str)

            )
            
            # compute intra-label and inter-label overlaps for domain-IL
            if cl_mode == 'domain':

                # intra-target overlaps
                for chosen_task_0 in range(num_tasks):
                    for chosen_task_1 in range(chosen_task_0 + 1, num_tasks):
                        for seed in range(num_seeds):
                            overlaps = []
                            for chosen_tgt in range(2):
                                chosen_tgt_0 = chosen_tgt
                                chosen_tgt_1 = chosen_tgt

                                overlaps.append(overlap_func(combined_activations, task, tgt, name, layer, chosen_task_0, chosen_task_1, chosen_tgt_0, chosen_tgt_1, seed, abs_activity=abs_activity))

                            task_overlap_vals[name][-1][chosen_task_0, chosen_task_1, seed] = np.mean(overlaps)

                # inter-target overlaps
                chosen_tgt_0 = 0
                chosen_tgt_1 = 1
                for chosen_task_0 in range(num_tasks):
                    for chosen_task_1 in range(num_tasks):
                        overlaps = []
                        for seed in range(num_seeds):
                            overlap = (overlap_func(combined_activations, task, tgt, name, layer, chosen_task_0, chosen_task_1, chosen_tgt_0, chosen_tgt_1, seed, abs_activity=abs_activity))
                            tgt_overlap_vals[name][-1][chosen_task_0, chosen_task_1, seed] = overlap

            # compute inter-digit overlaps for class-IL (all are inter-target in class-IL)
            else:
                for chosen_number_0 in range(num_numbers):
                    for chosen_number_1 in range(chosen_number_0, num_numbers):
                        overlaps = []
                        for seed in range(num_seeds):
                            overlap = (overlap_func(combined_activations, task, tgt, name, layer, -1, -1, chosen_number_0, chosen_number_1, seed, abs_activity=abs_activity))
                            all_overlap_vals[name][-1][chosen_number_0, chosen_number_1, seed] = overlap


    return task_overlap_vals, tgt_overlap_vals, all_overlap_vals


def compute_mean_digit_separations_domainIL(name, num_seeds, tgt_overlap_vals, task_overlap_vals):
    """
    For every seed and LR, we compute the average intra-label and inter-label separation
    across all relevent digit pairs, and the difference between average inter-label and intra-label separation.
    """
    
    separation_mean_diffs = []
    inter_tgt_separation_means = []
    inter_task_separation_means = []
    
    for seed in range(num_seeds):
        inter_tgt_separation_mean = 1 - np.array([(arr[:,:,seed]).mean() for arr in tgt_overlap_vals[name]])
        inter_task_separation_mean = 1 - np.array([(arr[:,:,seed][arr[:,:,seed] > 0]).mean() for arr in task_overlap_vals[name]])
        
        separation_mean_diffs.append(
            inter_tgt_separation_mean -
            inter_task_separation_mean
        )
        
        inter_tgt_separation_means.append(inter_tgt_separation_mean)
        inter_task_separation_means.append(inter_task_separation_mean)
    
    return np.array(separation_mean_diffs), np.array(inter_tgt_separation_means), np.array(inter_task_separation_means)


def compute_mean_digit_separation_classIL(name, num_seeds, all_overlap_vals):
    """
    For every seed and LR, we compute the average inter-digit separation across all digit pairs.
    """
    all_separation_means = []
    for seed in range(num_seeds):
        all_separation_mean = 1 - np.array([(arr[:,:,seed]).mean() for arr in all_overlap_vals[name]])
        all_separation_means.append(all_separation_mean)
    return np.array(all_separation_means)


def run_hyperplane_analysis(
    dist_func,
    to_plot,  # learning algorithms used
    activation_types,  # type of activation that is compared against hyperplane
    activation_types_hyp=None,  # type of activation used to create hyperplane
    mode='across_task_before_learned',
    layer=1,
    num_tasks=5,
    num_seeds=5,
    detailed_labels=False,
    lr_rec=40,
    num_classes_per_task=2,
    cl_mode='domain'
):
    """
    Computes functions of activations with respect to hyperplanes (which are inferred from activations).
    Metrics are comptuted across LRs.
    """
    sparsity = "0.4,0.8,0.5" if cl_mode == 'domain' else "0.2,0.8,0.9"
    size = "20,20" if cl_mode == 'domain' else "200,200"
    chosen_lrs = [1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
    lrs = [10 ** (-neg_exp) for neg_exp in chosen_lrs]



    distance_means = {}
    distance_stds = {}

    names = []

    # use same activation types as analyzed activations to create hyperplane if no separate hyperplane
    # activation type was specified
    if activation_types_hyp is None:
        activation_types_hyp = activation_types

    for model, activation_type, activation_type_hyp in zip(to_plot, activation_types, activation_types_hyp):
        if detailed_labels:
            name = f"{model}{activation_type}"
        else:
            name = model
        names.append(name)
        distance_means[name] = []
        distance_stds[name] = []
        for lr in lrs:
            print(lr)
            distances = get_distances(model, dist_func=dist_func,
                                      num_tasks=num_tasks, 
                                      layer=layer, num_seeds=num_seeds, 
                                      runs=[model], size=size, sparsity=sparsity,
                                      lr_rec=lr_rec,
                                      num_classes_per_task=num_classes_per_task,
                                      lr=lr, activation_type=activation_type, activation_type_hyp=activation_type_hyp,
                                      mode=mode)
            distance_means[name].append(distances.mean())
            distance_stds[name].append(distances.std())
    return names, distance_means, distance_stds

def sparsify(arr, sparsity):
    """
    Sparsify array of activations according to given sparsity fraction.
    """
    percentiles = np.repeat(np.percentile(abs(arr), sparsity * 100, axis=1).reshape(arr.shape[0], 1), arr.shape[1], axis=1)
    arr[abs(arr) < percentiles] = 0
    return arr

def compute_dim_across_tasks(lr, layer, models, record_strs,
                             cl_mode='class',
                             mode='across_task_before_learned',
                             accumulate_activations=False,
                             untouched_frac=False,
                             n_components=None):
    
    """
    Computes the effective rank of a matrix with target activations as rows across LRs and tasks.
    In other words, this function computes effective dimensionality of target activations.
    """
    
    # sparsity fraction, needed when using recurrent activations, since they need to be sparsified
    if cl_mode == 'domain':
        layer_sparsity = 0.4 if layer == 0 else 0.8
    else:
        layer_sparsity = 0.2 if layer == 0 else 0.8
    sparsity = "0.4,0.8,0.5" if cl_mode == 'domain' else "0.2,0.8,0"
    size = "20,20" if cl_mode == 'domain' else "200,200"
        
    results = {}
    # iterate through activation types
    for name, record_str in zip(models, record_strs):
        print(name, record_str)
        results[name] = np.zeros((5, 5))

        # iterate through tasks
        for task_idx in range(5):

            # iterate through seeds
            for seed in range(5):

                # load activations
                start_task = 0 if mode == 'across_task_first_learned' else task_idx
                activations_dict, targets_dict, combined_activations, combined_targets, concatenated_act, concatenated_tgt, task, tgt, number = (
                    load_data(start_task=start_task, mode=mode, layer=layer, runs=[name], size=size, num_seeds=5, sparsity=sparsity, lr_rec=40, num_classes_per_task=2, num_tasks=5, lr=lr,
                              record_str=record_str)
                )
                
                # compute metric quantifying fraction of effective dimensionality of 
                # previously learned tasks untouched by current task.
                if untouched_frac:
                    if task_idx == 0:
                        effective_dim = 0
                        continue
                    arr = np.concatenate(activations_dict[name][layer][seed][:task_idx+1])
                    dim_cum = effective_dimensionality(arr, n_components=n_components)
                    arr = activations_dict[name][layer][seed][task_idx]
                    dim_curr = effective_dimensionality(arr, n_components=n_components)
                    arr = np.concatenate(activations_dict[name][layer][seed][:task_idx])
                    dim_prev = effective_dimensionality(arr, n_components=n_components)
                    dim_cum = max(dim_prev, dim_curr, dim_cum)
                    dim_comb = dim_prev + dim_curr - dim_cum
                    effective_dim = 1 - dim_comb / dim_prev
                else:
                    # compute effective dimensionality of all previous and current tasks together
                    if accumulate_activations:
                        arr = np.concatenate(activations_dict[name][layer][seed][:task_idx+1])
                    else: # comptue effective dimensionality of current task
                        arr = activations_dict[name][layer][seed][task_idx]
                    if record_str == '-recurrent' and name == 'dfc-sparse-rec':
                        arr = sparsify(arr, layer_sparsity)  # sparsify recurrent activations
                    effective_dim = effective_dimensionality(arr, n_components=n_components)
                
                results[name][task_idx, seed] = effective_dim

    
    return results