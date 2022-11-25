#!/usr/bin/env python3
# Copyright 2021 Alexander Meulemans, Matilde Tristany, Maria Cervera
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title          :configs/old_into_new_cli.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :25/11/2021
# @version        :1.0
# @python_version :3.6.8
"""
A script to convert configs from old to new namings
---------------------------------------------------
"""
import __init__

import argparse
import importlib
import os
import pickle

from utils.args import parse_cmd_arguments

# The following lookup dictionaries give argument translations
# from the old to the new version.

LOOKUP_TABLE_BASE = {
    'forward_wd': 'weight_decay',
    'shallow_training': 'only_train_last_layer',
    'beta1': 'adam_beta1',
    'beta2': 'adam_beta2',
    'epsilon': 'adam_epsilon'}

LOOKUP_TABLE_DFC = {
    'ndi': 'ssa',
    'save_LU_angle': 'save_H_angle',
    'alpha_fb': 'alpha_di_fb',
    'feedback_wd': 'weight_decay_fb',
    'epochs_fb': 'init_fb_epochs',
    'beta1_fb': 'adam_beta1_fb',
    'beta2_fb': 'adam_beta2_fb',
    'epsilon_fb': 'adam_epsilon_fb',
    'save_condition_gn': 'save_condition_fb',
    'compute_gn_condition_init': 'save_condition_fb',
    'initialization_K': 'initialization_fb',
    'noise_K': 'sigma_init',
    'save_NDI_angle': 'save_ndi_angle',
    'save_BP_angle': 'save_bp_angle',
    'save_ratio_angle_ff_fb': 'save_ratio_ff_fb'}

# Special keywords.
# - include_non_converged_samples becomes NOT include_only_converged_samples
# - grad_deltav_cont becomes NOT ss or ssa
# - num_hidden and size_hidden are not a single hidden_sizes

# Should always be:
# - use_initial_activations False
# - fb_activation activation

def run():
    """Convert the config."""

    # Parse the provided config module.
    network_type = 'DFC'
    _SCRIPT_NAME = 'run_dfc.py'
    config_module = 'old_configs.mnist_sfb_2p'
    # Load default args.
    config_repo = vars(parse_cmd_arguments(network_type=network_type))

    # Load the config.
    config = importlib.import_module(config_module).config

    # dir_path = 'old_configs/idealq_mnist'
    # with open(os.path.join(dir_path, 'args.pickle'), 'rb') as f:
    #     config = pickle.load(f)
    #     config = vars(config)

    LOOKUP_TABLE_BASE = {
        'forward_wd': 'weight_decay',
        'shallow_training': 'only_train_last_layer',
        'beta1': 'adam_beta1',
        'beta2': 'adam_beta2',
        'epsilon': 'adam_epsilon'}

    LOOKUP_TABLE_DFC = {
        'ndi': 'ssa',
        'save_LU_angle': 'save_H_angle',
        'alpha_fb': 'alpha_di_fb',
        'feedback_wd': 'weight_decay_fb',
        'epochs_fb': 'init_fb_epochs',
        'beta1_fb': 'adam_beta1_fb',
        'beta2_fb': 'adam_beta2_fb',
        'epsilon_fb': 'adam_epsilon_fb',
        'save_condition_gn': 'save_condition_fb',
        'compute_gn_condition_init': 'save_condition_fb',
        'initialization_K': 'initialization_fb',
        'noise_K': 'sigma_init',
        'save_NDI_angle': 'save_ndi_angle',
        'save_BP_angle': 'save_bp_angle',
        'save_ratio_angle_ff_fb': 'save_ratio_ff_fb'}

    if 'DFC' not in network_type:
        LOOKUP_TABLE_DFC = {}

    def process_val(val):
        if type(val) == str:
            return '"' + val + '"'
        else:
            return val

    new_config = {}
    for k in config.keys():
        val = config[k]
        if k == 'network_type':
            assert val == network_type
        elif k == 'tmax_di' or k == 'tmax_di_fb':
            new_config[k] = int(process_val(val))
        elif k in LOOKUP_TABLE_BASE.keys():
            new_config[LOOKUP_TABLE_BASE[k]] = process_val(val)
        elif k in LOOKUP_TABLE_DFC.keys():
            new_config[LOOKUP_TABLE_DFC[k]] = process_val(val)
        elif k == 'single_precision':
            new_config['double_precision'] = not val
        elif k == 'include_non_converged_samples' and 'DFC' in network_type:
            new_config['include_only_converged_samples'] = not val
        elif k == 'grad_deltav_cont' and 'DFC' in network_type:  # this only works if `ndi` comes after 
                                                                 # `grad_deltav_cont` in the config
            new_config['ss'] = not val
            new_config['ssa'] = not val
        elif k == 'tmax_di_fb' and 'DFC' in network_type:
            new_config[k] = int(val)
        elif k == 'num_hidden':
            hidden_sizes = '"' + (str(config['size_hidden']) + ',')*(val - 1) \
                             + str(config['size_hidden']) + '"'
            new_config['size_hidden'] = hidden_sizes
        elif k == 'size_hidden' or k == 'out_dir':
            pass
        elif k in config_repo.keys():
            if 'epochs' in k:
                new_config[k] = int(val)
            else:
                new_config[k] = process_val(val)
        elif k == 'fb_activation':
            assert val == config['hidden_activation']
        elif k == 'use_initial_activations':
            assert not val
        elif k == 'at_steady_state' and 'DFC' in network_type:
            if config[k] == True:
                new_config['compute_jacobian_at'] = 'ss'
        else:
            print('Not processed key/value: %s/%s' % (k, str(val)))
    if 'include_only_converged_samples' not in new_config:
        # If it wasn't in the config, put in the default.
        new_config['include_only_converged_samples'] = True

    if network_type == 'DFC_single_phase':
        if not 'inst_transmission' in new_config:
            new_config['inst_transmission'] = True 
        if not 'inst_transmission_fb' in new_config:
            new_config['inst_transmission_fb'] = True 
        if not 'proactive_controller' in new_config:
            new_config['proactive_controller'] = True 
        if not 'scaling_fb_updates' in new_config:
            new_config['scaling_fb_updates'] = True 
        if not 'noisy_dynamics' in new_config:
            new_config['noisy_dynamics'] = True 
        new_config['strong_feedback'] = True 

    path = config_module + '_new'
    path = path.replace('.', '/')
    with open(path + '.py', 'w') as f:
        f.write("config = {\n")
        for key,value in new_config.items():
            f.write("'%s':%s,\n" % (key,value))
        f.write("}")

    def _args_to_cmd_str(cmd_dict):
        """Translate a dictionary of argument names to values into a string that
        can be typed into a console.

        Args:
            cmd_dict: Dictionary with argument names as keys, that map to a value.

        Returns:
            A string of the form:
                python3 train.py --out_dir=OUT_DIR --ARG1=VAL1 ...
        """
        cmd_str = 'python3 %s' % _SCRIPT_NAME

        for k, v in cmd_dict.items():
            if type(v) == bool:
                cmd_str += ' --%s' % k if v else ''
            else:
                cmd_str += ' --%s=%s' % (k, str(v))

        return cmd_str

    config_minimal = {}
    for key in config_repo.keys():
        if key in new_config.keys() and new_config[key] != config_repo[key]:
            if key in ['random_seed', 'out_dir', 'log_interval']:
                continue
            config_minimal[key] = new_config[key]

    with open(path + '.txt', 'w') as f:
        f.write(_args_to_cmd_str(config_minimal))

    return new_config

if __name__ == '__main__':
    run()