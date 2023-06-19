#!/usr/bin/env python3
# Copyright 2019 Christian Henning
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
# title          :hpsearch/hpconfig_template.py
# author         :ch
# contact        :henningc@ethz.ch
# created        :09/09/2019
# version        :1.0
# python_version :3.6.8
"""
Hyperparameter-search configuration template
--------------------------------------------

A configuration file for our custom hyperparameter search script. This
configuration is meant for hyperparameter searches of the simulation defined by
:mod:`main`.
"""
from utils.sim_utils import get_summary_keys

##########################################
### Please define all parameters below ###
##########################################

# Define a dictionary with parameter names as keys and a list of values for
# each parameter. For flag arguments, simply use the values [True, False].
# Note, the output directory is set by the hyperparameter search script.
#
# Example: {'option1': [12, 24], 'option2': [0.1, 0.5],
#           'option3': [True]}
# This disctionary would correspond to the following 4 configurations:
#   python3 SCRIPT_NAME.py --option1=12 --option2=0.1 --option3
#   python3 SCRIPT_NAME.py --option1=12 --option2=0.5 --option3
#   python3 SCRIPT_NAME.py --option1=24 --option2=0.1 --option3
#   python3 SCRIPT_NAME.py --option1=24 --option2=0.5 --option3
#
# If fields are commented out (missing), the default value is used.
# Note, that you can specify special conditions below.

grid = {
    ###################################################################
    ### ALL COMMAND-LINE OPTIONS ACCESSIBLE TO THE HPSEARCH GO HERE ###
    ###################################################################

    ### Dataset options ###
    'dataset': ['split_mnist'],

    ### Network options ###
    'hidden_activation': ['relu'],
    'size_hidden': ["200,200"],

    ### Training options ###
    'epochs': [4],
    'cl_mode': ['class'],
    'clip_grad_norm': [-1],
    'initialization': ['xavier'],
    'optimizer': ['Adam'],
    'adam_beta1': [0.9],
    'adam_epsilon': [1e-08],
    'batch_size': [512],
    'lr': [1 / 10**(i / 2) for i in range(2, 13, 1)],

    
    'random_seed': [1, 2, 3, 4, 5],
    'num_tasks': [5],
    'shuffle_tasks': [False]
}

# Sometimes, not the whole grid should be searched. For instance, if an SGD
# optimizer has been chosen, then it doesn't make sense to search over multiple
# beta2 values of an Adam optimizer.
# Therefore, one can specify special conditions.
# NOTE, all conditions that are specified here will be enforced. Thus, they
# overwrite the grid options above.
#
# How to specify a condition? A condition is a key value tuple: whereas as the
# key as well as the value is a dictionary in the same format as in the grid
# above. If any configurations matches the values specified in the key dict,
# The values specified in the values dict will be searched instead.
#
# Note, if arguments are commented out above but appear in the conditions, the
# condition will be ignored.
conditions = [

]

####################################
### DO NOT CHANGE THE CODE BELOW ###
####################################
# The following set of conditions should always be enforced.
_BASE_CONDITIONS = []
conditions = conditions + _BASE_CONDITIONS

# Name of the script that should be executed by the hyperparameter search.
# Note, the working directory is set seperately by the hyperparameter search
# script, so don't include paths.
_SCRIPT_NAME = 'run_bp.py'

# This file is expected to reside in the output folder of the simulation.
_SUMMARY_FILENAME = 'performance_overview.txt'

# These are the keywords that are supposed to be in the summary file.
# A summary file always has to include the keyword `finished`!.
_SUMMARY_KEYWORDS = [
    'finished',
    'task_test_loss',
    'task_test_accu',
    'task_test_loss_last',
    'task_test_accu_last',
    'task_train_loss_last',
    'task_train_accu_last',
    'task_test_accu_taskIL_last',
    'task_train_accu_taskIL_last'
]

# The name of the command-line argument that determines the output folder
# of the simulation.
_OUT_ARG = 'out_dir'

def _get_performance_summary(out_dir, cmd_ident):
    """See docstring of method
    :func:`hpsearch.hpsearch._get_performance_summary`.

    You only need to implement this function, if the default parser in module
    :func:`hpsearch.hpsearch` is not sufficient for your purposes.

    In case you would like to use a custom parser, you have to set the
    attribute :attr:`_SUMMARY_PARSER_HANDLER` correctly.
    """
    pass

# In case you need a more elaborate parser than the default one define by the
# function :func:`hpsearch.hpsearch._get_performance_summary`, you can pass a
# function handle to this attribute.
# Value `None` results in the usage of the default parser.
_SUMMARY_PARSER_HANDLE = None # Default parser is used.
#_SUMMARY_PARSER_HANDLE = _get_performance_summary # Custom parser is used.

def _performance_criteria(summary_dict, performance_criteria):
    """Evaluate whether a run meets a given performance criteria.

    This function is needed to decide whether the output directory of a run is
    deleted or kept.

    Args:
        summary_dict: The performance summary dictionary as returned by
            :attr:`_SUMMARY_PARSER_HANDLE`.
        performance_criteria (float): The performance criteria. E.g., see
            command-line option `performance_criteria` of script
            :mod:`hpsearch.hpsearch_postprocessing`.

    Returns:
        bool: If :code:`True`, the result folder will be kept as the performance
        criteria is assumed to be met.
    """
    ### Example:
    # return summary_dict['performance_measure1'] > performance_criteria
    return True
    #raise NotImplementedError('TODO implement')

# A function handle, that is used to evaluate the performance of a run.
#_PERFORMANCE_EVAL_HANDLE = None
_PERFORMANCE_EVAL_HANDLE = _performance_criteria

# A key that must appear in the `_SUMMARY_KEYWORDS` list. If `None`, the first
# entry in this list will be selected.
# The CSV file will be sorted based on this keyword. See also attribute
# `_PERFORMANCE_SORT_ASC`.
_PERFORMANCE_KEY = None
assert(_PERFORMANCE_KEY is None or _PERFORMANCE_KEY in _SUMMARY_KEYWORDS)
# Whether the CSV should be sorted ascending or descending based on the
# `_PERFORMANCE_KEY`.
_PERFORMANCE_SORT_ASC = False

# FIXME: This attribute will vanish in future releases.
# This attribute is only required by the `hpsearch_postprocessing` script.
# A function handle to the argument parser function used by the simulation
# script. The function handle should expect the list of command line options
# as only parameter.
# Example:
# >>> from classifier.imagenet import train_args as targs
# >>> f = lambda argv : targs.parse_cmd_arguments(mode='cl_ilsvrc_cub',
# ...                                             argv=argv)
# >>> _ARGPARSE_HANDLE = f
from utils.args import parse_cmd_arguments
_ARGPARSE_HANDLE = lambda argv : parse_cmd_arguments(network_type='BP',
                                                     argv=argv)

if __name__ == '__main__':
    pass