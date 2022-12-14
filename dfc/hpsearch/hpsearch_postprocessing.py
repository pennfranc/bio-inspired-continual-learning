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
# @title          :hpsearch/hpsearch_postprocessing.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :09/06/2019
# @version        :1.0
# @python_version :3.6.8
"""
Hyperparameter Search - Postprocessing
--------------------------------------

A postprocessing for a hyperparameter search that has been executed via the
script :mod:`hypnettorch.hpsearch.hpsearch`.

Post-processing can be executed as: 

.. code-block:: console

   $ python3 -m hpsearch.hpsearch_postprocessing out_dir \\
     --grid_module=configs.hpsearch.hpconfig_template

IMPORTANT NOTE! This script is copied from the 
`hypnettorch <https://github.com/chrhenning/hypnettorch>`_ 
repository from Christian Henning, and is only slightly modified for our own
purposes. It allows, for example, to condition based on relative values of
hyperparameters. We mainly use it to run searches in clusters and save the
results in a CSV file.
"""
import argparse
from datetime import datetime
import importlib
from glob import glob
import os
import pandas
import pickle
import shutil
import sys
from warnings import warn

import hpsearch.hpsearch as hpsearch
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= \
        'Postprocessing of the Automatic Parameter Search')
    parser.add_argument('out_dir', type=str, default='./out/hyperparam_search',
                        help='The output directory of the hyperparameter ' +
                             'search. Default: %(default)s.')
    parser.add_argument('--performance_criteria', type=float, default=None,
                        help='Delete all runs that do not meet the ' +
                             'performance criteria. Note, how this ' +
                             'performance criteria is evaluated depends on ' +
                             'the function handle specified in the ' +
                             'configuration file (see argument ' +
                             '"grid_module"). Default: %(default)s.')
    parser.add_argument('--grid_module', type=str,
                        default=hpsearch._DEFAULT_GRID,
                        help='Name of the configuration module. Note, the ' +
                             '"grid" and "conditions" attributes may have ' +
                             'changed, as they are not used by this script. ' +
                             'Default: %(default)s.')
    parser.add_argument('--run_cwd', type=str, default='.',
                        help='The working directory in which runs were ' +
                             'executed (in case the run script resides at a ' +
                             'different folder than this hpsearch script. ' +
                             'Provide the same option here as provided to the' +
                             '"hpsearch" script. Default: "%(default)s".')
    args = parser.parse_args()

    print('### Running Hyperparameter Search Postprocessing ...')

    grid_module = importlib.import_module(args.grid_module)
    hpsearch._read_config(grid_module,
        require_perf_eval_handle=args.performance_criteria is not None,
        require_argparse_handle=True)

    if args.run_cwd != '.':
        os.chdir(args.run_cwd)
        print('Current working directory: %s.' % os.path.abspath(os.curdir))

    if not os.path.exists(args.out_dir):
        raise FileNotFoundError('Output directory %s does not exist.' % \
                                (args.out_dir))

    commands_fn = os.path.join(args.out_dir, 'commands.pickle')
    if not os.path.exists(commands_fn):
        raise FileNotFoundError('File containing job commands could not be ' +
                                'found in output folder: %s.'% (commands_fn))

    # All folders can be assumed to be result folders of individual jobs.
    result_dirs = [(f, os.path.join(args.out_dir, f)) for f in
                   os.listdir(args.out_dir) if
                   os.path.isdir(os.path.join(args.out_dir, f))]

    # Note, these commands are already in the execution order. Though, that
    # doesn't mean that the cluster has scheduled them in this order!
    with open(commands_fn, 'rb') as f:
        commands = pickle.load(f)

    # Staring jobs with the python method busb always results in .err and .out
    # files in the current directory independent of the specified path.
    # Therefore, we need to move them to their target location.
    bsub_out_files = [f for f in os.listdir('.') if
        os.path.isfile(os.path.join('.', f)) and f.startswith('job_') and
        f.endswith('.out')]
    bsub_err_files = [f for f in os.listdir('.') if
        os.path.isfile(os.path.join('.', f)) and f.startswith('job_') and
        f.endswith('.err')]

    # Result folders that should be deleted.
    to_be_deleted = []

    finished_jobs = []
    missing_jobs = [] # Commands that didn't run at all.
    unfinished_jobs = [] # Commands that didn't do all train iters.

    csv_file_content = None

    for i, cmd in enumerate(commands):
        if len(result_dirs) == 0:
            break

        cmd_path = cmd['out_dir']
        dir_name = os.path.basename(cmd_path)

        ### Read performance Summary.
        try:
            performance_dict = hpsearch._SUMMARY_PARSER_HANDLE(cmd_path, i)
        except:
            #traceback.print_exc(file=sys.stdout)
            warn('Could not read performance summary from command %d.' % i)
            missing_jobs.append(cmd)
            to_be_deleted.append(cmd_path)
            continue

        # Check whether jobs have finished properly.
        if not 'finished' in performance_dict:
            continue
        has_finished = int(performance_dict['finished'][0])
        if has_finished == 1:
            finished_jobs.append(cmd)
        else:
            unfinished_jobs.append(cmd)


        ### Add data frame to CSV file.
        cmd[hpsearch._OUT_ARG] = cmd_path
        for k, v in performance_dict.items():
            cmd[k] = v

        cmd_frame = pandas.DataFrame.from_dict(cmd)
        if csv_file_content is None:
            csv_file_content = cmd_frame
        else:
            csv_file_content = pandas.concat([csv_file_content, cmd_frame],
                                             sort=True)

        ### Move .out and .err files.
        if len(bsub_out_files) > 0:
            for f in bsub_out_files:
                if f.startswith('job_%s' % dir_name):
                    os.rename(f, os.path.join(cmd_path, f))
            for f in bsub_err_files:
                if f.startswith('job_%s' % dir_name):
                    os.rename(f, os.path.join(cmd_path, f))
        ### Slurm out files.
        job_out_file = glob('%s_*.out' % (dir_name))
        job_err_file = glob('%s_*.err' % (dir_name))
        job_script_fn = '%s_script.sh' % (dir_name)
        for job_f in [*job_out_file, *job_err_file, job_script_fn]:
            if os.path.exists(job_f):
                os.rename(job_f, os.path.join(cmd_path, job_f))

    results_fn = os.path.join(args.out_dir, 'postprocessing_results.csv')
    fn_pickle_finished = os.path.join(args.out_dir,
                                      'postprocessing_finished_jobs.pickle')
    fn_pickle_unfinished = os.path.join(args.out_dir,
                                        'postprocessing_unfinished_jobs.pickle')
    fn_pickle_missing = os.path.join(args.out_dir,
                                     'postprocessing_missing_jobs.pickle')

    # Rename existing postprocessing results.
    if os.path.exists(results_fn):
        # The user might have run the script before and now is just rerunning it
        # with refined arguments. Though, the results might not be identical as
        # some result folders might habe been deleted already.
        date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        os.rename(results_fn, os.path.join(args.out_dir,
            'postprocessing_results_%s.csv' % (date_str)))
        if os.path.exists(fn_pickle_finished):
            os.rename(fn_pickle_finished, os.path.join(args.out_dir,
            'postprocessing_finished_jobs_%s.pickle' % (date_str)))
        if os.path.exists(fn_pickle_unfinished):
            os.rename(fn_pickle_unfinished, os.path.join(args.out_dir,
            'postprocessing_unfinished_jobs_%s.pickle' % (date_str)))
        if os.path.exists(fn_pickle_missing):
            os.rename(fn_pickle_missing, os.path.join(args.out_dir,
            'postprocessing_missing_jobs_%s.pickle' % (date_str)))

    print('%d jobs have been completed.' % (len(finished_jobs)))
    with open(fn_pickle_finished, 'wb') as f:
        pickle.dump(finished_jobs, f)
    print('%d jobs stopped before completion.' % (len(unfinished_jobs)))
    with open(fn_pickle_unfinished, 'wb') as f:
        pickle.dump(unfinished_jobs, f)
    print('%d jobs have not been started.' % (len(missing_jobs)))
    with open(fn_pickle_missing, 'wb') as f:
        pickle.dump(missing_jobs, f)

    csv_file_content = csv_file_content.sort_values(hpsearch._PERFORMANCE_KEY,
        ascending=hpsearch._PERFORMANCE_SORT_ASC)
    csv_file_content.to_csv(results_fn, sep=';', index=False)

    print('### Running Hyperparameter Search Postprocessing ... Done')
