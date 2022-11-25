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
# @title          :run_config.py
# @author         :am
# @contact        :ameulema@ethz.ch
# @created        :25/11/2021
# @version        :1.0
# @python_version :3.6.8
"""
Script for running an experiment from a certain configuration
-------------------------------------------------------------

This is a script to run a model with a certain hyperparameter configuration.
"""
import argparse
import importlib
import main
import numpy as np
import sys

def _override_cmd_arg(config):
    """Override the default command line arguments with the provided config.

    Args:
        config (dict): The desired config.
    """
    sys.argv = [sys.argv[0]]
    for k, v in config.items():
        if isinstance(v, bool):
            cmd = '--%s' % k if v else ''
        else:
            cmd = '--%s=%s' % (k, str(v))
        if not cmd == '':
            sys.argv.append(cmd)

def run():
    """Run the experiment."""

    # Parse the provided config module.
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_module', type=str,
                        default='configs.configs_mnist.mnist_bp',
                        help='The name of the module containing the config.')
    parser.add_argument('--network_type', type=str, default='BP',
                        choices=['BP', 'DFC', 'DFA'],
                        help='Type of network to be used for training. '
                             'See the layer classes for explanations '
                             'of the names. It HAS to correspond to the '
                             'provided config. Default: %(default)s.')
    args = parser.parse_args()

    # Override the default arguments.
    config_module = importlib.import_module(args.config_module)
    _override_cmd_arg(config_module.config)

    # Run the experiment.
    summary = main.run(network_type=args.network_type)

    return summary

if __name__ == '__main__':
    run()