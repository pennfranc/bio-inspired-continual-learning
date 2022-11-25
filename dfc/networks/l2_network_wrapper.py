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
# @title          :networks/bp_network.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :28/11/2021
# @version        :1.0
# @python_version :3.7.4
"""
Implementation of a simple network that is trained with backpropagation
-----------------------------------------------------------------------

A simple network that is prepared to be trained with backprop.
"""
import numpy as np
import torch
import torch.nn as nn
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path+"/../../Continual-Learning-Benchmark")
from agents.regularization import L2

        
class L2Network(L2):

    def __init__(self, *args, **kwargs):
        
        super(L2Network, self).__init__(**kwargs)
        self.use_recurrent_weights = False
        self.reset_optimizer = False

    @property
    def name(self):
        return 'L2Network'

    @property
    def layer_class(self):
        """Define the layer type to be used."""
        return None


    def load_state_dict(self, state_dict):
        """Load a state into the network.

        This function sets the forward and backward weights.

        Args:
            state_dict (dict): The state with forward and backward weights.
        """
        pass