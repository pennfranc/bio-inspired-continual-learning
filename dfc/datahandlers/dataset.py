#!/usr/bin/env python3
# Copyright 2021 Maria Cervera
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
# @title          :datahandlers/dataset.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :19/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Implementation of a general Dataset
-----------------------------------

Extremely simple implementation of a dataset wrapper. 
"""
import numpy as np
from  torch import from_numpy

class DatasetWrapper():
	"""Implementation of a simple dataset wrapper.

	Besides containing the train, test and validation sets for a given task,
	some information about the dataset is also contained, like input and
	output size, and whether the targets exist.

	Args:
		trainset (torch.utils.data.Dataloader): The train set.
		testset (torch.utils.data.Dataloader): The test set.
		valset (torch.utils.data.Dataloader): The (optional) validation set.
		name (str): The name of the dataset.
		in_size (str): The input size.
		out_size (str): The output size.
	"""
	def __init__(self, trainset, testset, valset=None, name=None,
		         in_size=None, out_size=None):

		self.train = trainset
		self.test = testset
		self.val = valset
		self.name = name

		self.has_targets = True
		if len(self.train.dataset[0]) == 1:
			self.has_targets = False
			raise NotImplementedError('TODO')

		# Even though these could be calculated inside this function, it's
		# easier to provide them externally and use knowledge about the dataset
		# being used for safety.
		self.in_size = in_size
		self.out_size =  out_size


class HypnettorchDatasetWrapper(DatasetWrapper):

	def __init__(self, dhandler, batch_size, in_size, out_size, device, double_precision):
		self.batch_size = batch_size
		self.device = device
		self.in_size = in_size
		self.out_size = out_size
		self.dhandler = dhandler
		self.double_precision = double_precision
		self.name = dhandler.get_identifier()

		self.dhandler.shuffle_test_samples = False

	def torch_generator(self, hypnettorch_dhandler, device):
		for batch_size, x_numpy, y_numpy in hypnettorch_dhandler:
			x, y = (self.dhandler.input_to_torch_tensor(x_numpy, device),
				    self.dhandler.output_to_torch_tensor(y_numpy, device))
			if self.double_precision:
				x, y = x.double(), y.double()
			yield x, y
	
	@property
	def train(self):
		return self.torch_generator(self.dhandler.train_iterator(self.batch_size), self.device)

	@property
	def val(self):
		return self.torch_generator(self.dhandler.val_iterator(self.batch_size), self.device)
	
	@property
	def test(self):
		return self.torch_generator(self.dhandler.test_iterator(self.batch_size), self.device)