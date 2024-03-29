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
# @title          :datahandlers/data_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :19/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Helper functions for generating different datasets
--------------------------------------------------

A collection of helper functions for generating datasets to keep other scripts
clean.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import copy

from hypnettorch.data.mnist_data import MNISTData
from hypnettorch.data.special.split_mnist import get_split_mnist_handlers
from hypnettorch.data.special.split_fashion_mnist import get_split_fashion_mnist_handlers
from hypnettorch.data.special.split_cifar import SplitCIFAR100Data
from hypnettorch.data.special.permuted_mnist import PermutedMNISTList

from datahandlers.dataset import DatasetWrapper, HypnettorchDatasetWrapper

MNISTData._SUBFOLDER = ''
MNIST_DIR = 'data/MNISTData/raw'
KMNIST_DIR = 'data/KMNISTData/raw'
CIFAR_DIR = 'data/CIFARData/'
FASHION_MNIST_DIR = 'data/FashionMNISTData/raw'

def generate_task(config, logger, device):
    """Generate the user defined task.

    Args:
        config: Command-line arguments.
        logger: The logger.
        device: The cuda device.

    Returns:
        (DatasetWrapper): A dataset.
    """
    data_dir = './data'

    if config.dataset in ['mnist', 'fashion_mnist', 'cifar10']:
        dhandler = generate_computer_vis_task(config, logger, device, data_dir)
    elif config.dataset == 'mnist_autoencoder':
        dhandler = generate_mnist_auto_task(config, logger, device, data_dir)
    elif config.dataset == 'student_teacher':
        dhandler = generate_student_teacher_task(config, logger, device)
    elif config.dataset in ['split_mnist', 'split_kmnist']:
        if config.dataset == 'split_kmnist':
            path = KMNIST_DIR
            kmnist = True
        else:
            path = MNIST_DIR
            kmnist = False
        dhandlers = get_split_mnist_handlers(path, use_one_hot=True, cl_mode=config.cl_mode,
                                             num_classes_per_task=config.num_classes_per_task, permute_labels=config.permute_labels,
                                             custom_permutation=config.custom_label_permutation, kmnist=kmnist)
        dwrappers = []
        out_size = 10 if config.cl_mode == 'class' else config.num_classes_per_task
        in_size = 784
        for handler in dhandlers:
            dwrapper = HypnettorchDatasetWrapper(handler, config.batch_size, 
                                                 in_size=in_size, out_size=out_size,
                                                 device=device,
                                                 double_precision=config.double_precision)
            dwrappers.append(dwrapper)

        dhandler = dwrappers
    elif config.dataset == 'permuted_mnist':
        pd = 2 # Apply padding as in original paper.
        in_shape = [28 + 2*pd, 28 + 2*pd, 1]
        in_size = np.prod(in_shape)
        out_size = 10 * config.num_tasks if config.cl_mode == 'class' else 10
        print(in_size, out_size)

        rand = np.random.RandomState(0)
        permutations = [None] + [rand.permutation(in_size)
                                 for _ in range(config.num_tasks - 1)]

        dhandlers = PermutedMNISTList(permutations, MNIST_DIR, padding=pd,
                                      trgt_padding=None, show_perm_change_msg=False,
                                      cl_mode=config.cl_mode)
        dwrappers = []
        for handler in dhandlers:
            # deepcopy of the handler is needed here because otherwise the same MNIST
            # dataset with the same permutation is shared across handlers 
            dwrapper = HypnettorchDatasetWrapper(copy.deepcopy(handler), config.batch_size, 
                                                 in_size=in_size, out_size=out_size,
                                                 device=device,
                                                 double_precision=config.double_precision)
            dwrappers.append(dwrapper)

        dhandler = dwrappers

    elif config.dataset == 'split_fashion_mnist':
        dhandlers = get_split_fashion_mnist_handlers(FASHION_MNIST_DIR, use_one_hot=True, cl_mode=config.cl_mode,
                                             num_classes_per_task=config.num_classes_per_task, permute_labels=config.permute_labels,
                                             custom_permutation=config.custom_label_permutation)
        dwrappers = []
        out_size = 10 if config.cl_mode == 'class' else config.num_classes_per_task
        in_size = 784
        for handler in dhandlers:
            dwrapper = HypnettorchDatasetWrapper(handler, config.batch_size, 
                                                 in_size=in_size, out_size=out_size,
                                                 device=device,
                                                 double_precision=config.double_precision)
            dwrappers.append(dwrapper)

        dhandler = dwrappers

    elif config.dataset == 'split_cifar':
        num_classes_per_task = 10
        out_size = 100 if config.cl_mode == 'class' else num_classes_per_task
        in_size = 32 * 32 * 3

        steps = num_classes_per_task
        dwrappers = []
        for i in range(0, 100, steps):
            handler = SplitCIFAR100Data(CIFAR_DIR, full_out_dim=(config.cl_mode == 'class'),
                use_one_hot=True, labels=range(i, i+steps))

            dwrapper = HypnettorchDatasetWrapper(handler, config.batch_size, 
                                                 in_size=in_size, out_size=out_size,
                                                 device=device,
                                                 double_precision=config.double_precision)
            dwrappers.append(dwrapper)
            if len(dwrappers) == config.num_tasks:
                break

        dhandler = dwrappers
    else:
        raise ValueError('The requested dataset is not supported.')

    return dhandler

def generate_computer_vis_task(config, logger, device, data_dir):
    """Generate a computer vision datahandler.

    Args:
        config: Command-line arguments.
        logger: The logger.
        device: The cuda device.
        data_dir (str): The data directory.

    Returns:
        (....): See docstring of function `generate_task`.
    """
    transform = None
    if config.dataset in ['mnist', 'mnist_autoencoder']:
        # Downloading MNIST from the page of Yann Lecun can give errors. This
        # problem is solved in torchvision version 0.9.1 but for earlier versions
        # the following fix can be used.
        if torchvision.__version__ < '0.9.1':
            datasets.MNIST.resources = [
                ('https://ossci-datasets.s3.amazonaws.com/mnist/train' + 
                 '-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
                ('https://ossci-datasets.s3.amazonaws.com/mnist/train' +
                 '-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
                ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k' +
                 '-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
                ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k' +
                 '-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
            ]

        if config.dataset == 'mnist':
            logger.info('Loading MNIST dataset.')
            from datahandlers.mnist_data import MNISTData as CVData
        elif config.dataset == 'mnist_autoencoder':
            logger.info('Loading MNIST autoencoder dataset.')
            from datahandlers.mnist_auto_data import MNISTAutoData as CVData
        train_val_split = [55000, 5000]
    elif config.dataset == 'fashion_mnist':
        logger.info('Loading Fashion MNIST dataset.')
        from datahandlers.fashionmnist_data import FashionMNISTData as CVData
        train_val_split = [55000, 5000]
    elif config.dataset == 'cifar10':
        logger.info('Loading CIFAR-10 dataset.')
        from datahandlers.cifar10_data import CIFAR10Data as CVData
        train_val_split = [45000, 5000]

    ### Load the testing data.
    testset = CVData(data_dir, device, train=False, download=True,
                     double_precision=config.double_precision,
                     target_class_value=config.target_class_value)
    test_loader = DataLoader(testset, batch_size=config.batch_size)

    ### Load the training data and split with validation if necessary.
    trainset = CVData(data_dir, device, train=True, download=True,
                      double_precision=config.double_precision,
                      target_class_value=config.target_class_value)
    val_loader = None
    if not config.no_val_set:
        trainset, valset = torch.utils.data.random_split(trainset,
                                                         train_val_split)
        val_loader = DataLoader(valset, batch_size=config.batch_size,
                                shuffle=False)
    train_loader = DataLoader(trainset, batch_size=config.batch_size,
                              shuffle=True)

    ### Create the dataset.
    ds = DatasetWrapper(train_loader, test_loader, valset=val_loader,
                        name=config.dataset, in_size=testset._in_size,
                        out_size=testset._out_size)

    return ds


def generate_mnist_auto_task(config, logger, device, data_dir):
    """Generate an MNIST autoencoder datahandler.

    Args:
        (....): See docstring of function `generate_computer_vision_task`.

    Returns:
        (....): See docstring of function `generate_task`.
    """
    # Downloading MNIST from the page of Yann Lecun can give errors. This
    # problem is solved in torchvision version 0.9.1 but for earlier versions
    # the following fix can be used.
    if torchvision.__version__ != '0.9.1':
        datasets.MNIST.resources = [
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train' + 
             '-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train' +
             '-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k' +
             '-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k' +
             '-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
        ]

    logger.info('Loading MNIST autoencoder dataset.')
    from datahandlers.mnist_auto_data import MNISTAutoData as CVData
    train_val_split = [55000, 5000]

    ### Load the testing data.
    testset = CVData(data_dir, device, train=False, download=True,
                     double_precision=config.double_precision)
    test_loader = DataLoader(testset, batch_size=config.batch_size)

    ### Load the training data and split with validation if necessary.
    trainset = CVData(data_dir, device, train=True, download=True,
                      double_precision=config.double_precision)
    val_loader = None
    if not config.no_val_set:
        trainset, valset = torch.utils.data.random_split(trainset,
                                                         train_val_split)
        val_loader = DataLoader(valset, batch_size=config.batch_size,
                                shuffle=False)
    train_loader = DataLoader(trainset, batch_size=config.batch_size,
                              shuffle=True)

    ### Create the dataset.
    ds = DatasetWrapper(train_loader, test_loader, valset=val_loader,
                        name=config.dataset, in_size=testset._in_size,
                        out_size=testset._out_size)

    return ds

def generate_student_teacher_task(config, logger, device):
    """Generate a teacher network-based datahandler.

    Args:
        (....): See docstring of function `generate_computer_vision_task`.

    Returns:
        (....): See docstring of function `generate_task`.
    """
    logger.info('Loading a teacher-based dataset.')
    from datahandlers.student_teacher_data import RegressionDataset
    from networks import net_utils
    activation = 'linear' if config.teacher_linear else 'tanh'

    # Get the random seeds for each data split.
    fixed_random_seed = np.random.RandomState(config.data_random_seed)

    ### Load the testing data.
    testset = RegressionDataset(device,
                                n_in=config.teacher_n_in,
                                n_out=config.teacher_n_out, 
                                n_hidden=config.teacher_size_hidden,
                                num_data=config.teacher_num_test,
                                activation=activation,
                                double_precision=config.double_precision,
                                random_seed=fixed_random_seed.randint(100))
    test_loader = DataLoader(testset, batch_size=config.batch_size)

    ### Load the training data and split with validation if necessary.
    trainset = RegressionDataset(device,
                                n_in=config.teacher_n_in,
                                n_out=config.teacher_n_out, 
                                n_hidden=config.teacher_size_hidden,
                                num_data=config.teacher_num_train,
                                activation=activation,
                                double_precision=config.double_precision,
                                random_seed=fixed_random_seed.randint(100))
    train_loader = DataLoader(trainset, batch_size=config.batch_size,
                              shuffle=True)

    val_loader = None
    if not config.no_val_set:
        valset = RegressionDataset(device,
                                    n_in=config.teacher_n_in,
                                    n_out=config.teacher_n_out, 
                                    n_hidden=config.teacher_size_hidden,
                                    num_data=config.teacher_num_val,
                                    activation=activation,
                                    double_precision=config.double_precision,
                                    random_seed=fixed_random_seed.randint(100))
        val_loader = DataLoader(valset, batch_size=config.batch_size,
                                shuffle=True)

    ### Create the dataset.
    ds = DatasetWrapper(train_loader, test_loader, valset=val_loader,
                        name=config.dataset, in_size=testset._in_size,
                        out_size=testset._out_size)

    net_utils.log_net_details(logger, testset.teacher)

    return ds
