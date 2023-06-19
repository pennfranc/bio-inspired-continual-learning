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
# @title          :utils/train_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :28/11/2021
# @version        :1.0
# @python_version :3.7.4
"""
Helper functions for training and testing networks
--------------------------------------------------

A collection of functions for training and testing networks.
"""
from hypnettorch.utils import torch_ckpts as ckpts
import numpy as np
import os
import time
import torch
import torch.nn as nn
import warnings

from networks import dfc_network_utils as dfc
from utils import math_utils as mutils
from utils import plt_utils
from utils import sim_utils
from utils.optimizer_utils import extract_parameters

def train_incremental(config, logger, device, writer, dloaders, net, optimizers, shared,
                      network_type, loss_fn):
    """
    Training and testing procedure for continual learning.
    """

    shared.train_var.task_test_loss = []
    shared.train_var.task_test_accu = []
    shared.train_var.task_train_loss = []
    shared.train_var.task_train_accu = []
    shared.train_var.task_test_accu_taskIL = []
    shared.train_var.task_train_accu_taskIL = []
    
    for i in range(config.num_tasks):
        setattr(shared.train_var, f"task_{i}_test_loss", [])
        setattr(shared.train_var, f"task_{i}_test_accu", [])
        setattr(shared.train_var, f"task_{i}_train_loss", [])
        setattr(shared.train_var, f"task_{i}_train_accu", [])

    if config.shuffle_tasks:
        permutation = np.random.permutation(config.num_tasks)
        dloaders = [dloaders[i] for i in permutation]
        print("Chosen permutation:", permutation)

    # iterate over tasks
    for idx, dloader in enumerate(dloaders):
        print(f"Task {idx}")

        # train current task
        shared = train(config, logger, device, writer, dloader, net, optimizers, shared,
                       network_type, loss_fn, idx)

        # when recording activations, the network is tested on all tasks after learning each task
        if config.record_first_batch_activations:
            test_accu, test_loss, test_accu_taskIL, _, test_accus, test_losses = \
                test_tasks(config, logger, device, writer, shared, dloaders,
                        net, loss_fn,  network_type, data_split='test', train_idx=idx)
            train_accu, train_loss, train_accu_taskIL, _, train_accus, train_losses = \
                test_tasks(config, logger, device, writer, shared, dloaders,
                        net, loss_fn,  network_type, data_split='train', train_idx=idx)
        else: # test network on all tasks up until current one
            test_accu, test_loss, test_accu_taskIL, _, test_accus, test_losses = \
                test_tasks(config, logger, device, writer, shared, dloaders[:(idx + 1)],
                        net, loss_fn,  network_type, data_split='test', train_idx=idx)
            train_accu, train_loss, train_accu_taskIL, _, train_accus, train_losses = \
                test_tasks(config, logger, device, writer, shared, dloaders[:(idx + 1)],
                        net, loss_fn,  network_type, data_split='train', train_idx=idx)

        for i in range(idx + 1):
            getattr(shared.train_var, f"task_{i}_test_accu").append(test_accus[i])
            getattr(shared.train_var, f"task_{i}_test_loss").append(test_losses[i])
            getattr(shared.train_var, f"task_{i}_train_accu").append(train_accus[i])
            getattr(shared.train_var, f"task_{i}_train_loss").append(train_losses[i])

        print(f"{test_accu=}, {test_loss=}")
        shared.train_var.task_test_loss.append(test_loss)
        shared.train_var.task_test_accu.append(test_accu)
        shared.train_var.task_train_loss.append(train_loss)
        shared.train_var.task_train_accu.append(train_accu)
        shared.train_var.task_train_accu_taskIL.append(train_accu_taskIL)
        shared.train_var.task_test_accu_taskIL.append(test_accu_taskIL)
        

    # separately record last accuracy/loss
    shared.train_var.task_test_accu_last = shared.train_var.task_test_accu[-1]
    shared.train_var.task_test_loss_last = shared.train_var.task_test_loss[-1]
    shared.train_var.task_train_accu_last = shared.train_var.task_train_accu[-1]
    shared.train_var.task_train_loss_last = shared.train_var.task_train_loss[-1]
    shared.train_var.task_train_accu_taskIL_last = shared.train_var.task_train_accu_taskIL[-1]
    shared.train_var.task_test_accu_taskIL_last = shared.train_var.task_test_accu_taskIL[-1]
    
    sim_utils.update_summary_info_cl(config, shared, network_type)
    shared.summary['finished'] = 1

    return shared

def test_tasks(config, logger, device, writer, shared, dloaders_subset, net, loss_fn,
               network_type, data_split='test', train_idx=-1):
    """
    Tests network on all tasks within the 'dloaders_subset' list and computes
    the total accuracy and loss.
    """
    accu_weighted_sum = 0
    loss_weighted_sum = 0
    accu_taskIL_weighted_sum = 0
    total_num_samples = 0
    accus = []
    losses = []
    for idx, dloader in enumerate(dloaders_subset):
        test_loss, test_accu, test_accu_taskIL, num_samples = test(config, logger, device, writer,
                                                 shared, dloader, net, loss_fn,
                                                 network_type, record_results=False,
                                                 data_split=data_split,
                                                 train_idx=train_idx, test_idx=idx)
        
        accus.append(test_accu)
        losses.append(test_loss)

        if num_samples == 0:
            continue

        accu_weighted_sum += test_accu * num_samples
        loss_weighted_sum += test_loss * num_samples
        accu_taskIL_weighted_sum += test_accu_taskIL * num_samples
        total_num_samples += num_samples

        print(f"{data_split=}, {idx=}, {test_accu=}")

    return (accu_weighted_sum / total_num_samples,
            loss_weighted_sum / total_num_samples,
            accu_taskIL_weighted_sum / total_num_samples,
            total_num_samples, accus, losses)
        

def train(config, logger, device, writer, dloader, net, optimizers, shared,
          network_type, loss_fn, task_id=-1):
    """Train the network.

    Args:
        config: The command-line arguments.
        logger: The logger.
        device: The cuda device.
        writer: The tensorboard writer.
        dloader: The dataset.
        net: The network.
        optimizers: The optimizers.
        shared: Shared object with task information.
        network_type (str): The type of network.
        loss_fn: The loss function.
        task_id: Corresponds to the task id when continual learning, otherwise
            it is set to -1.

    Return:
        (dict): The shared object containing summary information.
    """
    if config.test:
        logger.info('Option "test" is active. This is a dummy run!')

    logger.info('Training network ...')
    net.train()
    net.zero_grad()

    if network_type in ['EWC', 'SI', 'BPExt', 'L2']:
        net.learn_batch(dloader)
        return shared

    # If the error needs to be computed as the gradient of the loss within 
    # the network, provide the name of the loss function being used.
    if hasattr(config, 'error_as_loss_grad') and config.error_as_loss_grad:
        if shared.classification:
            loss_function_name = 'cross_entropy'
        else:
            loss_function_name = 'mse'
        net.loss_function_name = loss_function_name

    for e in range(config.epochs):
        logger.info('Training epoch %i/%i...' % (e+1, config.epochs))
        epoch_initial_time = time.time()

        ### Train.
        # Feedback training for two-phase DFC.
        if network_type == 'DFC' and not config.freeze_fb_weights:
            dfc.train_epoch_feedback(config, logger, writer, dloader,
                                     optimizers['feedback'], net, shared,
                                     loss_fn, epoch=e)

        # Forward training.
        epoch_losses, epoch_accs = train_epoch_forward(config, logger,
                                               device, writer, shared, dloader,
                                               net, optimizers, loss_fn,
                                               network_type, epoch=e)

        # If required train feedback weights for extra epochs.
        if 'DFC' in network_type and not config.freeze_fb_weights:
            dfc.train_feedback_parameters(config, logger, writer, device,
                                          dloader, net, optimizers, shared,
                                          loss_fn)

        if not shared.continual_learning:
            ### Test.
            epoch_test_loss, epoch_test_accu, _, _ = test(config, logger, device, writer,
                                                    shared, dloader, net, loss_fn,
                                                    network_type)

            ### Validate.
            if not config.no_val_set:
                epoch_val_loss, epoch_val_accu, _, _ = test(config, logger, device,
                                                        writer, shared, dloader, net,
                                                        loss_fn, network_type,
                                                        data_split='validation')

            # Keep track of performance results.
            epoch_time = np.round(time.time() - epoch_initial_time)
            shared.train_var.epochs_time.append(epoch_time)

            # Log some information.
            logger.info('     Test loss: %.4f' % epoch_test_loss)
            if shared.classification:
                logger.info('     Test accu: %.2f%%' % (epoch_test_accu*100))
            logger.info('     Time %i s' % epoch_time)

            # Write summary information.
            shared = sim_utils.update_summary_info(config, shared, network_type)

            # Add results to the writer.
            sim_utils.add_summary_to_writer(config, shared, writer, e+1)
            sim_utils.log_stats_to_writer(config, writer, e+1, net)

            # Same the performance summary.
            sim_utils.save_summary_dict(config, shared)
            if config.epoch_summary_interval != -1 and \
                                            e % config.epoch_summary_interval == 0:
                # Every few epochs, save separate summary file.
                sim_utils.save_summary_dict(config, shared, epoch=e)

        if net.contains_nans():
            logger.info('Network contains NaNs, terminating training.')
            shared.summary['finished'] = -1
            break

        # Save the training network.
        if e % config.checkpoint_interval == 0 and config.save_checkpoints:
            store_dict = {'state_dict': net.state_dict,
                          'net_state': 'epoch_%i' % e,
                          'train_loss': shared.train_var.epochs_train_loss[-1],
                          'test_loss': shared.train_var.epochs_test_loss[-1]}
            if shared.classification:
                store_dict['train_acc'] = shared.train_var.epochs_train_accu[-1]
                store_dict['test_acc'] = shared.train_var.epochs_test_accu[-1]
            ckpts.save_checkpoint(store_dict,
                    os.path.join(config.out_dir, 'ckpts/training'), None)

        # Kill the run if results are below desired threshold.
        if shared.classification and e == 3 and epoch_accs[-1] < config.min_acc:
            logger.info('Simulation killed: low accuracy at epoch %i.'%(e+1))
            shared.summary['finished'] = -1
            break
        
        # Stop training early if desired accuracy is already achieved
        if shared.classification and epoch_accs[-1] > config.stop_early_at_accu:
            break


    # Save the final network.
    if config.save_checkpoints:
        store_dict = {'state_dict': net.state_dict,
                      'net_state': 'trained',
                      'train_loss': shared.train_var.epochs_train_loss[-1],
                      'test_loss': shared.train_var.epochs_test_loss[-1]}
        if shared.classification:
            store_dict['train_acc'] = shared.train_var.epochs_train_accu[-1]
            store_dict['test_acc'] = shared.train_var.epochs_test_accu[-1]
        ckpts.save_checkpoint(store_dict,
                              os.path.join(config.out_dir, 'ckpts/final'), None)

    # Finish up the training.
    if shared.summary['finished'] == 0 and (not shared.continual_learning):
        # Only overwrite if the training hasn't been stopped due to NaNs.
        shared.summary['finished'] = 1
    logger.info('Training network ... Done.')

    return shared

def train_epoch_forward(config, logger, device, writer, shared, dloader, net,
                        optimizers, loss_fn, network_type, epoch=None):
    """Train forward weights for one epoch.

    For backpropagation, remember that forward and feedback parameters are one
    and the same, so this function is equivalent to normal training.

    Args:
        config (Namespace): The command-line arguments.
        logger: The logger.
        device: The PyTorch device to be used.
        writer (SummaryWriter): TensorboardX summary writer to save logs.
        shared: Shared object with task information.
        dloader: The dataset.
        net: The neural network.
        optimizers (dict): The optimizers.
        loss_fn: The loss function to use.
        network_type (str): The type of network.
        epoch: The current epoch.

    Returns:
        (....): Tuple containing:

        - **epoch_losses**: The list of losses in all batches of the epoch.
        - **epoch_accs**: The list of accuracies in all batches of the epoch.
            ``None`` for non classification tasks.
    """
    epoch_losses = []
    epoch_accs = [] if shared.classification else None
    single_phase = network_type == 'DFC_single_phase'

    # Do we need to compute the gradients in this function?
    compute_gradient = not config.freeze_fw_weights
    if single_phase:
        compute_gradient = compute_gradient or not config.freeze_fb_weights

    if 'DFC' in network_type:
        if config.save_lu_loss:
            epoch_loss_lu = 0
        net.rel_dist_to_NDI = []

    num_samples = 0
    for i, (inputs, targets) in enumerate(dloader.train):
        if config.limited_batch_nr > 0 and i > config.limited_batch_nr:
            break
        # Reset optimizers.
        optimizers['forward'].zero_grad()
        if net.use_recurrent_weights:
            optimizers['recurrent'].zero_grad()
        if single_phase:
            optimizers['feedback'].zero_grad()
        batch_size = inputs.shape[0]

        # Make predictions.
        predictions = net.forward(inputs, sparsity=True)

        # Inform the network whether values should be logged (once per epoch)
        if 'DFC' in network_type:
            net.save_ndi_updates = False
            if i == 0:
                net.save_ndi_updates = config.save_ndi_angle

        ### Compute loss and accuracy.
        batch_loss = loss_fn(predictions, targets)
        batch_accuracy = None
        if shared.classification:
            batch_accuracy = compute_accuracy(predictions, targets)

        ### Compute gradients and update weights.
        if compute_gradient:
            net.backward(batch_loss, targets=targets)
            if config.clip_grad_norm != -1:
                for param in extract_parameters(net, config, network_type):
                    nn.utils.clip_grad_norm_(param, config.clip_grad_norm)
                if net.use_recurrent_weights:
                    for param in extract_parameters(net, config, network_type,
                                                    params_type='recurrent'):
                        nn.utils.clip_grad_norm_(param, config.clip_grad_norm)
                if single_phase:
                    for param in extract_parameters(net, config, network_type,\
                            params_type='feedback'):
                        nn.utils.clip_grad_norm_(param, config.clip_grad_norm)
                #assert net.get_max_grad() <= config.clip_grad_norm
                
            if hasattr(config, 'use_bp_updates') and config.use_bp_updates:
                net.set_grads_to_bp(batch_loss, retain_graph=True)

            # Perform the update.
            optimizers['forward'].step()
            if net.use_recurrent_weights:
                optimizers['recurrent'].step()
            if single_phase:
                optimizers['feedback'].step()

        ### Compute H loss.
        if  hasattr(config, 'save_lu_loss') and config.save_lu_loss:
            epoch_loss_lu += dfc.loss_function_H(config, net, shared)

        ### Store values.
        epoch_losses.append(batch_loss.detach().cpu().numpy())
        if shared.classification:
            epoch_accs.append(batch_accuracy)

        shared.train_var.batch_idx += 1
        num_samples += batch_size

        if config.test and i == 1:
            break

        # stop early if desired training accuracy is already achieved
        if shared.classification and (batch_accuracy > config.stop_early_at_accu):
            break

    # Compute angles if needed.
    if not config.no_plots or config.save_df:
        if 'DFC' in network_type:
            dfc.save_angles(config, writer, epoch+1, net, batch_loss)
            if config.save_H_angle:
                shared.train_var.epochs_lu_angle.append(\
                    net.lu_angles[0].tolist()[-1])
            if config.save_condition_fb:
                gn_condition = net.compute_condition_two()
                shared.train_var.gn_condition.append(gn_condition.item())

    # Save results in train_var.
    shared.train_var.epochs_train_loss.append(np.mean(epoch_losses))
    if shared.classification:
        shared.train_var.epochs_train_accu.append(np.mean(epoch_accs))

    if 'DFC' in network_type:
        if config.save_lu_loss:
            shared.train_var.epochs_train_loss_lu.append(epoch_loss_lu/num_samples)
        if config.compare_with_ndi:
            shared.train_var.rel_dist_to_ndi.append(np.mean(net.rel_dist_to_ndi))

    return epoch_losses, epoch_accs

def test(config, logger, device, writer, shared, dloader, net, loss_fn,
         network_type, data_split='test', record_results=True,
         train_idx=-1, test_idx=-1):
    """Test the network.

    Args:
        (....): See docstring of function :func:`train`.
        data_split (str): The test split to use: `test` or `validation`.
        record_results (bool): Whether to record loss (and accuracy)
            in the 'shared' dictionary. This is turned off when
            'test()' is called as a subroutine of 'test_tasks()'.

    Return:
        (....): Tuple containing:

        - **test_loss**: The average test loss.
        - **test_acc**: The average test accuracy. ``None`` for non
            classification tasks.
    """
    # Chose the correct data split.
    if data_split == 'test':
        data = dloader.test
    elif data_split == 'validation':
        data = dloader.val
    elif data_split == 'train':
        data = dloader.train
    
    with torch.no_grad():
        test_loss = 0
        test_accu = 0 if shared.classification else None
        test_accu_taskIL = 0 if shared.classification else None
        num_samples = 0

        # Quick fix for generator state to be the same when recording activations
        # compared to when not recording activations
        if config.record_first_batch_activations and (test_idx >= (train_idx + 1)) and (data_split=='train'):
            return 1, 1, 1

        for i, (inputs, targets) in enumerate(data):
            batch_size = inputs.shape[0]

            if network_type in ['EWC', 'SI', 'BPExt', 'L2']:
                predictions = net.predict(data)
            else:
                predictions = net.forward(inputs, sparsity=True)
            

            ### Compute loss and accuracy.
            test_loss += batch_size * loss_fn(predictions, targets).item()
            if shared.classification:
                test_accu += batch_size * compute_accuracy(predictions, targets)   
                test_accu_taskIL += batch_size * compute_accuracy_taskIL(predictions, targets, config.num_classes_per_task, test_idx)   

            num_samples += batch_size
            
            # saving activations/targets, if required
            if config.record_first_batch_activations and (data_split == 'test') and (i == 0):

                # saving feedforward activations, if required
                dir_name = config.out_dir + "/activations-feedforward"
                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
                
                for layer in range(len(net.layers)):
                    with open(dir_name + f"/{train_idx=}{test_idx=}{layer=}.npy", 'wb') as f:
                        np.save(f, net.layers[layer].activations.detach().cpu().numpy())
                
                with open(dir_name + f"/targets-{train_idx=}{test_idx=}.npy", 'wb') as f:
                        np.save(f, targets.detach().cpu().numpy())

                # saving controller-induced target activations, if DFC
                if network_type == 'DFC':
                    dir_name = config.out_dir + "/activations-controller"
                    if not os.path.exists(dir_name):
                        os.mkdir(dir_name)

                    r, u, (v_fb, v_ff, v), sample_error = net.controller(targets, net.alpha_di, net.dt_di,
                                    net.tmax_di,
                                    k_p=net.k_p,
                                    noisy_dynamics=net.noisy_dynamics,
                                    inst_transmission=net.inst_transmission,
                                    time_constant_ratio=net.time_constant_ratio,
                                    apical_time_constant=net.apical_time_constant,
                                    proactive_controller=net.proactive_controller,
                                    sigma=net.sigma,
                                    sigma_output=net.sigma_output)
                    
                    for layer in range(len(net.layers)):
                        with open(dir_name + f"/{train_idx=}{test_idx=}{layer=}.npy", 'wb') as f:
                            np.save(f, r[layer][-1, :, :].detach().cpu().numpy())
                    
                    dir_name = config.out_dir + "/activations-recurrent"
                    if not os.path.exists(dir_name):
                        os.mkdir(dir_name)

                    for layer in range(len(net.layers)):
                        with open(dir_name + f"/{train_idx=}{test_idx=}{layer=}.npy", 'wb') as f:
                            np.save(f, torch.tanh(v_ff[layer][-1, :, :]).detach().cpu().numpy())

            if config.test and i == 1:
                break

    # For auto-encoding runs, plot some reconstructions.
    if config.dataset == 'mnist_autoencoder' and not config.no_plots:
        plt_utils.plot_auto_reconstructions(config, writer, inputs, predictions)
 
    # Because we use mean reduction and the last batch might have
    # different size, we multiply in each batch by the number of samples and
    # redivide here by the total number.
    test_loss /= num_samples
    if shared.classification:
        test_accu /= num_samples
        test_accu_taskIL /= num_samples

    # Save results in train_var.
    if record_results and data_split == 'test':
        shared.train_var.epochs_test_loss.append(test_loss)
        if shared.classification:
            shared.train_var.epochs_test_accu.append(test_accu)
    elif record_results and data_split == 'validation':
        shared.train_var.epochs_val_loss.append(test_loss)
        if shared.classification:
            shared.train_var.epochs_val_accu.append(test_accu)

    return test_loss, test_accu, test_accu_taskIL, num_samples


def compute_accuracy(predictions, labels):
    """Compute the average accuracy of the given predictions.

    Inspired by
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    
    Args:
        predictions (torch.Tensor): Tensor containing the output of the linear
            output layer of the network.
        labels (torch.Tensor): Tensor containing the labels of the mini-batch.

    Returns:
        (float): Average accuracy of the given predictions.
    """
    if len(labels.shape) > 1:
        # In case of one-hot-encodings, need to extract class.
        labels = labels.argmax(dim=1)

    _, pred_labels = torch.max(predictions.data, 1)
    total = labels.size(0)
    correct = (pred_labels == labels).sum().item()

    return correct/total

def compute_accuracy_taskIL(predictions, labels, num_classes_per_task, task_idx):

    task_idx_start = task_idx * num_classes_per_task
    task_idx_end = (task_idx + 1) * num_classes_per_task
    
    labels = labels[:, task_idx_start:task_idx_end]
    labels = labels.argmax(dim=1)

    _, pred_labels = torch.max(predictions.data[:, task_idx_start:task_idx_end], 1)
    total = labels.size(0)
    correct = (pred_labels == labels).sum().item()

    return correct/total
