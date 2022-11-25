import numpy as np
import torch
from torch.nn.functional import one_hot
from torch import nn
import datetime
import pytest

from tests.utils import reset_seed, reset_layers, run_forward_pass, setup_training_pipeline
from utils import args
from utils import train_utils as tu
import networks.dfc_network_utils as dfcu
from networks.dfc_network import DFCNetwork
from networks.dfc_network_single_phase import DFCNetworkSinglePhase

eps = 1e-20
torch.set_default_dtype(torch.float64)

def training_sanity_checks(config, net, network_type, pre_training_params):
    """Do some sanity checks.

    This function checks that:

    * For backprop, the options "only_train_last_layer" and 
      "only_train_first_layer" don't lead to weight changes in the wrong layers.

    Args:
        config: The configuration.
        net: The network.
        network_type (str): The type of network.
        pre_training_params (dict): The parameter values before training.
    """
    if network_type == 'BP':
        for i, param in enumerate(net.params):
            if (i != 0 and config.only_train_first_layer) or \
                    config.freeze_fw_weights or \
                    (i != len(net.params) - 1 and config.only_train_last_layer):
                if isinstance(param, list):
                    for p, p_pre in zip(param, pre_training_params['ff'][i]):
                        assert torch.equal(p, p_pre)
                else:
                    assert torch.equal(param, pre_training_params['ff'][i])
            else:
                if isinstance(param, list):
                    for p, p_pre in zip(param, pre_training_params['ff'][i]):
                        assert not torch.equal(p, p_pre)
                else:
                    assert not torch.equal(param, pre_training_params['ff'][i])

    elif 'DFC' in network_type:
        for i, param in enumerate(net.feedback_params):
            if isinstance(param, list):
                for p, p_pre in zip(param, pre_training_params['fb'][i]):
                    if config.freeze_fb_weights or \
                            (config.freeze_fb_weights_output and i == \
                             net.depth - 1):
                        assert torch.equal(p, p_pre)
                    else:
                        assert not torch.equal(p, p_pre)
            else:
                if config.freeze_fb_weights:
                    assert torch.equal(param, pre_training_params['fb'][i])
                else:
                    assert not torch.equal(param, pre_training_params['fb'][i])

        for i, param in enumerate(net.forward_params):
            if isinstance(param, list):
                for p, p_pre in zip(param, pre_training_params['ff'][i]):
                    if config.freeze_fw_weights:
                        assert torch.equal(p, p_pre)
                    else:
                        assert not torch.equal(p, p_pre)
            else:
                if config.freeze_fw_weights:
                    assert torch.equal(param, pre_training_params['ff'][i])
                else:
                    assert not torch.equal(param, pre_training_params['ff'][i])

def get_network_params(net, network_type):
    training_params = {}
    training_params['ff'] = net.clone_params('forward_params') # REMOVE
    if 'DFC' in network_type:
        training_params['fb'] = net.clone_params('feedback_params')
    return training_params

@pytest.fixture(scope='session')
def mnist_pipeline_dfc():
    """
    Prepares all the relevant files that are needed to train a network using various
    functions from train_utils.py or dfc_network_utils.py.
    By setting `scope='session'` in the `pytest.fixture()` decorator, we make sure
    that this function is only executed once, thereby saving computation time since
    we want to run several tests on a given setup.
    """
    reset_seed()
    network_type = 'DFC'
    config = args.parse_cmd_arguments(network_type=network_type, default=True)
    config.clip_grad_norm = 1
    config.test = True
    config.teacher_num_train = 10
    config.teacher_num_val = 10
    config.target_stepsize = 0.5  # has no effect on feedback weights
    config.double_precision = True
    config.time_constant_ratio = 0.9
    config.time_constant_ratio_fb = 0.9
    config, logger, writer, device, dloader, net, \
        optimizers, shared, loss_fn = setup_training_pipeline(config, network_type)
    return config, logger, writer, device, dloader, net, \
        optimizers, shared, loss_fn, network_type

def test_compute_accuracy(n=10):
    reset_seed()

    # test regression output
    ones = torch.ones(n, dtype=torch.long)
    zeros = torch.zeros(n, dtype=torch.long)
    half_ones = ones.clone()
    half_ones[n // 2:] = 0

    # output not one hot encoded
    assert tu.compute_accuracy(one_hot(ones), zeros) == 0
    assert tu.compute_accuracy(one_hot(ones), ones) == 1
    assert tu.compute_accuracy(one_hot(half_ones), ones) == (n // 2) / n

    # output one hot encoded
    assert tu.compute_accuracy(one_hot(ones), one_hot(ones)) == 1
    assert tu.compute_accuracy(one_hot(ones), one_hot(zeros)) == 0
    assert tu.compute_accuracy(one_hot(half_ones), one_hot(ones)) == (n // 2) / n

def test_train_epoch_forward(mnist_pipeline_dfc):
    reset_seed()
   
    # get setup
    config, logger, writer, device, dloader, net, \
        optimizers, shared, loss_fn, network_type = mnist_pipeline_dfc

    # train forward weights for one epoch
    tu.train_epoch_forward(config, logger, device, writer, shared, dloader, net,
                           optimizers, loss_fn, network_type, epoch=1)

    # test
    assert np.abs(net.layers[0].weights.grad.mean().item() - -0.0006005257974320308) < eps
    assert np.abs(net.layers[1].weights.grad.mean().item() - -0.0025593458745836483) < eps


def test_train_feedback_parameters(mnist_pipeline_dfc): 
    reset_seed()

    # get setup
    config, logger, writer, device, dloader, net, \
        optimizers, shared, loss_fn, network_type = mnist_pipeline_dfc

    # train feedback weights for 3 epochs, extra training
    config.extra_fb_epochs = 3
    dfcu.train_feedback_parameters(config, logger, writer, device, dloader, net,
                                   optimizers, shared, loss_fn,
                                   pretraining=False)

    assert np.abs(net.layers[0].weights_backward.grad.mean().item() - 0.002011697057311368) < eps
    assert np.abs(net.layers[1].weights_backward.grad.mean().item() - -0.007530866951065589) < eps

    # train feedback weights for 3 epochs, pre-training
    config.extra_fb_epochs = 3
    dfcu.train_feedback_parameters(config, logger, writer, device, dloader, net,
                                   optimizers, shared, loss_fn,
                                   pretraining=True)

    assert np.abs(net.layers[0].weights_backward.grad.mean().item() - -0.0023901390002335145) < eps
    assert np.abs(net.layers[1].weights_backward.grad.mean().item() - -0.005765106773719057) < eps     

def test_train_epoch_feedback(mnist_pipeline_dfc):
    reset_seed()
    
    # get setup
    config, logger, writer, device, dloader, net, \
        optimizers, shared, loss_fn, network_type = mnist_pipeline_dfc
    optimizer = optimizers['feedback']

    # train feedback weights for one epoch
    dfcu.train_epoch_feedback(config, logger, writer, dloader, optimizer, net,
                              shared, loss_fn)

    # test results
    assert np.abs(net.layers[0].weights_backward.grad.mean().item() - -0.00025769896472518065) < eps
    assert np.abs(net.layers[1].weights_backward.grad.mean().item() - -0.008565876793414182) < eps
    assert net.get_max_grad() <= config.clip_grad_norm

def test_test(mnist_pipeline_dfc):
    reset_seed()

    # get setup
    config, logger, writer, device, dloader, net, \
        _, shared, loss_fn, network_type = mnist_pipeline_dfc

    # run model evaluation
    test_loss, test_accu = tu.test(config, logger, device, writer, shared,
                                   dloader, net, loss_fn, network_type, data_split='test')

    # check loss and accuracy
    assert np.abs(test_loss - 3.058435571228098) < eps
    assert np.abs(test_accu - 0.1015625) < eps

def test_train(mnist_pipeline_dfc):
    reset_seed()

    # get setup
    config, logger, writer, device, dloader, net, \
        optimizers, shared, loss_fn, network_type = mnist_pipeline_dfc
    
    # Save the parameters before training for doing sanity checks.
    before_training_params = get_network_params(net, network_type)

    # train net
    tu.train(config, logger, device, writer, dloader, net, optimizers, shared,
             network_type, loss_fn)

    # check weights, sanity tests
    assert np.abs(net.layers[0].weights.grad.mean().item() - -0.0003545898058078398) < eps
    assert np.abs(net.layers[1].weights.grad.mean().item() - -0.0026864826192486593) < eps
    assert np.abs(net.layers[0].weights_backward.grad.mean().item() - 0.002378138934380744) < eps
    assert np.abs(net.layers[1].weights_backward.grad.mean().item() - -0.006121552952664824) < eps
    training_sanity_checks(config, net, network_type, before_training_params)

    # perform sanity checks with `config.freeze_fw_weights==True`
    before_training_params = get_network_params(net, network_type)
    config.freeze_fw_weights = True
    tu.train(config, logger, device, writer, dloader, net, optimizers, shared,
             network_type, loss_fn)
    training_sanity_checks(config, net, network_type, before_training_params)
    config.freeze_fw_weights = False

    # perform sanity checks with `config.freeze_fb_weights==True`
    before_training_params = get_network_params(net, network_type)
    config.freeze_fb_weights = True
    tu.train(config, logger, device, writer, dloader, net, optimizers, shared,
             network_type, loss_fn)
    training_sanity_checks(config, net, network_type, before_training_params)
    config.freeze_fb_weights = False

    # perform sanity checks with `config.only_train_first_layer==True`
    before_training_params = get_network_params(net, network_type)
    config.only_train_first_layer = True
    tu.train(config, logger, device, writer, dloader, net, optimizers, shared,
             network_type, loss_fn)
    training_sanity_checks(config, net, network_type, before_training_params)
    config.only_train_first_layer = False

def test_loss_function_H(mnist_pipeline_dfc):
    
    # get setup
    config, logger, writer, device, dloader, net, \
        optimizers, shared, loss_fn, network_type = mnist_pipeline_dfc
    num_neurons = sum(net.n_hidden) + net.n_out
    batch_size = net._u.shape[0]

    # broadcasting the same vector u for every batch
    net._u[:, :] = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # set all elements of Q to 0
    for l in net.layers:
        l.weights_backward[:] = 0
    
    # set one element of the first column of Q to 1
    net.layers[-1].weights_backward[1, 0] = 1
    assert dfcu.loss_function_H(config, net, shared) == 1 / num_neurons
    net.layers[-1].weights_backward[1, 0] = 0

    # set one element of the second column of Q to 1
    net.layers[0].weights_backward[1, 1] = 1
    assert dfcu.loss_function_H(config, net, shared).detach().cpu().numpy() == 0
    net.layers[0].weights_backward[1, 1] = 0

    # only set u of one batch to a non-zero value
    net._u[:, :] = torch.zeros(net._u.shape)
    net._u[0, :] = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    net.layers[-1].weights_backward[1, 0] = 1
    assert dfcu.loss_function_H(config, net, shared) == 1 / num_neurons / batch_size
