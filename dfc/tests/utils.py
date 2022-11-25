import torch
import numpy as np

from argparse import Namespace
from datahandlers.data_utils import generate_task
from networks.net_utils import generate_network
from utils.optimizer_utils import get_optimizers
from utils import math_utils as mutils
from utils import sim_utils

def reset_seed(seed=10):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.empty_cache()

def reset_layers(net):
    for layer in net.layers:
        layer._weights.data = torch.randn(*layer.weights.shape)
        layer._weights.grad = torch.zeros(*layer.weights.shape)
        layer._bias.data = torch.randn(layer.weights.shape[0])
        layer._bias.grad = torch.zeros(layer.weights.shape[0])
        layer._weights_backward.data = torch.randn_like(layer._weights_backward)
        layer._weights_backward.grad = torch.zeros_like(layer._weights_backward)
    return net

def run_forward_pass(net, batch_size=8):
    reset_seed()
    net = reset_layers(net)
    inputs = torch.randn(batch_size, net.n_in)
    outputs = net.forward(inputs)
    return outputs 

def assert_equal_weights(net):
    for layer in net.layers:
        print(layer.weights.mean().item(), layer.weights_backward.mean().item())

def setup_training_pipeline(config, network_type):
    # code below taken from run() in main.py, added default=True argument to parse_cmd_arguments
    shared = Namespace()
    if config.dataset in ['mnist', 'fashion_mnist', 'cifar10']:
        shared.classification = True
    elif config.dataset in ['mnist_autoencoder', 'student_teacher']:
        shared.classification = False
    device, writer, logger = sim_utils.setup_environment(config)
    dloader = generate_task(config, logger, device)
    net = generate_network(config, dloader, device, network_type, logger=logger)
    optimizers = get_optimizers(config, net, network_type=network_type,
                                logger=logger)
    shared = sim_utils.setup_summary_dict(config, shared, network_type)
    if shared.classification:
        loss_fn = mutils.cross_entropy_fn()
    else:
        loss_fn = torch.nn.MSELoss()
    shared = sim_utils.setup_summary_dict(config, shared, network_type)
    
    return config, logger, writer, device, dloader, net, \
        optimizers, shared, loss_fn