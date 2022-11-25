import numpy as np
import torch

from tests.utils import reset_seed, reset_layers, run_forward_pass
from networks.dfc_layer import DFCLayer
from networks.dfc_network import DFCNetwork

eps = 1e-20
torch.set_default_dtype(torch.float64)

def reset_layer(layer):
    layer._weights.data = torch.randn(*layer.weights.shape)
    layer._weights.grad = torch.zeros(*layer.weights.shape)
    layer._bias.data = torch.randn(layer.weights.shape[0])
    layer._bias.grad = torch.zeros(layer.weights.shape[0])
    return layer

def test_forward(batch_size=8, num_in=5, num_out=3, num_last_layer=2):
    layer = DFCLayer(num_in, num_out, num_last_layer)

    reset_seed()
    layer = reset_layer(layer)

    # Random inputs.
    inputs = torch.randn(batch_size, num_in)

    # Compute expected outputs.
    a = inputs.mm(layer.weights.t())
    if layer.bias is not None:
        a += layer.bias.unsqueeze(0).expand_as(a)
    outputs_expected = layer.forward_activation_function(a)

    # Compare to actual outputs.
    outputs = layer.forward(inputs)

    assert torch.all(torch.eq(outputs_expected, outputs))
    assert np.abs(outputs.mean().item() -  0.4220820210647865) < eps

def test_compute_forward_gradients(batch_size=8, num_in=5, num_out=3,
                                   num_last_layer=2):
    layer = DFCLayer(num_in, num_out, num_last_layer)

    reset_seed()
    layer = reset_layer(layer)

    # Get teaching signals.
    delta_v = torch.randn(batch_size, num_out)
    r_previous = torch.randn(batch_size, num_in)
    # NOTE: in new repo it's always the voltage_difference
    layer.compute_forward_gradients(delta_v, r_previous)

    assert np.abs(layer.weights.grad.mean().item() - 0.15126690032206255) < eps
    assert np.abs(layer.bias.grad.mean().item() - 0.15994413200699895) < eps

def test_compute_forward_gradients_continuous(batch_size=8, num_in=5, num_out=3,
                                              num_last_layer=2):

    layer = DFCLayer(num_in, num_out, num_last_layer)

    reset_seed(11)
    layer = reset_layer(layer)

    # Get teaching signals.
    vs_time = torch.randn(num_last_layer, batch_size, num_out)
    vb_time = torch.randn(num_last_layer, batch_size, num_out)
    r_previous_time = torch.randn(num_last_layer, batch_size, num_in)
    # NOTE: in new repo it's always the voltage_difference
    layer.compute_forward_gradients_continuous(vs_time, vb_time, r_previous_time)

    assert np.abs(layer.weights.grad.mean().item() - 0.0774976061028567)< eps
    assert np.abs(layer.bias.grad.mean().item() - -0.19918700516593005)< eps

def test_compute_feedback_gradients_continuous(batch_size=8, num_in=5, num_out=3,
                                               num_last_layer=2, num_ts=8):

    layer = DFCLayer(num_in, num_out, num_last_layer)

    reset_seed(11)
    layer = reset_layer(layer)
    layer._weights_backward.data = torch.randn_like(layer._weights_backward)
    layer._weights_backward.grad = torch.zeros_like(layer._weights_backward)

    # Get teaching signals.
    va_time = torch.randn(num_ts-1, batch_size, num_out)
    u_time = torch.randn(num_ts-1, batch_size, num_last_layer)
    layer.compute_feedback_gradients_continuous(va_time, u_time)
    assert np.abs(layer.weights_backward.grad.mean().item() - \
            0.011058851365295191)< eps

    layer._weights_backward.data = torch.randn_like(layer._weights_backward)
    layer._weights_backward.grad = torch.zeros_like(layer._weights_backward)
    layer.compute_feedback_gradients_continuous(va_time, u_time, sigma=0.01, scaling=1)
    assert np.abs(layer.weights_backward.grad.mean().item() - \
            110.58851365295193)< eps

    # More thorough test in a realistic setting.
    reset_seed()
    n_in=10
    n_hidden=[5,5]
    n_out=2
    batch_size=8

    net = DFCNetwork(n_in, n_hidden, n_out, tmax_di_fb=30, target_stepsize=0.1, strong_feedback=True)
    outputs = run_forward_pass(net, batch_size=batch_size)
    output_target = torch.ones_like(outputs)
    _, u, (v_fb, _, _), _ =  \
        net.controller(output_target=output_target,
                        alpha=net.alpha_di_fb,
                        dt=net.dt_di_fb,
                        tmax=net.tmax_di_fb,
                        k_p=net.k_p_fb,
                        noisy_dynamics=True,
                        inst_transmission=net.inst_transmission_fb,
                        time_constant_ratio=net.time_constant_ratio_fb,
                        apical_time_constant=net.apical_time_constant_fb,
                        proactive_controller=net.proactive_controller,
                        sigma=net.sigma_fb,
                        sigma_output=net.sigma_output_fb)
    assert np.abs(u.mean().item() -0.12185914225884223) < eps
    assert np.abs(v_fb[0].mean().item() -0.027211822792208278) < eps
    assert np.abs(v_fb[1].mean().item() -0.03951716844353057) < eps
    assert np.abs(v_fb[2].mean().item() --0.0426075492617668) < eps

    # Compute gradient for each layer.
    u = u[1:, :, :] #  ignore the first timestep
    for i, layer in enumerate(net.layers):
        v_fb_i = v_fb[i][:-1, :, :]

        # compute a layerwise scaling for the feedback weights
        scaling = 1.
        if net.scaling_fb_updates:
            scaling = (1 + net.time_constant_ratio_fb / net.tau_noise) \
                 ** (len(net.layers) - i - 1)

        # get the amount of noise used
        sigma_i = net.sigma_fb
        if i == len(net.layers) - 1:
            sigma_i = net.sigma_output_fb

        layer.compute_feedback_gradients_continuous(v_fb_i, u,
                                                    sigma=sigma_i,
                                                    scaling=scaling)
    assert np.abs(net.layers[0]._weights_backward.grad.mean().item() -41.80957511628602) < eps
    assert np.abs(net.layers[1]._weights_backward.grad.mean().item() -73.82333244741136) < eps
    assert np.abs(net.layers[2]._weights_backward.grad.mean().item() -0.030598921155982284) < eps