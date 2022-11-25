import numpy as np
import torch

from tests.utils import reset_seed, reset_layers, run_forward_pass
from networks.dfc_network import DFCNetwork
from networks.dfc_network_single_phase import DFCNetworkSinglePhase

eps = 1e-20
torch.set_default_dtype(torch.float64)

def test_dynamical_inversion_single_phase(n_in=10, n_hidden=[5,5], n_out=2, batch_size=8):
    reset_seed()
    net = DFCNetworkSinglePhase(n_in, n_hidden, n_out, tmax_di=10, strong_feedback=True,
                                alpha_di=0.001, noisy_dynamics=True, inst_system_dynamics=True,
                                dt_di=0.1, proactive_controller=True, time_constant_ratio=0.18,
                                epsilon_di=1e-3, k_p=1.5, sigma=0.35, sigma_output=0.37)
    outputs = run_forward_pass(net, batch_size=batch_size)
    loss = outputs.mean()
    output_target = 1e-2 * torch.randn_like(outputs) + outputs
    u_ss, v_ss, r_ss, r_out_ss, delta_v_ss, \
                (u, v_fb, v, v_ff, r) = net.dynamical_inversion(output_target, verbose=False)

    assert np.abs(u.mean().item() - 285.1545950978515) < eps
    assert np.abs(np.mean([i.mean().item() for i in v_fb]) - 214.86679180369626) < eps
    assert np.abs(np.mean([i.mean().item() for i in v]) - -65.53044603324967) < eps
    assert np.abs(np.mean([i.mean().item() for i in v_ff]) - -280.3972378369459) < eps
    assert np.abs(np.mean([i.mean().item() for i in r]) - -27.890819775558764) < eps

    # No noisy dynamics.
    net._noisy_dynamics = False
    u_ss, v_ss, r_ss, r_out_ss, delta_v_ss, \
                (u, v_fb, v, v_ff, r) = net.dynamical_inversion(output_target, verbose=False)
    assert np.abs(np.mean([i.mean().item() for i in r]) - -5.88832689719852) < eps
    net._noisy_dynamics = True

    # No proactive controller.
    net._proactive_controller = False
    u_ss, v_ss, r_ss, r_out_ss, delta_v_ss, \
                (u, v_fb, v, v_ff, r) = net.dynamical_inversion(output_target, verbose=False)
    assert np.abs(np.mean([i.mean().item() for i in r]) - -1.1685502217590986) < eps
    net._proactive_controller = True

    # No instantaneous system dynamics.
    net._pinst_system_dynamics = False
    u_ss, v_ss, r_ss, r_out_ss, delta_v_ss, \
                (u, v_fb, v, v_ff, r) = net.dynamical_inversion(output_target, verbose=False)
    assert np.abs(np.mean([i.mean().item() for i in r]) - -52.084340598381914) < eps

def test_compute_feedback_gradients_single_phase(n_in=10, n_hidden=[5,5], n_out=2, batch_size=8):
    reset_seed()
    net = DFCNetworkSinglePhase(n_in, n_hidden, n_out, tmax_di=10, strong_feedback=True)
    outputs = run_forward_pass(net, batch_size=batch_size)
    loss = outputs.mean()
    output_target = torch.randn_like(outputs) + outputs
    net.compute_feedback_gradients(loss, output_target)
    assert np.abs(net.layers[0].weights_backward.grad.mean().item() - -3046.3089469759557) < eps
    assert np.abs(net.layers[1].weights_backward.grad.mean().item() - -7334.014992655114) < eps
    assert np.abs(net.layers[2].weights_backward.grad.mean().item() - -95.57854848090015) < eps

    # At init. Values after fixing alpha_noise bug of old repo.
    net.compute_feedback_gradients(loss, output_target, init=True)
    assert np.abs(net.layers[0].weights_backward.grad.mean().item() - 0.027906337291141902) < eps
    assert np.abs(net.layers[1].weights_backward.grad.mean().item() - 0.09090703930607533) < eps
    assert np.abs(net.layers[2].weights_backward.grad.mean().item() - 0.00026759328657823195) < eps

def test_backward_single_phase(n_in=10, n_hidden=[5,5], n_out=2, batch_size=8):
    reset_seed()
    net = DFCNetworkSinglePhase(n_in, n_hidden, n_out, tmax_di=10, strong_feedback=True)
    outputs = run_forward_pass(net, batch_size=batch_size)
    loss = outputs.mean()
    targets = torch.randn_like(outputs) + outputs
    net.backward(loss, targets, verbose=False)
    assert np.abs(net.layers[0].weights.grad.mean().item() - -0.0252657118579017) < eps
    assert np.abs(net.layers[1].weights.grad.mean().item() - -0.17159854170437655) < eps
    assert np.abs(net.layers[2].weights.grad.mean().item() - 0.6317597677399861) < eps
    assert np.abs(net.layers[0].bias.grad.mean().item() - -0.05791736484468758) < eps
    assert np.abs(net.layers[1].bias.grad.mean().item() - -0.4512001460590628) < eps
    assert np.abs(net.layers[2].bias.grad.mean().item() - 0.13558249752735324) < eps
    assert np.abs(net.layers[0].weights_backward.grad.mean().item() - -2.39197946980593657) < eps
    assert np.abs(net.layers[1].weights_backward.grad.mean().item() - -5.797933714494006) < eps
    assert np.abs(net.layers[2].weights_backward.grad.mean().item() - -7.701461950131922) < eps

    # With continuous updates.
    reset_seed()
    net = DFCNetworkSinglePhase(n_in, n_hidden, n_out, tmax_di=10, strong_feedback=True,
                                   cont_updates=True)
    outputs = run_forward_pass(net, batch_size=batch_size)
    loss = outputs.mean()
    targets = torch.randn_like(outputs) + outputs
    net.backward(loss, targets, verbose=False)
    assert np.abs(net.layers[0].weights.grad.mean().item() - -0.011244470632084932) < eps
    assert np.abs(net.layers[1].weights.grad.mean().item() - -0.13892320810825062) < eps
    assert np.abs(net.layers[2].weights.grad.mean().item() - 0.5397947300588651) < eps
    assert np.abs(net.layers[0].bias.grad.mean().item() - 0.0008524011002704925) < eps
    assert np.abs(net.layers[1].bias.grad.mean().item() - -0.2307700150853588) < eps
    assert np.abs(net.layers[2].bias.grad.mean().item() - 0.11238692967260777) < eps
    assert np.abs(net.layers[0].weights_backward.grad.mean().item() - -2.39197946980593657) < eps
    assert np.abs(net.layers[1].weights_backward.grad.mean().item() - -5.797933714494006) < eps
    assert np.abs(net.layers[2].weights_backward.grad.mean().item() - -7.701461950131922) < eps

    # With noisy dynamics.
    reset_seed()
    net = DFCNetworkSinglePhase(n_in, n_hidden, n_out, tmax_di=10, strong_feedback=True,
                                   noisy_dynamics=True)
    outputs = run_forward_pass(net, batch_size=batch_size)
    loss = outputs.mean()
    reset_seed()
    targets = torch.randn_like(outputs) + outputs
    net.backward(loss, targets, verbose=False)
    assert np.abs(net.layers[0].weights.grad.mean().item() - -0.03291373818708932) < eps
    assert np.abs(net.layers[1].weights.grad.mean().item() - -0.3304696361070064) < eps
    assert np.abs(net.layers[2].weights.grad.mean().item() - 6.661941361196871) < eps
    assert np.abs(net.layers[0].bias.grad.mean().item() - 0.03177960812061968) < eps
    assert np.abs(net.layers[1].bias.grad.mean().item() - -0.4159978517247483) < eps
    assert np.abs(net.layers[2].bias.grad.mean().item() - 0.36963907948646657) < eps
    assert np.abs(net.layers[0].weights_backward.grad.mean().item() - -0.6415937387281272) < eps
    assert np.abs(net.layers[1].weights_backward.grad.mean().item() - -1.0206709550168174) < eps
    assert np.abs(net.layers[2].weights_backward.grad.mean().item() - -3.1879061449547215) < eps