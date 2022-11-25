import numpy as np
import torch
import torch.nn as nn

from tests.utils import reset_seed, reset_layers, run_forward_pass
from networks.dfc_network import DFCNetwork
from networks.dfc_network_single_phase import DFCNetworkSinglePhase

eps = 1e-20
torch.set_default_dtype(torch.float64)

def test_init_feedback_layers_weight_product(n_in=10, n_hidden=[5,5], n_out=2):
    reset_seed()
    net = DFCNetwork(n_in, n_hidden, n_out, initialization_fb='weight_product')
    reset_seed()
    net = reset_layers(net)
    net.init_feedback_layers_weight_product()
    assert np.abs(net.layers[0].weights_backward.mean().item() --0.3078581112925426) < eps
    assert np.abs(net.layers[1].weights_backward.mean().item() --0.47228085471446) < eps
    assert np.abs(net.layers[2].weights_backward.mean().item() -0.1873265286609857) < eps

def test_forward(n_in=10, n_hidden=[5,5], n_out=2, batch_size=8):
    reset_seed()
    net = DFCNetwork(n_in, n_hidden, n_out)
    outputs = run_forward_pass(net, batch_size=batch_size)
    assert np.abs(outputs.mean().item() - -7.40089668480458) < eps

def test_compute_output_target(n_in=10, n_hidden=[5,5], n_out=2, batch_size=8):
    reset_seed()
    net = DFCNetwork(n_in, n_hidden, n_out)
    outputs = run_forward_pass(net, batch_size=batch_size)
    targets = torch.randn_like(outputs)
    loss = outputs.mean()
    output_targets = net.compute_output_target(loss, targets=targets)
    assert np.abs(output_targets.mean().item() - -7.4058966848045795) < eps

    net._strong_feedback = True
    output_targets = net.compute_output_target(loss, targets=targets)
    assert np.abs(output_targets.mean().item() - 0.05125384628960872) < eps

    # Using MSE.
    reset_seed()
    loss_fn = nn.MSELoss(reduction='mean')
    net = DFCNetwork(n_in, n_hidden, 1)
    outputs = run_forward_pass(net, batch_size=1)
    targets = torch.randn_like(outputs)
    net._target_stepsize = 0.5
    lamb = net.target_stepsize
    loss = loss_fn(outputs, targets) 
    output_targets = net.compute_output_target(loss)
    assert (targets.item() - output_targets.item()) < eps

def test_compute_error(n_in=10, n_hidden=[5,5], n_out=2, batch_size=8):
    reset_seed()
    net = DFCNetwork(n_in, n_hidden, n_out)
    outputs = run_forward_pass(net, batch_size=batch_size)
    targets = torch.randn_like(outputs)
    error = net.compute_error(outputs, targets)
    assert np.abs(error.mean().item() - -7.45215053109419) < eps

def test_compute_loss(n_in=10, n_hidden=[5,5], n_out=2, batch_size=8):
    reset_seed()
    net = DFCNetwork(n_in, n_hidden, n_out)
    outputs = run_forward_pass(net, batch_size=batch_size)
    targets = torch.randn_like(outputs)
    loss = net.compute_loss(outputs, targets)
    assert np.abs(loss.mean().item() - 12.861337601801736) < eps

def test_controller(n_in=10, n_hidden=[5,5], n_out=2, batch_size=8):
    reset_seed()
    net = DFCNetwork(n_in, n_hidden, n_out)
    net._target_stepsize = 0.001
    outputs = run_forward_pass(net, batch_size=batch_size)
    targets = torch.randn_like(outputs)
    loss = outputs.mean()
    output_targets = net.compute_output_target(loss, targets=targets)

    r, u, (v_fb, v_ff, v), sample_error = \
        net.controller(output_targets, alpha=0.1, dt=0.1, tmax=10, k_p=0.,
                   noisy_dynamics=False, inst_transmission=False,
                   time_constant_ratio=1., apical_time_constant=-1,
                   proactive_controller=False, sigma=0.01, sigma_output=0.01)
    assert np.abs(np.sum([r.mean().item() for r in r]) - -1.6494444738506866) < eps
    assert np.abs(np.sum([r.mean().item() for r in v_fb]) - -0.0002376787379955045) < eps
    assert np.abs(np.sum([r.mean().item() for r in v_ff]) - -3.820235574717419) < eps
    assert np.abs(np.sum([r.mean().item() for r in v]) - -3.8203458359025984) < eps
    assert np.abs(u.mean().item() - -0.00021708806753766504) < eps
    assert np.abs(sample_error.mean().item() - 0.0006974311489508217) < eps

    # Some random values for controller.
    r, u, (v_fb, v_ff, v), sample_error = \
        net.controller(output_targets, alpha=0.5, dt=0.001, tmax=10, k_p=0.,
                   noisy_dynamics=False, inst_transmission=False,
                   time_constant_ratio=0.005, apical_time_constant=-1,
                   proactive_controller=False, sigma=0.01, sigma_output=0.1)
    assert np.abs(np.sum([r.mean().item() for r in r]) - -1.6494014033597955) < eps
    assert np.abs(np.sum([r.mean().item() for r in v_fb]) - -2.396094440385542e-06) < eps
    assert np.abs(np.sum([r.mean().item() for r in v_ff]) - -3.820290870755081) < eps
    assert np.abs(np.sum([r.mean().item() for r in v]) - -3.82029260303643) < eps
    assert np.abs(u.mean().item() - -2.246867548125076e-06) < eps
    assert np.abs(sample_error.mean().item() - 0.0007071481144847871) < eps

    # Instantaneous transmission.
    r, u, (v_fb, v_ff, v), sample_error = \
        net.controller(output_targets, alpha=0.1, dt=0.1, tmax=10, k_p=0.,
                   noisy_dynamics=False, inst_transmission=True,
                   time_constant_ratio=1., apical_time_constant=-1,
                   proactive_controller=False, sigma=0.01, sigma_output=0.01)
    assert np.abs(np.sum([r.mean().item() for r in r]) - -1.6494388812201617) < eps
    assert np.abs(np.sum([r.mean().item() for r in v_fb]) - -0.00023891709267780928) < eps
    assert np.abs(np.sum([r.mean().item() for r in v_ff]) - -3.8202137009588544) < eps
    assert np.abs(np.sum([r.mean().item() for r in v]) - -3.820339577356381) < eps
    assert np.abs(u.mean().item() - -0.00021816697543379715) < eps
    assert np.abs(sample_error.mean().item() - 0.0007054751672896876) < eps

    # Proactive controller.
    r, u, (v_fb, v_ff, v), sample_error = \
        net.controller(output_targets, alpha=0.1, dt=0.1, tmax=10, k_p=0.,
                   noisy_dynamics=False, inst_transmission=False,
                   time_constant_ratio=1., apical_time_constant=-1,
                   proactive_controller=True, sigma=0.01, sigma_output=0.01)
    assert np.abs(np.sum([r.mean().item() for r in r]) - -1.6494583891148036) < eps
    assert np.abs(np.sum([r.mean().item() for r in v_fb]) - -0.000301298045610099) < eps
    assert np.abs(np.sum([r.mean().item() for r in v_ff]) - -3.82021374042069) < eps
    assert np.abs(np.sum([r.mean().item() for r in v]) - -3.820363237409966) < eps
    assert np.abs(u.mean().item() - -0.00021625293648317635) < eps
    assert np.abs(sample_error.mean().item() - 0.0006963426860830256) < eps

    # Noisy dynamics.
    reset_seed()
    r, u, (v_fb, v_ff, v), sample_error = \
        net.controller(output_targets, alpha=0.1, dt=0.1, tmax=10, k_p=0.,
                   noisy_dynamics=True, inst_transmission=False,
                   time_constant_ratio=1., apical_time_constant=-1,
                   proactive_controller=False, sigma=0.01, sigma_output=0.01)
    assert np.abs(np.sum([r.mean().item() for r in r]) - -1.650671093170783) < eps
    assert np.abs(np.sum([r.mean().item() for r in v_fb]) - -0.004735099920059644) < eps
    assert np.abs(np.sum([r.mean().item() for r in v_ff]) - -3.820078502316233) < eps
    assert np.abs(np.sum([r.mean().item() for r in v]) - -3.822057682550941) < eps
    assert np.abs(u.mean().item() - 0.0002487272603821102) < eps
    assert np.abs(sample_error.mean().item() - 0.007900083356696808) < eps

    # Realistic setting.
    reset_seed()
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

def test_compute_full_jacobian(n_in=10, n_hidden=[5,5], n_out=2, batch_size=8):
    reset_seed()
    net = DFCNetwork(n_in, n_hidden, n_out, use_jacobian_as_fb=True)
    outputs = run_forward_pass(net, batch_size=batch_size)
    J = net.compute_full_jacobian(linear=True)
    assert np.abs(J.mean().item() - -0.16522002445411907) < eps

    net = DFCNetwork(n_in, [3,3], 20, use_jacobian_as_fb=True)
    outputs = run_forward_pass(net, batch_size=10)
    J = net.compute_full_jacobian(linear=True)
    assert J.shape[0] == 10
    assert J.shape[1] == 20
    assert J.shape[2] == 26
    assert np.abs(J.mean().item() - -0.011128318260050197) < eps
    assert np.abs(torch.mean(J, dim=0)[0][0].item() - -1.0616236873431883) < eps
    assert np.abs(torch.mean(J, dim=1)[0][0].item() - -0.7468410476171041) < eps
    assert np.abs(torch.mean(J, dim=2)[0][0].item() - -0.0428126524126276) < eps

def test_dynamical_inversion(n_in=10, n_hidden=[5,5], n_out=2, batch_size=8):
    reset_seed()
    net = DFCNetwork(n_in, n_hidden, n_out, tmax_di=10)
    outputs = run_forward_pass(net, batch_size=batch_size)
    output_target = torch.randn_like(outputs) + outputs
    u_ss, v_ss, r_ss, r_out_ss, delta_v_ss, \
                (u, v_fb, v, v_ff, r) = net.dynamical_inversion(output_target,
                                                verbose=False)
    assert np.abs(np.mean([i.mean().item() for i in r_ss])- -0.5599598130218117) < eps
    assert np.abs(u_ss.mean().item() - 0.3008014069686614) < eps
    assert np.abs(np.mean([i.mean().item() for i in v_ss]) - -1.3450783102865096) < eps
    assert np.abs(r_out_ss.mean().item() - -7.495230460813419) < eps
    assert np.abs(np.mean([i.mean().item() for i in delta_v_ss]) - 0.021817621364279338) < eps
    assert np.abs(np.mean([i.mean().item() for i in v]) --1.311278885440058) < eps
    assert np.abs(np.mean([i.mean().item() for i in v_ff]) --1.3119637051832393) < eps
    assert np.abs(np.mean([i.mean().item() for i in r]) --0.5646603409864884) < eps

    assert np.abs(u.mean().item() - 0.18714092774507513) < eps
    assert np.abs(np.mean([i.mean().item() for i in v_fb]) --0.06379810100877346) < eps

def test_non_dynamical_inversion(n_in=10, n_hidden=[5,5], n_out=2, batch_size=8):
    reset_seed()
    net = DFCNetwork(n_in, n_hidden, n_out, tmax_di=10)
    outputs = run_forward_pass(net, batch_size=batch_size)
    output_target = torch.randn_like(outputs) + outputs
    u_ndi, v_ndi, r, r_out_ndi, delta_v_ndi_split = net.non_dynamical_inversion(output_target)

    assert np.abs(u_ndi.mean().item() - 0.09293629398322524) < eps
    assert np.abs(np.mean([i.mean().item() for i in v_ndi]) --1.222879780982703) < eps
    assert np.abs(np.mean([i.mean().item() for i in r]) --0.5027040424977818) < eps
    assert np.abs(r_out_ndi.mean().item() --7.341980388589249) < eps
    assert np.abs(np.mean([i.mean().item() for i in delta_v_ndi_split]) -0.10321249663928582) < eps

def test_compute_feedback_gradients(n_in=10, n_hidden=[5,5], n_out=2, batch_size=8):
    reset_seed()
    net = DFCNetwork(n_in, n_hidden, n_out, tmax_di_fb=30, target_stepsize=0.1, strong_feedback=True)
    outputs = run_forward_pass(net, batch_size=batch_size)
    output_target = torch.ones_like(outputs)
    loss = None
    net.compute_feedback_gradients(loss, output_target)

    assert np.abs(net.layers[0].weights_backward.grad.mean().item() -41.80957511628602) < eps
    assert np.abs(net.layers[1].weights_backward.grad.mean().item() -73.82333244741136) < eps
    assert np.abs(net.layers[2].weights_backward.grad.mean().item() -0.030598921155982284) < eps

def test_backward(n_in=10, n_hidden=[5,5], n_out=2, batch_size=8):
    reset_seed()
    net = DFCNetwork(n_in, n_hidden, n_out, tmax_di=10)
    outputs = run_forward_pass(net, batch_size=batch_size)
    loss = outputs.mean()
    net.backward(loss, verbose=False)

    assert np.abs(net.layers[0].weights.grad.mean().item() - -0.00039703203963749735) < eps
    assert np.abs(net.layers[1].weights.grad.mean().item() - 0.01141533886051771) < eps
    assert np.abs(net.layers[2].weights.grad.mean().item() - 0.0735888097862398) < eps
    assert np.abs(net.layers[0].bias.grad.mean().item() - 0.0016724634367679624) < eps
    assert np.abs(net.layers[1].bias.grad.mean().item() - 0.006915678719734894) < eps
    assert np.abs(net.layers[2].bias.grad.mean().item() - 0.01719209356462867) < eps