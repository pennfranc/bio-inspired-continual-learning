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
# @title          :networks/dfc_layer.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :28/11/2021
# @version        :1.0
# @python_version :3.7.4
"""
Implementation of a layer for Deep Feedback Control
---------------------------------------------------

A layer that is prepared to be trained with DFC.
"""
from contextlib import suppress
import numpy as np
import torch
import torch.nn as nn

from networks.layer_interface import LayerInterface
from networks.dfc_network_utils import get_layer_sparsity, sparsifier

class DFCLayer(LayerInterface):
    """Implementation of a Deep Feedback Control layer.

    It contains the following important functions:

    * forward: which computes the linear activation based on the previous layer
      as well as the post non-linearity activation. It stores these in the
      attributes "_linear_activations" and "_activa.tions".
    * compute_forward_gradients: computes the forward parameter updates and
      stores them under "grad", by using the pre-synaptic activations and the
      controller feedback. The ule is based on a voltage difference rule.
    * compute_forward_gradients_continuous: same as "compute_forward_gradients"
      but it performs an integration over time.
    * compute_feedback_gradients: compute the feedback gradients.
    * compute_feedback_gradients_continuous: same as
      "compute_feedback_gradients" but it performs an integration over time.

    Args:
        (....): See docstring of class :class:`layer_interface.LayerInterface`.
        last_layer_features (int): The size of the output layer.
    """
    def __init__(self, in_features, out_features, last_layer_features,
                 bias=True, use_recurrent_weights=False, requires_grad=False,
                 forward_activation='tanh',
                 initialization='orthogonal',
                 initialization_fb='weight_product'):
        super().__init__(in_features, out_features, bias=bias,
                         requires_grad=requires_grad,
                         forward_activation=forward_activation,
                         initialization=initialization)

        if initialization_fb is None:
            initialization_fb = initialization
        self._initialization_fb = initialization_fb
        self._last_features = last_layer_features
        self._activations = None
        self._linear_activations = None
        self.use_recurrent_weights = use_recurrent_weights

        # Create and initialize feedback weights.
        self.set_direct_feedback_layer(last_layer_features, out_features)

        # Create and initialize recurrent weights.
        if self.use_recurrent_weights:
            self.set_recurrent_weights(out_features)
            self.init_layer(self._weights_recurrent, initialization='zeros')


        # The "weight_product" initialization is applied at the network level,
        # since it requires knowledge of all weight matrices. So here, we
        # initialize them equal to the feedfoward weights and then it will get
        # overwritten.
        if initialization_fb=='weight_product':
            initialization_fb = initialization
        self.init_layer(self._weights_backward,
                        initialization=initialization_fb)

    @property
    def weights_recurrent(self):
        """Getter for read-only attribute :attr:`_weights_backward`."""
        return self._weights_recurrent

    @weights_recurrent.setter
    def weights_recurrent(self, tensor):
        """Setter for feedback weights.

        Args:
            tensor (torch.Tensor): The tensor of values to set.
        """
        self._weights_recurrent = tensor

    @property
    def weights_backward(self):
        """Getter for read-only attribute :attr:`_weights_backward`."""
        return self._weights_backward

    @weights_backward.setter
    def weights_backward(self, tensor):
        """Setter for feedback weights.

        Args:
            tensor (torch.Tensor): The tensor of values to set.
        """
        self._weights_backward = tensor

    @property
    def activations(self):
        """Getter for read-only attribute :attr:`activations` """
        return self._activations

    @activations.setter
    def activations(self, value):
        """ Setter for the attribute activations"""
        self._activations = value

    @property
    def linear_activations(self):
        """Getter for read-only attribute :attr:`linear_activations` """
        return self._linear_activations

    @linear_activations.setter
    def linear_activations(self, value):
        """Setter for the attribute :attr:`linear_activations` """
        self._linear_activations = value

    def set_direct_feedback_layer(self, last_features, out_features):
        """Create the network backward parameters.

        This layer connects the output layer to a hidden layer. No biases are
        used in direct feedback layers. These backward parameters have no
        gradient as they are fixed.

        Note that as opposed to DFA, here the backwards weights are not
        Parameters.

        Args:
            (....): See docstring of method
                :meth:`layer_interface.LayerInterface.set_layer`.
        """
        self._weights_backward = torch.empty((out_features, last_features))

    def set_recurrent_weights(self, out_features):
        """Create the layer recurrent connections.

        Args:
            out_features (int):
                The number of features in the current layer.
        """
        self._weights_recurrent = torch.empty((out_features, out_features))

    def recurrent_activation(self, v_ff, r_previous, turn_off_rec_norm_normalization):
        """
        Updates feedforward activations v_ff to incorporate recurrent gating effects
        dependent on post-linearity activations of the same layer  r_previous.

        Args:
            v_ff (torch.Tensor): Feedforward activations (no recurrent effects).
            r_prevous (torch.Tensor): Activations of same-layer neurons of previous
                timestep including recurrent (and controller) effects.
            turn_off_rec_norm_normalization (bool): Whether to turn off rescaling of
                neuron activities after applying recurrent gating.
        """
        new_v_ff = v_ff * torch.sigmoid((abs(r_previous)).mm(self.weights_recurrent.clone().fill_diagonal_(1).t()))
        if not turn_off_rec_norm_normalization:
            new_v_ff *= (torch.norm(v_ff) / torch.norm(new_v_ff))
        return new_v_ff

    def forward(self, x, rec_inference_iterations=0, 
                layer_sparsity=0, maintain_total_activity=False,
                dt_di=0.001, time_constant_ratio=0.03553062335953924,
                turn_off_rec_norm_normalization=False):
        """Compute the output of the layer.

        This method applies first a linear mapping with the parameters
        ``weights`` and ``bias``, after which it applies the forward activation
        function.

        In the forward pass there is no noise, and thus the normal activations
        and the low-pass filtered activations are identical.
        
        Args:
            x (torch.Tensor): Mini-batch of size `[B, in_features]` with input
                activations from the previous layer or input.
            rec_inference_iterations (int): The number of iterations to simulate the
                recurrent dynamics.
            layer_sparsity (float):
                The desired sparsity fraction for the current layer.
            maintain_total_activity (bool):
                Whether to rescale activity after sparsification to maintain L2 norm.
            dt_di (float): The timestep to be used in the differential equations.
                Only needed here if `rec_inference_iterations > 0`.
            time_constant_ratio (float): Ratio of the time constant of the voltage
                dynamics w.r.t. the controller dynamics. Only needed here if
                `rec_inference_iterations > 0`.
            turn_off_rec_norm_normalization (bool): Whether to turn off rescaling of
                neuron activities after applying recurrent gating.

        Returns:
            The mini-batch of output activations of the layer.
        """

        # compute feedforward activations
        a_1 = x.mm(self.weights.t())
        if self.bias is not None:
            a_1 += self.bias.unsqueeze(0).expand_as(a_1)

        # sparsify feedforward activations according to 'layer_sparsity'
        a_1 = sparsifier(a_1, a_1, layer_sparsity, maintain_total_activity=maintain_total_activity)
        a = a_1

        # simulate recurrent dynamics if rec_inference_iterations>0 (generally not used)
        if self.use_recurrent_weights:
            for _ in range(rec_inference_iterations):
                a_new = self.recurrent_activation(a_1, self.forward_activation_function(a),
                                                  turn_off_rec_norm_normalization)
                a = a + dt_di / time_constant_ratio * (a_new - a)
                a = sparsifier(a, a, layer_sparsity, maintain_total_activity=maintain_total_activity)

        self.linear_activations = a
        self.linear_activations_lp = a

        self.activations = self.forward_activation_function(a)
        self.activations_lp = self.forward_activation_function(a)

        return self.activations

    def compute_recurrent_gradients(self, delta_v, r_previous,
                                  r_current,
                                  suppressed_neurons=None,
                                  learning_rule='nonlinear_difference',
                                  rec_learning_neurons='all',
                                  rec_grad_normalization=None,
                                  turn_off_rec_norm_normalization=False,
                                  presynaptic_inhibition=False):
        """Computes recurrent gradients using a local-in-time learning rule.

        Args:
            delta_v: The feedback teaching signal from the controller.
            r_previous (torch.Tensor): The activations of the previous layer.
            r_current (torch.Tensor): The activations of the current layer.
            suppressed_neurons (torch.Tensor): Optionally, a boolean tensor indicating
                which neurons were suppressed by the WTA mechanism.
            rec_learning_neurons (str): Indicates which neuron's incoming weights should
                be updated.
            rec_grad_normalization (str): Optionally, specifies how recurrent weight updates
                are normalized.
            turn_off_rec_norm_normalization (bool): Whether to turn off rescaling of
                neuron activities after applying lateral inhibition.
            presynaptic_inhibition (bool): Whether to only consider presynaptic activity, 
                instead of also taking into account postsynaptic v_ff, when computing
                recurrent weight updates.
        """
        batch_size = r_previous.shape[0]

        # Compute feedforward activations in basal and somatic compartments.
        v_ff = torch.matmul(r_previous, self.weights.t())
        if self.bias is not None:
            v_ff += self.bias.unsqueeze(0).expand_as(v_ff)
        if self.use_recurrent_weights:
            v_ff = self.recurrent_activation(v_ff, r_current, turn_off_rec_norm_normalization)
        v = delta_v + v_ff 
        
        # Compute weight updates only using presynaptic activity information (generally not used)
        if presynaptic_inhibition:
            if rec_learning_neurons != 'suppressed_only':
                raise ValueError("'presynaptic_inhibition' requires 'rec_learning_neurons=suppressed_only'")
            teaching_signal_rec = - torch.ones(v.shape, device=v.device)
        else:
            teaching_signal_rec = abs(self.forward_activation_function(v)) - \
                                abs(self.forward_activation_function(v_ff))

        # Sparsify teaching signal to restrict learning to selected neurons.
        # Generally we learn only on suppressed neurons.
        if not suppressed_neurons is None:
            if rec_learning_neurons == 'active_only':
                teaching_signal_rec = teaching_signal_rec * (1 - suppressed_neurons.long())
            elif rec_learning_neurons == 'suppressed_only':
                teaching_signal_rec = teaching_signal_rec * suppressed_neurons.long()

        # Compute the gradients and actual updates.
        weights_rec_grad = - 2 * 1./batch_size * teaching_signal_rec.t().mm(abs(r_current))
        weight_rec_update = weights_rec_grad

        if rec_grad_normalization == 'incoming':
            weight_rec_update -= weight_rec_update.detach().mean(dim=1).reshape(-1, 1)
        elif rec_grad_normalization == 'outgoing':
            weight_rec_update -= weight_rec_update.detach().mean(dim=0)
        elif rec_grad_normalization == 'all':
            weight_rec_update -= weight_rec_update.detach()
        
        self._weights_recurrent.grad += weight_rec_update

    
    def compute_forward_gradients(self, delta_v, r_previous,
                                  r_current, scale=1.,
                                  saving_ndi_updates=False,
                                  learning_rule='nonlinear_difference',
                                  suppressed_neurons=None,
                                  freeze_suppressed_neuron_weights=False,
                                  fw_grad_normalization='',
                                  turn_off_rec_norm_normalization=False,
                                  hebbian_fw_learning=False,
                                  from_ff_learning=False):
        """Computes forward gradients using a local-in-time learning rule.

        This function applies a non-linear difference learning rule as described
        in Eq. (4) in the paper in https://arxiv.org/pdf/2204.07249.pdf. 
        Specifically, it compues the difference between the non-linear transformation
        of basal and somatic voltages.

        Moreover, it allows for selectively updating incoming weights of neurons
        that were excluded from the sparsified population, if sparsity is used.

        Depending on the option ``saving_ndi_updates`` these updates will be
        stored in different locations (see argument docstring).

        Args:
            delta_v: The feedback teaching signal from the controller.
            r_previous (torch.Tensor): The activations of the previous layer.
            r_current (torch.Tensor): The activations of the current layer.
            scale (float): Scaling factor for the gradients.
            saving_ndi_updates (boolean): Whether to save the non-dynamical
                inversion updates. When ``True``, computed updates are added to
                ``ndi_updates_weights`` (and bias) to later compare with the
                steady-state/continuous updates. When ``False``, computed
                updates are added to ``weights.grad`` (and bias), to be later
                updated.
            learning_rule (str): The type of learning rule.
            suppressed_neurons (torch.Tensor) Boolean tensor indicating which
                neurons were inactivated by the WTA sparsity.
            freeze_suppressed_neuron_weights (bool): Whether to freeze forward
                weights of inactivated neurons.
            fw_grad_normalization (str): Normalization procedure for forward
                weight updates.
            turn_off_rec_norm_normalization (bool): Whether to turn off rescaling of
                neuron activities after applying lateral inhibition.
            hebbian_fw_learning (bool): Whether to update incoming weights to non-sparsified
                neurons in a hebbian fashion rather than comparing their somatic to their
                feedforward activity. Generally not used.
            from_ff_learning (bool): Whether weight updates are performed with respect to pure
                feedforward activations as opposed to activations incorporating recurrent
                activity. Generally not used.  
        """
        batch_size = r_previous.shape[0]

        if learning_rule == "voltage_difference":
            teaching_signal_ff = 2 * (-delta_v)
        elif learning_rule == "nonlinear_difference":
            # Compute feedforward activations in basal and somatic compartments.
            v_ff = torch.matmul(r_previous, self.weights.t())
            if self.bias is not None:
                v_ff += self.bias.unsqueeze(0).expand_as(v_ff)
            if self.use_recurrent_weights:
                v_rec = self.recurrent_activation(v_ff, r_current, turn_off_rec_norm_normalization)
            else:
                v_rec = v_ff
            v = delta_v + v_rec 

            # Compute the teaching signal based on the basal-somatic difference.
            if hebbian_fw_learning:
                teaching_signal_ff = self.forward_activation_function(v)
            elif from_ff_learning:
                teaching_signal_ff = self.forward_activation_function(v) - \
                                     self.forward_activation_function(v_ff)
            else:
                teaching_signal_ff = self.forward_activation_function(v) - \
                                    self.forward_activation_function(v_rec)
        else:
            raise ValueError('The rule %s is not valid.' % learning_rule)

        # Optionally mask the teaching signal to freeze weights of inactivated neurons
        if not suppressed_neurons is None and freeze_suppressed_neuron_weights:
            teaching_signal_ff = teaching_signal_ff * (1 - suppressed_neurons.long())
        
        # compute the gradients and actual updates.
        weights_grad = - 2 * 1./batch_size * teaching_signal_ff.t().mm(r_previous)
        weight_update = scale * weights_grad.detach()
        if self.bias is not None:
            bias_grad = - 2 * teaching_signal_ff.mean(0)
            bias_update = scale * bias_grad.detach()

        # Optionally, center weight updates such that the mean of incoming weight updates is equal to 0
        if fw_grad_normalization == 'incoming':
            weight_update -= weights_grad.detach().mean(dim=1).reshape(-1, 1)

        # Store the updates appropriately.
        if saving_ndi_updates:
            self.ndi_updates_weights = weight_update
            if self.bias is not None:
                self.ndi_updates_bias = bias_update
        else:
            self._weights.grad += weight_update
            if self.bias is not None:
                self._bias.grad += bias_update

    def compute_forward_gradients_continuous(self, v_time, v_ff_time,
                                     r_previous_time, t_start=None, t_end=None, 
                                     learning_rule='nonlinear_difference'):
        r"""Computes forward gradients using an integration (sum) of voltage
        differences across comparments.

        This weight update is identical to ``compute_forward_gradients``
        except that it allows to integrate over more than one timestep.
        However, here the somatic and basal voltages are assumed to have been
        computed outside and provided as an input argument.

        Args:
            v_time: The somatic voltages at different timesteps.
            v_ff_time: The basal voltages at different timesteps.
            r_previous_time: The activations of the previous layer at different
                timesteps.
            t_start (int): The initial time index for the integration.
            t_end (int): The final time index for the integration.
            learning_rule (str): The type of learning rule.
        """
        batch_size = r_previous_time.shape[1]

        # Get the boundaries accross which to compute the summation.
        if t_start is None: 
            t_start = 0
        if t_end is None:
            t_end = v_time.shape[0]
        T = t_end - t_start

        if learning_rule == "voltage_difference":
            # Compute the teaching signal based on the voltage difference.
            teaching_signal = v_time[t_start:t_end] - v_ff_time[t_start:t_end]
        elif learning_rule == "nonlinear_difference":
            # Compute the teaching signal based on the basal-somatic difference.
            teaching_signal = \
                self.forward_activation_function(v_time[t_start:t_end]) - \
                self.forward_activation_function(v_ff_time[t_start:t_end])
        else:
            raise ValueError('The rule %s is not valid.' % learning_rule)

        # Compute the gradients.
        if self.bias is not None:
            bias_grad = -2 * 1. / T * torch.sum(teaching_signal, axis=0).mean(0)
        teaching_signal = teaching_signal.permute(0, 2, 1)
        weights_grad = -2 * 1. / batch_size * 1. / T * \
                    torch.sum(teaching_signal @ \
                              r_previous_time[t_start:t_end, :, :], axis=0)

        # Store the updates appropriately.
        if self.bias is not None:
            self._bias.grad = bias_grad.detach()
        self._weights.grad = weights_grad.detach()

    def compute_feedback_gradients_continuous(self, v_fb_time, u_time,
                                              t_start=None, t_end=None,
                                              sigma=1., beta=0., scaling=1.):
        r"""Computes feedback gradients using an integration (sum) of voltage.

        This weight update is identical to :meth:`compute_feedback_gradients`
        except that it allows to integrate over more than one timestep.

        It follows the differential equation:

        .. math::
            
            \frac{dQ_i}{dt} = -\mathbf{v}_i^\text{fb} \mathbf{u}(t)^T - \
                \beta Q_i

        Refer to :meth:`compute_feedback_gradients` for variable details.

        Note that pytorch saves the positive gradient, hence we should save
        :math:`-\Delta Q_i`.

        Args:
            v_fb_time (torch.Tensor): The apical compartment voltages over
                a certain time period.
            u_time (torch.Tensor): The control inputs over  certain time period.
            t_start (torch.Tensor): The start index from which the summation
                over time should start.
            t_end (torch.Tensor): The stop index at which the summation over
                time should stop.
            sigma (float): The standard deviation of the noise in the network
                dynamics. This is used to scale the fb weight update, such that
                its magnitude is independent of the noise variance.
            beta (float): The homeostatic weight decay parameter.
            scaling (float): In the theory for the feedback weight updates, the
                update for each layer should be scaled with
                :math:`(1+\tau_{v}/\tau_{\epsilon})^{L-i}`, with L the amount of
                layers and i the layer index. ``scaling`` should be the factor
                :math:`(1+\tau_{v}/\tau_{\epsilon})^{L-i}` for this layer.
        """
        batch_size = v_fb_time.shape[1]

        # Get the boundaries accross which to compute the summation.
        if t_start is None: 
            t_start = 0 
        if t_end is None:
            t_end = v_fb_time.shape[0]
        T = t_end - t_start

        # Compute the gradient scaling.
        if sigma < 0.01:
            scale = 1 / 0.01 ** 2
        else:
            scale = 1 / sigma ** 2
        scale *= scaling

        # Compute the update.
        feedbackweights_grad = scale/(T * batch_size) * \
                torch.sum(v_fb_time[t_start:t_end].permute(0,2,1) \
                          @ u_time[t_start:t_end], axis=0)
        feedbackweights_grad += beta * self._weights_backward

        self._weights_backward.grad = feedbackweights_grad.detach()

    def save_feedback_batch_logs(self, writer, step, name, no_gradient=False,
                                 pretraining=False):
        """Save feedback weight stats for the latest mini-batch.

        Args:
            writer (SummaryWriter): Summary writer from tensorboardX.
            step (int): The global step used for the x-axis of the plots.
            name (str): The name of the layer.
            no_gradient (bool): Flag indicating whether we should skip saving
                the gradients of the feedback weights.
            pretraining (bool): Flag indicating that the training is in the
                initialization phase (only training the feedback weights).
        """
        if pretraining:
            prefix = 'feedback_training/{}/'.format(name)
        else:
            prefix = name + '/'

        feedback_weights_norm = torch.norm(self.weights_backward)
        writer.add_scalar(tag=prefix + 'feedback_weights_norm',
                          scalar_value=feedback_weights_norm,
                          global_step=step)
        if self.weights_backward.grad is not None:
            feedback_weights_grad_norm = torch.norm(self.weights_backward.grad)
            writer.add_scalar(tag=prefix + 'feedback_weights_gradient_norm',
                              scalar_value=feedback_weights_grad_norm,
                              global_step=step)

    @property
    def name(self):
        return 'DFCLayer'
