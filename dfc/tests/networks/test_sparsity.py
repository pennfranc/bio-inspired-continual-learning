import numpy as np
import torch
import torch.nn as nn

from tests.utils import reset_seed, reset_layers, run_forward_pass
from networks.dfc_network_utils import get_layer_sparsity, sparsifier

torch.set_default_dtype(torch.float64)

def test_get_layer_sparsity():
    # test that layer sparsity is 0 at the start
    assert get_layer_sparsity(2, 0, 1) == 0

    # test that layer sparsity is maximal in the end
    assert get_layer_sparsity(2, 1, 1) == 2

    # test that layer sparsity is half of max in the middle
    assert get_layer_sparsity(2, 1, 2) == 1

    # test that layer sparsity increases with time
    assert get_layer_sparsity(5, 20, 100) < get_layer_sparsity(5, 21, 100)

def test_sparsifier():
    ### step

    # create mock layer activations with batch_size=2 and num_neurons=5
    # test 80% sparsity
    layer_v = torch.Tensor([
        [1, 0, 0, 0, 0],
        [-1, 1.1, 0, 0, 0]
    ])
    layer_v_sparse = sparsifier(layer_v, layer_v, 0.8)
    layer_v_sparse_expected = torch.Tensor([
        [1, 0, 0, 0, 0],
        [0, 1.1, 0, 0, 0]
    ])
    assert torch.equal(layer_v_sparse, layer_v_sparse_expected)

    # test 0% sparsity
    layer_v_sparse = sparsifier(layer_v, layer_v, 0)
    layer_v_sparse_expected = layer_v
    assert torch.equal(layer_v_sparse, layer_v_sparse_expected)

    # test 20% sparsity
    layer_v = torch.Tensor([
        [1, 0.8, 0.6, 0.9, 0.9],
        [1, 1.1, 2, -2, 2]
    ])
    layer_v_sparse = sparsifier(layer_v, layer_v, 0.2)
    layer_v_sparse_expected = torch.Tensor([
        [1, 0.8, 0, 0.9, 0.9],
        [0, 1.1, 2, -2, 2]
    ])
    assert torch.equal(layer_v_sparse, layer_v_sparse_expected)

    # test 50% on big, random tensor
    layer_v = torch.rand(50, 100)
    layer_v_sparse = sparsifier(layer_v, layer_v, 0.5)
    assert (layer_v_sparse > 0).sum() - 2500 < 5

    # test 100% on big, random tensor
    layer_v = torch.rand(50, 200)
    layer_v_sparse = sparsifier(layer_v, layer_v, 1)
    assert (layer_v_sparse > 0).sum() < 51

    # test 0% on big, random tensor
    layer_v = torch.rand(50, 100)
    layer_v_sparse = sparsifier(layer_v, layer_v, 0)
    assert (layer_v_sparse == 0).sum() < 1



    