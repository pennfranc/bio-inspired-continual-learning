[![DOI](https://zenodo.org/badge/570550727.svg)](https://zenodo.org/badge/latestdoi/570550727)

# Sparse-Recurrent DFC

This codebase contains the implementation of sparse-recurrent DFC, as well as support for continual learning on split-MNIST in both the domain-IL and class-IL learning paradigms. Moreover, this repository also contains code adapted from this [continual learning benchmark library](https://github.com/GT-RIPL/Continual-Learning-Benchmark) to compare CL performance of DFC variants against Elastic Weight Consolidation and Synaptic Intelligence. Please note that both the underlying DFC repository and the continual learning benchmark repository have been modified independently of their original authors.

## Requirements

The necessary conda environment to run the code is provided in the file [dfc_environment.yml](dfc_environment.yml). To generate the environment type `conda env create -f dfc_environment.yml` and activate it with `conda activate DFC_cuda_env`. Note that you'll also need to install an adapted version of the [hypnettorch library](https://github.com/pennfranc/hypnettorch). We recommend cloning the repository, and running `pip install .` in the root directory of the repository.

## Running experiments

For recreating all of the performance-related data of standard, sparse, recurrent, sparse-recurrent DFC, as well as BP, EWC and SI, please go to `dfc/` and run `python run_hpsearches.py`. The results will be saved in `dfc/out`. The script will utilize CUDA GPU resources, if available.
For recreating the activation recordings used for the representational overlap, hyperplane and dimensionality analyses, please got to `dfc/` and run `python run_activation_recording.py --cl_mode=<mode>`, where `<mode>` should be chosen as `domain` for the domain-IL paradigm, and `class` for the class-IL paradigm. The script will utilize CUDA GPU resources, if available. Please note that the class-IL activations take up a significant amount of memory (close to 30GB).

## Reproducing figures

Having run all scripts specified above to obtain the data, all figures can be created by running the notebooks contained in `dfc/notebooks`.

# Deep Feedback Control

Implementation of Deep Feedback Control and some extensions. The basis of this repository is a cleaned-up version of this 
[public repository](https://github.com/meulemansalex/deep_feedback_control/tree/main).

## Documentation

How to build the documentation is explained in the `docs` folder.

## Running experiments

For running experiments, move to the `dfc` subfolder. Further instructions can be found on the [README](dfc/README.rst) there.

## Citation

When using this package in your research project, please consider citing one of our papers for which this package has been developed.

```
@inproceedings{Meulemans2021Dec,
   title={Credit Assignment in Neural Networks through Deep Feedback Control},
   author={Alexander Meulemans and Matilde Tristany Farinha and Javier Garcia Ordonez and Pau Vilimelis Aceituno and Joao Sacramento and Benjamin F. Grewe},
   booktitle={Advances in Neural Information Processing Systems},
   year={2021},
   url={https://arxiv.org/abs/2106.07887}
}
```

```
@misc{https://doi.org/10.48550/arxiv.2204.07249,
  title = {Minimizing Control for Credit Assignment with Strong Feedback},
  author = {Meulemans, Alexander and Farinha, Matilde Tristany and Cervera, Maria R. and Sacramento, João and Grewe, Benjamin F.},
  publisher = {arXiv},
  year = {2022},
  url = {https://arxiv.org/abs/2204.07249},
}

```
