***********************
Hyperparameter searches
***********************

.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.
.. content-inclusion-marker-do-not-remove

This module allows running hyperparameter searches.

Preparing the config file for the search
########################################

For preparing a hyperparameter search, you need to create a hyperparameter configuration file similar to :mod:`hpconfig_template`.
In particular, make sure to modify the following accordingly:

* Add the lists of desired hyperparameters to be explored by filling the ``grid``
* Don't forget to select the correct ``dataset``
* Add any necessary ``conditions`` that need to be taken into account to avoid duplicate runs
* Specify the correct ``_SCRIPT_NAME``
* At the very bottom when calling ``parse_cmd_arguments`` don't forget to specify the desired network type

Running the hyperparameter search
#################################

For running the search, you need to run from the root directory:

.. code-block:: console

  $ python -m hpsearch.hpsearch --grid_module=hpsearch.hpconfig_template

Running the hyperparameter search in Dalco
##########################################

For running searches in Dalco execute:

.. code-block:: console

  $ python -m hpsearch.hpsearch --grid_module=hpsearch.hpconfig_template --visible_gpus=0,1

where the specific GPUs to be used can be provided.

Running the hyperparameter search in Euler
##########################################

For running searches in Euler, manually copy any necessary data into the ``data`` folder, and then execute:

.. code-block:: console

  $ bsub -n 1 -W 1:00 -e search_name.err -o search_name.out -R "rusage[mem=8000]" python3 -m hpsearch.hpsearch --grid_module=hpsearch.hpconfig_template --run_cluster --num_jobs=10 --num_hours=1 --num_searches=2 --resources="\"rusage[mem=8000, ngpus_excl_p=1]\""

Important choices to make for the overall search are: 

* total amount of time for the search (here ``-W 1:00`` indicates 1 hour)
* amount of resources for building the grid (here ``-R "rusage[mem=8000]"`` indicates 8Gb which should be plenty).

Important choices to make for individual runs are:

* the number of jobs that can be scheduled in parallel at any one time (``--num_jobs=10``)
* the maximum number of hours (``--num_hours=1``)
* the maximum number of runs (``--num_searches=2``)
* the resources per run (``--resources="\"rusage[mem=8000, ngpus_excl_p=1]\""`` indicates 8Gb and 1GPU per run)

Furthermore, if the gathering or results into the summary file failed one can post-hoc run the following post-processing script:

.. code-block:: console

  $ python3 -m hpsearch.hpsearch_postprocessing --grid_module=hpsearch.hpconfig_template OUT_DIR

where ``OUT_DIR`` should be replaced by the location of the hyperparameter search results.