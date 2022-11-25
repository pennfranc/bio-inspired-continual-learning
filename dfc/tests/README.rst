Tests
=====

.. content-inclusion-marker-do-not-remove

Tests can be run from the ``dfc`` directory as follows:

.. code-block:: console

    $  python3 -m pytest

For this you need to install ``pytest`` using:

.. code-block:: console

    $  pip install pytest

Note that the results of some tests are extremely sensitive and might be machine-dependent. This doesn't necessarily mean that the code is broken. Notably, our tests pass in Dalco7 and Dalco8.