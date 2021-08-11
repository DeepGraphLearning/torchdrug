Installation
============

TorchDrug is compatible with Python >= 3.5 and PyTorch >= 1.4.0.

From Conda
----------

.. code:: bash

    conda install -c milagraph -c conda-forge torchdrug

From Source
-----------

TorchDrug depends on rdkit, which is only available via conda.
You can install rdkit with the following line.

.. code:: bash

    conda install -c conda-forge rdkit

.. code:: bash

    git clone https://github.com/DeepGraphLearning/torchdrug
    cd torchdrug
    pip install -r requirements.txt
    python setup.py install