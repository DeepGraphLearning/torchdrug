Installation
============

TorchDrug is compatible with Python 3.7/3.8 and PyTorch >= 1.4.0.

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


From Pip (Not Recommended)
-----------------------------------

This is not a recommended way to install TorchDrug, since we have noticed some
bugs in RDKit installed via pip.

Create venv
^^^^^^^^^^^

.. code:: bash

    mkdir ~/.venv
    /usr/bin/python3 -m venv ~/.venv/torchdrug
    source ~/.venv/torchdrug/bin/activate

Install requirements
^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    # Contents in requirements.txt
    torch>=1.4.0
    decorator<5,>=4.3
    numpy>=1.11
    matplotlib
    tqdm
    networkx
    ninja
    jinja2
    rdkit-pypi

.. code:: bash

    pip install -r requirements.txt

Install PyTorch Scatter
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    # PyTorch 1.9.0
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html


.. code:: bash

    # PyTorch 1.8.0/1.8.1
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cpu.html


**Note:** Binaries of older versions are also provided for PyTorch 1.4.0, PyTorch 1.5.0,
PyTorch 1.6.0 and PyTorch 1.7.0/1.7.1 (following the same procedure).

See https://github.com/rusty1s/pytorch_scatter for more details.

Install TorchDrug
^^^^^^^^^^^^^^^^^

.. code:: bash

    git clone https://github.com/DeepGraphLearning/torchdrug
    cd torchdrug
    python setup.py install