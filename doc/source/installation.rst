Installation
============

TorchDrug is compatible with Python 3.7/3.8 and PyTorch >= 1.4.0.

From Conda
----------

.. code:: bash

    conda install -c milagraph -c conda-forge torchdrug

From Pip
--------

First, let's install PyTorch.

.. code:: bash

    pip3 install torch

To install ``torch-scatter``, we need to check the version of PyTorch and CUDA.

We can get the version of PyTorch by ``python3 -c "import torch; print(torch.__version__")``.
The version of CUDA can be get by ``nvcc -V``. For example, if our PyTorch is ``1.8.0``
and CUDA is ``10.2``, the command should be

.. code:: bash

    pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html

Replace the versions in the above url according to your case. If you don't have GPUs
or CUDA installed, treat the CUDA version in the url as ``cpu``. See
https://github.com/rusty1s/pytorch_scatter for more details about installation.

Finally, install TorchDrug via

.. code:: bash

    pip3 install torchdrug

From Source
-----------

.. code:: bash

    git clone https://github.com/DeepGraphLearning/torchdrug
    cd torchdrug
    pip install -r requirements.txt
    python setup.py install
