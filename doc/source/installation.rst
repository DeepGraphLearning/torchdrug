Installation
============

TorchDrug can be installed on either Linux, Windows or macOS. It is compatible with
Python 3.7/3.8/3.9 and PyTorch >= 1.8.0.

For Windows

From Conda
----------

.. code:: bash

    conda install torchdrug -c milagraph -c conda-forge -c pytorch -c pyg

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

    pip3 install torch-scatter torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html

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

Windows (PowerShell)
--------------------

For Windows, we need to additionally install Visual Studio to enable the JIT
compilation. If you don't have Visual Studio installed, you can get a minimal
version of the build tools for Visual Studio at
https://visualstudio.microsoft.com/downloads/.

.. note::

    For non-English systems, you must select English as the language for Visual
    Studio. Otherwise, the ANSI encoding of Windows will cause errors in Python.

We then setup a command-line environment for JIT compilation. Launch a PowerShell
as administrator, and install the following extensions.

.. code:: powershell

    Install-Module Pscx -AllowClobber
    Install-Module VSSetup

Initialize Visual Studio in PowerShell with the following commands. We need to
change the library path based on our own Python path.

.. code:: powershell

    Import-VisualStudioVars -Architecture x64
    $env:LIB += ";C:\Program Files\Python37\libs"

The above commands should be executed for every PowerShell session. To setup this
for all PowerShell sessions, we can write them to the PowerShell profile. The
profile can be found by the ``$profile`` command in PowerShell. You may need to
create the profile if you use it for the first time.