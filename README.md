[![TorchDrug](asset/logo.svg)](https://torchdrug.ai/)

----------------------------

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Tbnr1Fog_YjkqU1MOhcVLuxqZ4DC-c8-#forceEdit=true&sandboxMode=true)
[![Contributions](https://img.shields.io/badge/contributions-welcome-blue)](https://github.com/DeepGraphLearning/torchdrug/blob/master/CONTRIBUTING.md)
[![License Apache-2.0](https://img.shields.io/github/license/DeepGraphLearning/torchdrug?color=blue)](https://github.com/DeepGraphLearning/torchdrug/blob/master/LICENSE)
[![TorchDrug Twitter](https://img.shields.io/twitter/url?label=TorchDrug&style=social&url=https%3A%2F%2Ftwitter.com%2FDrugTorch)](https://twitter.com/DrugTorch)

[Docs] | [Tutorials] | [Benchmarks] | [Papers Implemented]

[Docs]: https://deepgraphlearning.github.io/torchdrug-site/docs
[Tutorials]: https://deepgraphlearning.github.io/torchdrug-site/docs/tutorials
[Benchmarks]: https://deepgraphlearning.github.io/torchdrug-site/docs/benchmark
[Papers Implemented]: https://deepgraphlearning.github.io/torchdrug-site/docs/paper

TorchDrug is a [PyTorch]-based machine learning toolbox designed for several purposes.

- Easy implementation of graph operations in a PyTorchic style with GPU support
- Being friendly to practitioners with minimal knowledge about drug discovery
- Rapid prototyping of machine learning research

[PyTorch]: https://pytorch.org/

Installation
------------

TorchDrug can be installed on either Linux, Windows or macOS. It is compatible with
Python 3.7/3.8 and PyTorch >= 1.4.0.

### From Conda ###

```bash
conda install -c milagraph -c conda-forge torchdrug
```

### From Pip ###

```bash
pip3 install torch==1.9.0
pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip3 install torchdrug
```

To install `torch-scatter` for other PyTorch or CUDA versions, please see the
instructions in https://github.com/rusty1s/pytorch_scatter

### From Source ###

```bash
git clone https://github.com/DeepGraphLearning/torchdrug
cd torchdrug
pip install -r requirements.txt
python setup.py install
```

### Windows (PowerShell) ###

We need to first install the build tools for Visual Studio. We then install the
following modules in PowerShell.

```powershell
Install-Module Pscx -AllowClobber
Install-Module VSSetup
```

Initialize Visual Studio in PowerShell with the following commands. We may setup
this for all PowerShell sessions by writing it to the PowerShell profile. Change
the library path according to your own case.

```powershell
Import-VisualStudioVars -Architecture x64
$env:LIB += ";C:\Program Files\Python37\libs"
```

Quick Start
-----------

TorchDrug is designed for humans and focused on graph structured data.
It enables easy implementation of graph operations in machine learning models.
All the operations in TorchDrug are backed by [PyTorch] framework, and support GPU acceleration and auto differentiation.

```python
from torchdrug import data

edge_list = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
graph = data.Graph(edge_list, num_node=6)
graph = graph.cuda()
# the subgraph induced by nodes 2, 3 & 4
subgraph = graph.subgraph([2, 3, 4])
```

Molecules are also supported in TorchDrug. You can get the desired molecule properties without any domain knowledge.

```python
mol = data.Molecule.from_smiles("CCOC(=O)N", node_feature="default", edge_feature="default")
print(mol.node_feature)
print(mol.atom_type)
print(mol.to_scaffold())
```

You may also register custom node, edge or graph attributes. They will be automatically processed during indexing operations.

```python
with mol.edge():
	mol.is_CC_bond = (mol.edge_list[:, :2] == td.CARBON).all(dim=-1)
sub_mol = mol.subgraph(mol.atom_type != td.NITROGEN)
print(sub_mol.is_CC_bond)
```

TorchDrug provides a wide range of common datasets and building blocks for drug discovery.
With minimal code, you can apply standard models to solve your own problem.

```python
import torch
from torchdrug import datasets

dataset = datasets.Tox21()
dataset[0].visualize()
lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
lengths += [len(dataset) - sum(lengths)]
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)
```

```python
from torchdrug import models, tasks

model = models.GIN(dataset.node_feature_dim, hidden_dims=[256, 256, 256, 256])
task = tasks.PropertyPrediction(model, task=dataset.tasks)
```

Training and inference are accelerated by multiple CPUs or GPUs.
This can be seamlessly switched in TorchDrug by just a line of code.
```python
from torchdrug import core

# Single CPU / Multiple CPUs / Distributed CPUs
solver = core.Engine(task, train_set, valid_set, test_set, optimizer)
# Single GPU
solver = core.Engine(task, train_set, valid_set, test_set, optimizer, gpus=[0])
# Multiple GPUs
solver = core.Engine(task, train_set, valid_set, test_set, optimizer, gpus=[0, 1, 2, 3])
# Distributed GPUs
solver = core.Engine(task, train_set, valid_set, test_set, optimizer, gpus=[0, 1, 2, 3, 0, 1, 2, 3])
```

Contributing
------------

Everyone is welcome to contribute to the development of TorchDrug.
Please refer to [contributing guidelines](CONTRIBUTING.md) for more details.

License
-------

TorchDrug is released under [Apache-2.0 License](LICENSE).
