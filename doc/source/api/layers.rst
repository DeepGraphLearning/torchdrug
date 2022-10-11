torchdrug.layers
================

.. currentmodule:: torchdrug.layers

Common Layers
-------------

GaussianSmearing
^^^^^^^^^^^^^^^^
.. autoclass:: GaussianSmearing
    :members:

MultiLayerPerceptron
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: MultiLayerPerceptron
    :members:

MutualInformation
^^^^^^^^^^^^^^^^^
.. autoclass:: MutualInformation
    :members:

SinusoidalPositionEmbedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SinusoidalPositionEmbedding
    :members:

PairNorm
^^^^^^^^
.. autoclass:: PairNorm
    :members:

Sequential
^^^^^^^^^^
.. autoclass:: Sequential
    :members:


Convolution Layers
------------------

.. currentmodule:: torchdrug.layers

.. autoclass:: MessagePassingBase
    :members:

ChebyshevConv
^^^^^^^^^^^^^
.. autoclass:: ChebyshevConv
    :members:

ContinuousFilterConv
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ContinuousFilterConv
    :members:

GeometricRelationalGraphConv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GeometricRelationalGraphConv
    :members:

GraphAttentionConv
^^^^^^^^^^^^^^^^^^
.. autoclass:: GraphAttentionConv
    :members:

GraphConv
^^^^^^^^^
.. autoclass:: GraphConv
    :members:

GraphIsomorphismConv
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GraphIsomorphismConv
    :members:

MessagePassing
^^^^^^^^^^^^^^
.. autoclass:: MessagePassing
    :members:

NeuralFingerprintConv
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: NeuralFingerprintConv
    :members:

RelationalGraphConv
^^^^^^^^^^^^^^^^^^^
.. autoclass:: RelationalGraphConv
    :members:


Readout Layers
--------------

MeanReadout
^^^^^^^^^^^
.. autoclass:: MeanReadout
    :members:

SumReadout
^^^^^^^^^^
.. autoclass:: SumReadout
    :members:

MaxReadout
^^^^^^^^^^
.. autoclass:: MaxReadout
    :members:

Set2Set
^^^^^^^
.. autoclass:: Set2Set
    :members:

Softmax
^^^^^^^
.. autoclass:: Softmax
    :members:

Sort
^^^^^^^
.. autoclass:: Sort
    :members:


Pooling Layers
--------------

DiffPool
^^^^^^^^
.. autoclass:: DiffPool
    :members:

MinCutPool
^^^^^^^^^^
.. autoclass:: MinCutPool
    :members:


Sampler Layers
-------------------

EdgeSampler
^^^^^^^^^^^
.. autoclass:: EdgeSampler
    :members:

NodeSampler
^^^^^^^^^^^
.. autoclass:: NodeSampler
    :members:


Flow Layers
-----------

ConditionalFlow
^^^^^^^^^^^^^^^
.. autoclass:: ConditionalFlow
    :members:


Sequence Encoder Blocks
-----------------------

ProteinResNetBlock
^^^^^^^^^^^^^^^^^^
.. autoclass:: ProteinResNetBlock
    :members:

SelfAttentionBlock
^^^^^^^^^^^^^^^^^^
.. autoclass:: SelfAttentionBlock
    :members:

ProteinBERTBlock
^^^^^^^^^^^^^^^^
.. autoclass:: ProteinBERTBlock
    :members:


Distribution Layers
-------------------

These layers belong to `torchdrug.layers.distribution`.

.. currentmodule:: torchdrug.layers.distribution

IndependentGaussian
^^^^^^^^^^^^^^^^^^^
.. autoclass:: IndependentGaussian
    :members:


Graph Construction Layers
-------------------------

These layers belong to `torchdrug.layers.geometry`.

.. currentmodule:: torchdrug.layers.geometry

GraphConstruction
^^^^^^^^^^^^^^^^^
.. autoclass:: GraphConstruction
    :members:

SpatialLineGraph
^^^^^^^^^^^^^^^^
.. autoclass:: SpatialLineGraph
    :members:

BondEdge
^^^^^^^^
.. autoclass:: BondEdge
    :members:

KNNEdge
^^^^^^^
.. autoclass:: KNNEdge
    :members:

SpatialEdge
^^^^^^^^^^^
.. autoclass:: SpatialEdge
    :members:

SequentialEdge
^^^^^^^^^^^^^^
.. autoclass:: SequentialEdge
    :members:

AlphaCarbonNode
^^^^^^^^^^^^^^^
.. autoclass:: AlphaCarbonNode
    :members:

IdentityNode
^^^^^^^^^^^^
.. autoclass:: IdentityNode
    :members:

RandomEdgeMask
^^^^^^^^^^^^^^
.. autoclass:: RandomEdgeMask
    :members:

SubsequenceNode
^^^^^^^^^^^^^^^
.. autoclass:: SubsequenceNode
    :members:

SubspaceNode
^^^^^^^^^^^^
.. autoclass:: SubspaceNode
    :members:


Functional Layers
-----------------

These layers belong to `torchdrug.layers.functional`.

.. currentmodule:: torchdrug.layers.functional

Embedding Score Functions
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: transe_score

.. autofunction:: distmult_score

.. autofunction:: complex_score

.. autofunction:: simple_score

.. autofunction:: rotate_score

Sparse Matrix Multiplication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: generalized_spmm

.. autofunction:: generalized_rspmm

Variadic
^^^^^^^^
.. autofunction:: variadic_sum

.. autofunction:: variadic_mean

.. autofunction:: variadic_max

.. autofunction:: variadic_cross_entropy

.. autofunction:: variadic_log_softmax

.. autofunction:: variadic_softmax

.. autofunction:: variadic_sort

.. autofunction:: variadic_topk

.. autofunction:: variadic_arange

.. autofunction:: variadic_randperm

.. autofunction:: variadic_sample

.. autofunction:: variadic_meshgrid

.. autofunction:: variadic_to_padded

.. autofunction:: padded_to_variadic

Tensor Reduction
^^^^^^^^^^^^^^^^
.. autofunction:: masked_mean

.. autofunction:: mean_with_nan

Tensor Construction
^^^^^^^^^^^^^^^^^^^
.. autofunction:: as_mask

.. autofunction:: one_hot

.. autofunction:: multi_slice

.. autofunction:: multi_slice_mask

Sampling
^^^^^^^^
.. autofunction:: multinomial

Activation
^^^^^^^^^^
.. autofunction:: shifted_softplus
