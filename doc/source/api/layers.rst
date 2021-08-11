torchdrug.layers
================

Common Layers
-------------

.. currentmodule:: torchdrug.layers

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


Distribution Layers
-------------------

These layers belong to `torchdrug.layers.distribution`.

.. currentmodule:: torchdrug.layers.distribution

IndependentGaussian
^^^^^^^^^^^^^^^^^^^
.. autoclass:: IndependentGaussian
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

.. autofunction:: variadic_topk

Tensor Reduction
^^^^^^^^^^^^^^^^
.. autofunction:: masked_mean

.. autofunction:: mean_with_nan

Tensor Construction
^^^^^^^^^^^^^^^^^^^
.. autofunction:: as_mask

.. autofunction:: one_hot

.. autofunction:: multi_slice_mask

Sampling
^^^^^^^^
.. autofunction:: multinomial

Activation
^^^^^^^^^^
.. autofunction:: shifted_softplus
