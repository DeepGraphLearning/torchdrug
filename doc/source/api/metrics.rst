torchdrug.metrics
=================

.. currentmodule:: torchdrug.metrics


Basic Metrics
-------------

AUROC
^^^^^
.. autofunction:: area_under_roc
.. function:: AUROC

    alias of ``torchdrug.metrics.area_under_roc``


AUPRC
^^^^^
.. autofunction:: area_under_prc
.. function:: AUPRC

    alias of ``torchdrug.metrics.area_under_prc``

R2
^^
.. autofunction:: r2

Accuracy
^^^^^^^^
.. autofunction:: accuracy

Matthews Correlation Coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofuction:: matthews_corrcoef

Pearson Correlation Coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: pearsonr

Spearman Correlation Coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: spearmanr


Chemical Metrics
----------------

SA
^^
.. autofunction:: SA

QED
^^^
.. autofunction:: QED

Chemical Validity
^^^^^^^^^^^^^^^^^
.. autofunction:: chemical_validity

LogP
^^^^
.. autofunction:: logP

Penalized LogP
^^^^^^^^^^^^^^
.. autofunction:: penalized_logP
