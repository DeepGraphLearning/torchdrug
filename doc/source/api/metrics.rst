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
.. autofunction:: matthews_corrcoef

Pearson Correlation Coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: pearsonr

Spearman's Rank Correlation Coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: spearmanr

Variadic Accuracy
^^^^^^^^^^^^^^^^^
.. autofunction:: variadic_accuracy

Variadic Area Under ROC
^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: variadic_area_under_roc

Variadic Area Under PRC
^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: variadic_area_under_prc

Variadic Top Precision
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: variadic_top_precision

F1 Max
^^^^^^
.. autofunction:: f1_max


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
