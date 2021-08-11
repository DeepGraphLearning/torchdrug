Molecule Property Prediction
============================

.. include:: ../bibliography.rst

This page contains benchmarks of popular property prediction models.

We consider both vanilla random split and scaffold-based random split for molecule
datasets. The split for train/validation/test sets is 80%:10%:10%. For each model, we
evaluate it with 5 random splits and report the mean and the deviation of the
performance. We report MAE and R2 metrics for regression datasets, AUROC and AURPC
metrics for binary classification datasets.

For datasets containing a lot of tasks, only the first 16 tasks are showed here due
to space limitations.

Random Split
------------

QM9
^^^

.. image:: ../../../asset/benchmark/qm9_mae.png
.. image:: ../../../asset/benchmark/qm9_r2.png

QM8
^^^

.. image:: ../../../asset/benchmark/qm8_mae.png
.. image:: ../../../asset/benchmark/qm8_r2.png

BACE
^^^^

.. image:: ../../../asset/benchmark/bace_auroc.png
.. image:: ../../../asset/benchmark/bace_auprc.png

BBBP
^^^^

.. image:: ../../../asset/benchmark/bbbp_auroc.png
.. image:: ../../../asset/benchmark/bbbp_auprc.png

CEP
^^^

.. image:: ../../../asset/benchmark/cep_mae.png
.. image:: ../../../asset/benchmark/cep_r2.png

HIV
^^^

.. image:: ../../../asset/benchmark/hiv_auroc.png
.. image:: ../../../asset/benchmark/hiv_auprc.png

ClinTox
^^^^^^^

.. image:: ../../../asset/benchmark/clintox_auroc.png
.. image:: ../../../asset/benchmark/clintox_auprc.png

ESOL
^^^^

.. image:: ../../../asset/benchmark/esol_mae.png
.. image:: ../../../asset/benchmark/esol_r2.png

FreeSolv
^^^^^^^^

.. image:: ../../../asset/benchmark/freesolv_mae.png
.. image:: ../../../asset/benchmark/freesolv_r2.png

Lipophilicity
^^^^^^^^^^^^^

.. image:: ../../../asset/benchmark/lipophilicity_mae.png
.. image:: ../../../asset/benchmark/lipophilicity_r2.png

SIDER
^^^^^

.. image:: ../../../asset/benchmark/sider_auroc.png
.. image:: ../../../asset/benchmark/sider_auprc.png

Tox21
^^^^^

.. image:: ../../../asset/benchmark/tox21_auroc.png
.. image:: ../../../asset/benchmark/tox21_auprc.png

ToxCast
^^^^^^^

.. image:: ../../../asset/benchmark/toxcast_auroc.png
.. image:: ../../../asset/benchmark/toxcast_auprc.png

MUV
^^^

.. image:: ../../../asset/benchmark/muv_auroc.png
.. image:: ../../../asset/benchmark/muv_auprc.png


Malaria
^^^^^^^

.. image:: ../../../asset/benchmark/malaria_mae.png
.. image:: ../../../asset/benchmark/malaria_r2.png

Scaffold Split
--------------

QM9
^^^

.. image:: ../../../asset/benchmark/qm9_scaffold_mae.png
.. image:: ../../../asset/benchmark/qm9_scaffold_r2.png

QM8
^^^

.. image:: ../../../asset/benchmark/qm8_scaffold_mae.png
.. image:: ../../../asset/benchmark/qm8_scaffold_r2.png

BACE
^^^^

.. image:: ../../../asset/benchmark/bace_scaffold_auroc.png
.. image:: ../../../asset/benchmark/bace_scaffold_auprc.png

BBBP
^^^^

.. image:: ../../../asset/benchmark/bbbp_scaffold_auroc.png
.. image:: ../../../asset/benchmark/bbbp_scaffold_auprc.png

CEP
^^^

.. image:: ../../../asset/benchmark/cep_scaffold_mae.png
.. image:: ../../../asset/benchmark/cep_scaffold_r2.png

HIV
^^^

.. image:: ../../../asset/benchmark/hiv_scaffold_auroc.png
.. image:: ../../../asset/benchmark/hiv_scaffold_auprc.png

ClinTox
^^^^^^^

.. image:: ../../../asset/benchmark/clintox_scaffold_auroc.png
.. image:: ../../../asset/benchmark/clintox_scaffold_auprc.png

ESOL
^^^^

.. image:: ../../../asset/benchmark/esol_scaffold_mae.png
.. image:: ../../../asset/benchmark/esol_scaffold_r2.png

FreeSolv
^^^^^^^^

.. image:: ../../../asset/benchmark/freesolv_scaffold_mae.png
.. image:: ../../../asset/benchmark/freesolv_scaffold_r2.png

Lipophilicity
^^^^^^^^^^^^^

.. image:: ../../../asset/benchmark/lipophilicity_scaffold_mae.png
.. image:: ../../../asset/benchmark/lipophilicity_scaffold_r2.png

SIDER
^^^^^

.. image:: ../../../asset/benchmark/sider_scaffold_auroc.png
.. image:: ../../../asset/benchmark/sider_scaffold_auprc.png

Tox21
^^^^^

.. image:: ../../../asset/benchmark/tox21_scaffold_auroc.png
.. image:: ../../../asset/benchmark/tox21_scaffold_auprc.png

ToxCast
^^^^^^^

.. image:: ../../../asset/benchmark/toxcast_scaffold_auroc.png
.. image:: ../../../asset/benchmark/toxcast_scaffold_auprc.png

MUV
^^^

.. image:: ../../../asset/benchmark/muv_scaffold_auroc.png
.. image:: ../../../asset/benchmark/muv_scaffold_auprc.png

Malaria
^^^^^^^

.. image:: ../../../asset/benchmark/malaria_scaffold_mae.png
.. image:: ../../../asset/benchmark/malaria_scaffold_r2.png