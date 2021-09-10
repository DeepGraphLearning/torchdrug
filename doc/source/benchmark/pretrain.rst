Pretrained Molecular Representations
====================================

.. include:: ../bibliography.rst

This page contains benchmarks of property prediction models with pre-training.

We have two main methods for the pre-training.

- | Self-supervised pre-training is learning the graph structural information in a
  | self-supervised manner. Here we are pre-training on a subset of 2m molecules from
  | `ZINC15`_.
- | Supervised pre-training is doing pre-training on a large supervised dataset.
  | Here we are using 456k molecuels and 1,310 tasks from `ChEMBL`_.

For the downstream tasks, we consider the scaffold splitting for molecule data.
The split for train/validation/test sets is 80%:10%:10%. For each pre-training
method and downstream dataset, we evaluate with 10 random splits and report the
mean and the derivation of AUROC metric.

+----------------+-----------+-----------+------------+-----------+------------+-----------+-----------+-----------+------+
|                | `BBBP`_   | `Tox21`_  | `ToxCast`_ | `SIDER`_  | `ClinTox`_ | `MUV`_    | `HIV`_    | `BACE`_   | Avg. |
+================+===========+===========+============+===========+============+===========+===========+===========+======+
| No Pretrain    | 67.1(2.9) | 75.0(0.2) | 60.6(0.7)  | 58.9(0.8) | 60.8(3.9)  | 64.3(3.4) | 76.4(1.6) | 66.5(9.0) | 66.2 |
+----------------+-----------+-----------+------------+-----------+------------+-----------+-----------+-----------+------+
| `InfoGraph`_   | 68.9(0.6) | 76.4(0.4) | 71.2(0.6)  | 59.8(0.7) | 70.3(4.2)  | 69.4(0.8) | 75.5(0.7) | 73.7(2.6) | 70.7 |
+----------------+-----------+-----------+------------+-----------+------------+-----------+-----------+-----------+------+
| `EdgePred`_    | 67.1(2.6) | 74.6(0.7) | 69.8(0.5)  | 59.4(1.5) | 59.0(2.6)  | 66.8(1.0) | 76.3(2.0) | 68.4(3.9) | 67.7 |
+----------------+-----------+-----------+------------+-----------+------------+-----------+-----------+-----------+------+
| `AttrMasking`_ | 65.2(0.9) | 75.8(0.5) | 70.6(0.6)  | 58.9(0.9) | 79.0(2.3)  | 68.3(2.1) | 76.9(0.9) | 78.1(0.8) | 71.6 |
+----------------+-----------+-----------+------------+-----------+------------+-----------+-----------+-----------+------+
| `ContextPred`_ | 71.1(1.8) | 75.6(0.3) | 71.1(0.3)  | 61.7(0.5) | 65.9(1.9)  | 68.5(0.6) | 77.1(0.3) | 78.6(0.5) | 71.2 |
+----------------+-----------+-----------+------------+-----------+------------+-----------+-----------+-----------+------+
