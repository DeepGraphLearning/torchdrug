Molecule Generation
===================

.. include:: ../bibliography.rst

This page contains benchmarks of graph generative models for goal-directed property
optimization, which is aimed at generating novel molecules with optimized chemical
properties. We first pretrain the models on `ZINC250k`_ dataset, and then apply the
reinforcement learning algorithms to finetune the networks towards desired chemical
properties.

We choose penalized logP and QED score as our target property.

- Penalized logP score is the octanol-water partition coefficient penalized by the
  synthetic accessibility score and the number of long cycles.
- QED score measures the drug-likeness of the molecule.

We report the top-1 property scores of generated molecules by different models in
the following table. We also report the top-1 property scores of molecules in
`ZINC250k`_ dataset for reference. The maximum graph size is set as 38, which is the
same as the maximum graph size of molecules in `ZINC250k`_.

+----------------+-----------------------+---------+------------+
|                | `ZINC250k`_ (Dataset) | `GCPN`_ | `GraphAF`_ |
+================+=======================+=========+============+
| Penalized LogP | 4.52                  | 6.560   | 5.630      |
+----------------+-----------------------+---------+------------+
| QED            | 0.948                 | 0.948   | 0.948      |
+----------------+-----------------------+---------+------------+
