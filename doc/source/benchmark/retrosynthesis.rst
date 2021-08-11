Retrosynthesis
==============

.. include:: ../bibliography.rst

This page contains benchmarks of retrosynthesis models.

We evaluate models on `USPTO50k`_ dataset. The dataset is split into
train/validation/test sets with ratios of 80%:10%:10%. We consider two settings
where the reaction class is given or unknown. For each setting, we report the top-k
accuracy of the retrosynthesis prediction.

Given Reaction Class
--------------------

USPTO50k
^^^^^^^^

+----------------+-------+-------+-------+--------+
| `USPTO50k`_    | Top-1 | Top-3 | Top-5 | Top-10 |
+================+=======+=======+=======+========+
| `G2Gs`_        | 0.639 | 0.852 | 0.904 | 0.938  |
+----------------+-------+-------+-------+--------+

Unknown Reaction Class
----------------------

USPTO50k
^^^^^^^^

+----------------+-------+-------+-------+--------+
| `USPTO50k`_    | Top-1 | Top-3 | Top-5 | Top-10 |
+================+=======+=======+=======+========+
| `G2Gs`_        | 0.438 | 0.677 | 0.748 | 0.822  |
+----------------+-------+-------+-------+--------+