Sparse Group Lasso Benchmark
============================
|Build Status| |Python 3.6+|

This benchmark is dedicated to the **Sparse Group Lasso**. The optimization problem reads

$$
\\min_{\\beta \\in \\mathbb{R}^p} \\frac{1}{2n} \\lVert y - X\\beta \\rVert^2 + \\lambda(\\tau \\lVert \\beta \\rVert_1  + (1-\\tau)  \\sum_g \\lVert \\beta_{[g]} \\rVert_2)
$$

with

$$
y \\in \\mathbb{R}^n, \\, \\, X \\in \\mathbb{R}^{n \\times p}, \\lambda > 0, 0\\leq\\tau < 1
$$

where $n$ (or ``n_samples``) stands for the number of samples, $p$ (or ``n_features``) stands for the number of features, $\\beta$ are the coefficients of the features, and  $\\beta_{[g]}$ are the coefficients of the $g$-th group.


Install
-------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_group_lasso
   $ cd benchmark_group_lasso
   $ benchopt run .

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run . -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.



.. |Build Template| image:: https://github.com/benchopt/template_benchmark/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/template_benchmark/actions
.. |Build Status| image:: https://github.com/Badr-MOUFAD/benchmark_group_lasso/workflows/Tests/badge.svg
   :target: https://github.com/Badr-MOUFAD/benchmark_group_lasso/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
