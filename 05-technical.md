# Technical {#technical}

This chapter provides an overview of technical details of the [mlr3](https://mlr3.mlr-org.com) framework.

**Parallelization**

At first, some details about [Parallelization](#parallelization) and the usage of the [future](https://cran.r-project.org/package=future) are given.
Parallelization refers to the process of running multiple jobs simultaneously.
This process is employed to minimize the necessary computing power.
Algorithms consist of both sequential (non-parallelizable) and parallelizable parts.
Therefore, parallelization does not always alter performance in a positive substantial manner.
Summed up, this sub-chapter illustrates how and when to use parallelization in mlr3.

**Database Backends**

The section [Database Backends](#backends) describes how to work with database backends that [mlr3](https://mlr3.mlr-org.com) supports.
Database backends can be helpful for large data processing which does not fit in memory or is stored natively in a database (e.g. SQLite).
Specifically when working with large data sets, or when undertaking numerous tasks simultaneously, it can be advantageous to interface out-of-memory data.
The section provides an illustration of how to implement [Database Backends](#backends) using of NYC flight data.

**Parameters**

In the section [Parameters](#paradox) instructions are given on how to:

* define parameter sets for learners
* undertake parameter sampling
* apply parameter transformations

For illustrative purposes, this sub-chapter uses the [paradox](https://paradox.mlr-org.com) package, the successor of [ParamHelpers](https://cran.r-project.org/package=ParamHelpers).

**Logging and Verbosity**

The sub-chapter on [Logging and Verbosity](#logging) shows how to change the most important settings related to logging.
In [mlr3](https://mlr3.mlr-org.com) we use the [lgr](https://cran.r-project.org/package=lgr) package.

**Transition Guide**

Lastly, we provide a [Transition Guide](#transition) for users of the old [mlr](https://mlr.mlr-org.com) who want to switch to [mlr3](https://mlr3.mlr-org.com).
