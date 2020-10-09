# Pipelines {#pipelines}

[mlr3pipelines](https://mlr3pipelines.mlr-org.com) is a dataflow programming toolkit.
This chapter focuses on the applicant's side of the package.
A more in-depth and technically oriented vignette can be found in the [mlr3pipeline vignette](https://mlr3pipelines.mlr-org.com/articles/introduction.html).

Machine learning workflows can be written as directed “Graphs”/"Pipelines" that represent data flows between preprocessing, model fitting, and ensemble learning units in an expressive and intuitive language.
We will most often use the term "Graph" in this manual but it can interchangeably be used  with "pipeline" or "workflow".

Below you can examine an example for such a graph:

<img src="images/single_pipe.svg" width="98%" style="display: block; margin: auto;" />

Single computational steps can be represented as so-called PipeOps, which can then be connected with directed edges in a Graph.
The scope of [mlr3pipelines](https://mlr3pipelines.mlr-org.com) is still growing.
Currently supported features are:

* Data manipulation and preprocessing operations, e.g. PCA, feature filtering, imputation
* Task subsampling for speed and outcome class imbalance handling
* [mlr3](https://mlr3.mlr-org.com) Learner operations for prediction and stacking
* Ensemble methods and aggregation of predictions

Additionally, we implement several meta operators that can be used to construct powerful pipelines:

* Simultaneous path branching (data going both ways)
* Alternative path branching (data going one specific way, controlled by hyperparameters)

An extensive introduction to creating custom **PipeOps** (PO's) can be found in the [technical introduction](#extending-mlr3pipelines).

Using methods from [mlr3tuning](https://mlr3tuning.mlr-org.com), it is even possible to simultaneously optimize parameters of multiple processing units.

A predecessor to this package is the [mlrCPO](https://cran.r-project.org/package=mlrCPO) package, which works with [mlr](https://cran.r-project.org/package=mlr) 2.x.
Other packages that provide, to varying degree, some preprocessing functionality or machine learning domain specific language, are:

* the [caret](https://cran.r-project.org/package=caret) package and the related [recipes](https://cran.r-project.org/package=recipes)  project
* the [dplyr](https://cran.r-project.org/package=dplyr) package

An example for a Pipeline that can be constructed using [mlr3pipelines](https://mlr3pipelines.mlr-org.com) is depicted below:

<img src="04-pipelines_files/figure-html/04-pipelines-002-1.svg" width="672" style="display: block; margin: auto;" />
