## Modeling {#pipe-modeling}



The main purpose of a [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html) is to build combined preprocessing and model fitting pipelines that can be used as [mlr3](https://mlr3.mlr-org.com) [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html).

Conceptually, the process may be summarized as follows:

<img src="images/pipe_action.svg" style="display: block; margin: auto;" />

In the following we chain two preprocessing tasks:

* mutate (creation of a new feature)
* filter (filtering the dataset)

Subsequently one can chain a PO learner to train and predict on the modified dataset.


```r
mutate = mlr_pipeops$get("mutate")
filter = mlr_pipeops$get("filter",
  filter = mlr3filters::FilterVariance$new(),
  param_vals = list(filter.frac = 0.5))

graph = mutate %>>%
  filter %>>%
  mlr_pipeops$get("learner",
    learner = mlr_learners$get("classif.rpart"))
```

Until here we defined the main pipeline stored in [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html).
Now we can train and predict the pipeline:


```r
task = mlr_tasks$get("iris")
graph$train(task)
```

```
## $classif.rpart.output
## NULL
```

```r
graph$predict(task)
```

```
## $classif.rpart.output
## <PredictionClassif> for 150 observations:
##     row_id     truth  response
##          1    setosa    setosa
##          2    setosa    setosa
##          3    setosa    setosa
## ---                           
##        148 virginica virginica
##        149 virginica virginica
##        150 virginica virginica
```

Rather than calling `$train()` and `$predict()` manually, we can put the pipeline [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html) into a [`GraphLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html) object.
A [`GraphLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html) encapsulates the whole pipeline (including the preprocessing steps) and can be put into [`resample()`](https://mlr3.mlr-org.com/reference/resample.html)  or [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html) .
If you are familiar with the old _mlr_ package, this is the equivalent of all the `make*Wrapper()` functions.
The pipeline being encapsulated (here [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html) ) must always produce a [`Prediction`](https://mlr3.mlr-org.com/reference/Prediction.html)  with its `$predict()` call, so it will probably contain at least one [`PipeOpLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_pipeops_learner.html) .


```r
glrn = GraphLearner$new(graph)
```

This learner can be used for model fitting, resampling, benchmarking, and tuning:


```r
cv3 = rsmp("cv", folds = 3)
resample(task, glrn, cv3)
```

```
## <ResampleResult> of 3 iterations
## * Task: iris
## * Learner: mutate.variance.classif.rpart
## * Warnings: 0 in 0 iterations
## * Errors: 0 in 0 iterations
```

### Setting Hyperparameters {#pipe-hyperpars}

Individual POs offer hyperparameters because they contain `$param_set` slots that can be read and written from `$param_set$values` (via the paradox package).
The parameters get passed down to the [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html), and finally to the [`GraphLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html) .
This makes it not only possible to easily change the behavior of a [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)  / [`GraphLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html) and try different settings manually, but also to perform tuning using the [mlr3tuning](https://mlr3tuning.mlr-org.com) package.


```r
glrn$param_set$values$variance.filter.frac = 0.25
cv3 = rsmp("cv", folds = 3)
resample(task, glrn, cv3)
```

```
## <ResampleResult> of 3 iterations
## * Task: iris
## * Learner: mutate.variance.classif.rpart
## * Warnings: 0 in 0 iterations
## * Errors: 0 in 0 iterations
```

### Tuning {#pipe-tuning}

If you are unfamiliar with tuning in [mlr3](https://mlr3.mlr-org.com), we recommend to take a look at the section about [tuning](#tuning) first.
Here we define a [`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html) for the "rpart" learner and the "variance" filter which should be optimized during the tuning process.


```r
library("paradox")
ps = ParamSet$new(list(
  ParamDbl$new("classif.rpart.cp", lower = 0, upper = 0.05),
  ParamDbl$new("variance.filter.frac", lower = 0.25, upper = 1)
))
```

After having defined the `PerformanceEvaluator`, a random search with 10 iterations is created.
For the inner resampling, we are simply using holdout (single split into train/test) to keep the runtimes reasonable.


```r
library("mlr3tuning")
instance = TuningInstanceSingleCrit$new(
  task = task,
  learner = glrn,
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  search_space = ps,
  terminator = trm("evals", n_evals = 20)
)
```


```r
tuner = tnr("random_search")
tuner$optimize(instance)
```

The tuning result can be found in the respective `result` slots.


```r
instance$result_learner_param_vals
instance$result_y
```
