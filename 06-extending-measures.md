## Adding new Measures {#extending-measures}

In this section we showcase how to implement a custom performance measure.

A good starting point is writing down the loss function independently of mlr3 (we also did this in the [mlr3measures](https://mlr3measures.mlr-org.com) package).
Here, we illustrate writing measure by implementing the root of the mean squared error for regression problems:

```r
root_mse = function(truth, response) {
  mse = mean((truth - response)^2)
  sqrt(mse)
}

root_mse(c(0, 0.5, 1), c(0.5, 0.5, 0.5))
```

```
## [1] 0.4082
```

In the next step, we embed the `root_mse()` function into a new [R6](https://cran.r-project.org/package=R6) class inheriting from base classes [`MeasureRegr`](https://mlr3.mlr-org.com/reference/MeasureRegr.html)/[`Measure`](https://mlr3.mlr-org.com/reference/Measure.html).
For classification measures, use [`MeasureClassif`](https://mlr3.mlr-org.com/reference/MeasureClassif.html).
We keep it simple here and only explain the most important parts of the [`Measure`](https://mlr3.mlr-org.com/reference/Measure.html) class:


```r
MeasureRootMSE = R6::R6Class("MeasureRootMSE",
  inherit = mlr3::MeasureRegr,
  public = list(
    initialize = function() {
      super$initialize(
        # custom id for the measure
        id = "root_mse",

        # additional packages required to calculate this measure
        packages = character(),

        # properties, see below
        properties = character(),

        # required predict type of the learner
        predict_type = "response",

        # feasible range of values
        range = c(0, Inf),

        # minimize during tuning?
        minimize = TRUE
      )
    }
  ),

  private = list(
    # custom scoring function operating on the prediction object
    .score = function(prediction, ...) {
      root_mse = function(truth, response) {
        mse = mean((truth - response)^2)
        sqrt(mse)
      }

      root_mse(prediction$truth, prediction$response)
    }
  )
)
```

This class can be used as template for most performance measures.
If something is missing, you might want to consider having a deeper dive into the following arguments:

* `properties`: If you tag you measure with the property `"requires_task"`, the [`Task`](https://mlr3.mlr-org.com/reference/Task.html) is automatically passed to your `.score()` function (don't forget to add the argument `task` in the signature).
  The same is possible with `"requires_learner"` if you need to operate on the [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) and `"requires_train_set"` if you want to access the set of training indices in the score function.
* `aggregator`: This function (defaulting to `mean()`) controls how multiple performance scores, i.e. from different resampling iterations, are aggregated into a single numeric value if `average` is set to micro averaging.
  This is ignored for macro averaging.
* `predict_sets`: Prediction sets (subset of `("train", "test")`) to operate on.
  Defaults to the "test" set.

Finally, if you want to use your custom measure just like any other measure shipped with [mlr3](https://mlr3.mlr-org.com) and access it via the [`mlr_measures`](https://mlr3.mlr-org.com/reference/mlr_measures.html) dictionary, you can easily add it:

```r
mlr3::mlr_measures$add("root_mse", MeasureRootMSE)
```

Typically it is a good idea to put the measure together with the call to `mlr_measures$add()` in a new R file and just source it in your project.

```r
## source("measure_root_mse.R")
msr("root_mse")
```

```
## <MeasureRootMSE:root_mse>
## * Packages: -
## * Range: [0, Inf]
## * Minimize: TRUE
## * Properties: -
## * Predict type: response
```



