## Error Handling {#error-handling}

To demonstrate how to properly deal with misbehaving learners, [mlr3](https://mlr3.mlr-org.com) ships with the learner [`classif.debug`](https://mlr3.mlr-org.com/reference/mlr_learners_classif.debug.html):


```r
task = tsk("iris")
learner = lrn("classif.debug")
print(learner)
```

```
## <LearnerClassifDebug:classif.debug>
## * Model: -
## * Parameters: list()
## * Packages: -
## * Predict Type: response
## * Feature types: logical, integer, numeric, character, factor, ordered
## * Properties: missings, multiclass, twoclass
```

This learner comes with special hyperparameters that let us control

1. what conditions should be signaled (message, warning, error, segfault) with what probability
2. during which stage the conditions should be signaled (train or predict)
3. the ratio of predictions being `NA` (`predict_missing`)


```r
learner$param_set
```

```
## <ParamSet>
##                   id    class lower upper      levels        default value
##  1:    message_train ParamDbl     0     1                          0      
##  2:  message_predict ParamDbl     0     1                          0      
##  3:    warning_train ParamDbl     0     1                          0      
##  4:  warning_predict ParamDbl     0     1                          0      
##  5:      error_train ParamDbl     0     1                          0      
##  6:    error_predict ParamDbl     0     1                          0      
##  7:   segfault_train ParamDbl     0     1                          0      
##  8: segfault_predict ParamDbl     0     1                          0      
##  9:  predict_missing ParamDbl     0     1                          0      
## 10:       save_tasks ParamLgl    NA    NA  TRUE,FALSE          FALSE      
## 11:                x ParamDbl     0     1             <NoDefault[3]>
```

With the learner's default settings, the learner will do nothing special: The learner learns a random label and creates constant predictions.

```r
task = tsk("iris")
learner$train(task)$predict(task)$confusion
```

```
##             truth
## response     setosa versicolor virginica
##   setosa         50         50        50
##   versicolor      0          0         0
##   virginica       0          0         0
```

We now set a hyperparameter to let the debug learner signal an error during the train step.
By default,[mlr3](https://github.com/mlr-org/mlr3) does not catch conditions such as warnings or errors raised by third-party code like learners:

```r
learner$param_set$values = list(error_train = 1)
learner$train(tsk("iris"))
```

```
## Error in .__LearnerClassifDebug__.train(self = self, private = private, : Error from classif.debug->train()
```
If this would be a regular learner, we could now start debugging with [`traceback()`](https://www.rdocumentation.org/packages/base/topics/traceback) (or create a [MRE](https://stackoverflow.com/help/minimal-reproducible-example) to file a bug report).

However, machine learning algorithms raising errors is not uncommon as algorithms typically cannot process all possible data.
Thus, we need a mechanism to

  1. capture all signaled conditions such as messages, warnings and errors so that we can analyze them post-hoc, and
  2. a statistically sound way to proceed the calculation and be able to aggregate over partial results.

These two mechanisms are explained in the following subsections.

### Encapsulation

With encapsulation, exceptions do not stop the program flow and all output is logged to the learner (instead of printed to the console).
Each [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) has a field `encapsulate` to control how the train or predict steps are executed.
One way to encapsulate the execution is provided by the package [evaluate](https://cran.r-project.org/package=evaluate) (see [`encapsulate()`](https://mlr3misc.mlr-org.com/reference/encapsulate.html) for more details):


```r
task = tsk("iris")
learner = lrn("classif.debug")
learner$param_set$values = list(warning_train = 1, error_train = 1)
learner$encapsulate = c(train = "evaluate", predict = "evaluate")

learner$train(task)
```

After training the learner, one can access the recorded log via the fields `log`, `warnings` and `errors`:


```r
learner$log
```

```
##    stage   class                                 msg
## 1: train warning Warning from classif.debug->train()
## 2: train   error   Error from classif.debug->train()
```

```r
learner$warnings
```

```
## [1] "Warning from classif.debug->train()"
```

```r
learner$errors
```

```
## [1] "Error from classif.debug->train()"
```

Another method for encapsulation is implemented in the [callr](https://cran.r-project.org/package=callr) package.
[callr](https://cran.r-project.org/package=callr) spawns a new R process to execute the respective step, and thus even guards the current session from segfaults.
On the downside, starting new processes comes with a computational overhead.


```r
learner$encapsulate = c(train = "callr", predict = "callr")
learner$param_set$values = list(segfault_train = 1)
learner$train(task = task)
learner$errors
```

```
## [1] "callr process exited with status -11"
```

Without a model, it is not possible to get predictions though:


```r
learner$predict(task)
```

```
## Error: Cannot predict, Learner 'classif.debug' has not been trained yet
```

To handle the missing predictions in a graceful way during [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) or [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html), fallback learners are introduced next.

### Fallback learners

Fallback learners have the purpose to allow scoring results in cases where a [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) is misbehaving in some sense.
Some typical examples include:

* The learner fails to fit a model during training, e.g., if some convergence criterion is not met or the learner ran out of memory.
* The learner fails to predict for some or all observations.
  A typical case is e.g. new factor levels in the test data.

We first handle the most common case that a learner completely breaks while fitting a model or while predicting on new data.
If the learner fails in either of these two steps, we rely on a second learner to generate predictions: the fallback learner.

In the next example, in addition to the debug learner, we attach a simple featureless learner to the debug learner.
So whenever the debug learner fails (which is every time with the given parametrization) and encapsulation in enabled, `mlr3` falls back to the predictions of the featureless learner internally:


```r
task = tsk("iris")
learner = lrn("classif.debug")
learner$param_set$values = list(error_train = 1)
learner$encapsulate = c(train = "evaluate")
learner$fallback = lrn("classif.featureless")
learner$train(task)
learner
```

```
## <LearnerClassifDebug:classif.debug>
## * Model: -
## * Parameters: error_train=1
## * Packages: -
## * Predict Type: response
## * Feature types: logical, integer, numeric, character, factor, ordered
## * Properties: missings, multiclass, twoclass
## * Errors: Error from classif.debug->train()
```
Note that the log contains the captured error (which is also included in the print output), and although we don't have a model, we can still get predictions:

```r
learner$model
```

```
## NULL
```

```r
prediction = learner$predict(task)
prediction$score()
```

```
## classif.ce 
##     0.6667
```

While the fallback learner is of limited use for this stepwise train-predict procedure, it is invaluable for larger benchmark studies where only few resampling iterations are failing.
Here, we need to replace the missing scores with a number in order to aggregate over all resampling iterations.
And imputing a number which is equivalent to guessing labels often seems to be the right amount of penalization.

In the following snippet we compare the previously created debug learner with a simple classification tree.
We re-parametrize the debug learner to fail in roughly 30% of the resampling iterations during the training step:


```r
learner$param_set$values = list(error_train = 0.3)

bmr = benchmark(benchmark_grid(tsk("iris"), list(learner, lrn("classif.rpart")), rsmp("cv")))
aggr = bmr$aggregate(conditions = TRUE)
aggr
```

```
##    nr      resample_result task_id    learner_id resampling_id iters warnings
## 1:  1 <ResampleResult[21]>    iris classif.debug            cv    10        0
## 2:  2 <ResampleResult[21]>    iris classif.rpart            cv    10        0
##    errors classif.ce
## 1:      3    0.67333
## 2:      0    0.07333
```

To further investigate the errors, we can extract the [`ResampleResult`](https://mlr3.mlr-org.com/reference/ResampleResult.html):


```r
rr = aggr[learner_id == "classif.debug"]$resample_result[[1L]]
rr$errors
```

```
##    iteration                               msg
## 1:         2 Error from classif.debug->train()
## 2:         6 Error from classif.debug->train()
## 3:         9 Error from classif.debug->train()
```

A similar yet different problem emerges when a learner predicts only a subset of the observations in the test set (and predicts `NA` for others).
Handling such predictions in a statistically sound way is not straight-forward and a common source for over-optimism when reporting results.
Imagine that our goal is to benchmark two algorithms using a 10-fold cross validation on some binary classification task:

* Algorithm A is a ordinary logistic regression.
* Algorithm B is also a ordinary logistic regression, but with a twist:
  If the logistic regression is rather certain about the predicted label (> 90% probability), it returns the label and a missing value otherwise.

When comparing the performance of these two algorithms, it is obviously not fair to average over all predictions of algorithm A while only average over the "easy-to-predict" observations for algorithm B.
By doing so, algorithm B would easily outperform algorithm A, but you have not factored in that you can not generate predictions for many observations.
On the other hand, it is also not feasible to exclude all observations from the test set of a benchmark study where at least one algorithm failed to predict a label.
Instead, we proceed by imputing all missing predictions with something naive, e.g., by predicting the majority class with a featureless learner.
And as the majority class may depend on the resampling split (or we opt for some other arbitrary baseline learner), it is best to just train a second learner on the same resampling split.

Long story short, if a fallback learner is involved, missing predictions of the base learner will be automatically replaced with predictions from the fallback learner.
This is illustrated in the following example:

```r
task = tsk("iris")
learner = lrn("classif.debug")

# this hyperparameter sets the ratio of missing predictions
learner$param_set$values = list(predict_missing = 0.5)

# without fallback
p = learner$train(task)$predict(task)
table(p$response, useNA = "always")
```

```
## 
##     setosa versicolor  virginica       <NA> 
##         75          0          0         75
```

```r
# with fallback
learner$fallback = lrn("classif.featureless")
p = learner$train(task)$predict(task)
table(p$response, useNA = "always")
```

```
## 
##     setosa versicolor  virginica       <NA> 
##        150          0          0          0
```

Summed up, by combining encapsulation and fallback learners, it is possible to benchmark even quite unreliable or instable learning algorithms in a convenient way.
