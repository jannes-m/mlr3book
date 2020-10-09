## Nested Resampling {#nested-resampling}

In order to obtain unbiased performance estimates for learners, all parts of the model building (preprocessing and model selection steps) should be included in the resampling, i.e., repeated for every pair of training/test data.
For steps that themselves require resampling like hyperparameter tuning or feature-selection (via the wrapper approach) this results in two nested resampling loops.

<img src="images/nested_resampling.png" width="98%" style="display: block; margin: auto;" />

The graphic above illustrates nested resampling for parameter tuning with 3-fold cross-validation in the outer and 4-fold cross-validation in the inner loop.

In the outer resampling loop, we have three pairs of training/test sets.
On each of these outer training sets parameter tuning is done, thereby executing the inner resampling loop.
This way, we get one set of selected hyperparameters for each outer training set.
Then the learner is fitted on each outer training set using the corresponding selected hyperparameters.
Subsequently, we can evaluate the performance of the learner on the outer test sets.

In [mlr3](https://mlr3.mlr-org.com), you can run nested resampling for free without programming any loops by using the [`mlr3tuning::AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) class.
This works as follows:

1. Generate a wrapped Learner via class [`mlr3tuning::AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) or `mlr3filters::AutoSelect` (not yet implemented).
2. Specify all required settings - see section ["Automating the Tuning"](#autotuner) for help.
3. Call function [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) or [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html) with the created [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html).

You can freely combine different inner and outer resampling strategies.

A common setup is prediction and performance evaluation on a fixed outer test set.
This can be achieved by passing the [`Resampling`](https://mlr3.mlr-org.com/reference/Resampling.html) strategy (`rsmp("holdout")`) as the outer resampling instance to either [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) or [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html).

The inner resampling strategy could be a cross-validation one (`rsmp("cv")`) as the sizes of the outer training sets might differ.
Per default, the inner resample description is instantiated once for every outer training set.

Note that nested resampling is computationally expensive.
For this reason we use relatively small search spaces and a low number of resampling iterations in the examples shown below.
In practice, you normally have to increase both.
As this is computationally intensive you might want to have a look at the section on [Parallelization](#parallelization).

### Execution {#nested-resamp-exec}

To optimize hyperparameters or conduct feature selection in a nested resampling you need to create learners using either:

* the [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) class, or
* the `mlr3filters::AutoSelect` class (not yet implemented)

We use the example from section ["Automating the Tuning"](#autotuner) and pipe the resulting learner into a [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) call.


```r
library("mlr3tuning")
task = tsk("iris")
learner = lrn("classif.rpart")
resampling = rsmp("holdout")
measure = msr("classif.ce")
param_set = paradox::ParamSet$new(
  params = list(paradox::ParamDbl$new("cp", lower = 0.001, upper = 0.1)))
terminator = trm("evals", n_evals = 5)
tuner = tnr("grid_search", resolution = 10)

at = AutoTuner$new(learner, resampling, measure = measure,
  param_set, terminator, tuner = tuner)
```

Now construct the [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) call:


```r
resampling_outer = rsmp("cv", folds = 3)
rr = resample(task = task, learner = at, resampling = resampling_outer)
```

```
## INFO  [06:58:24.009] Starting to optimize 1 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [06:58:24.051] Evaluating 1 configuration(s) 
## INFO  [06:58:24.289] Result of batch 1: 
## INFO  [06:58:24.293]     cp classif.ce                                uhash 
## INFO  [06:58:24.293]  0.089     0.0303 5a67e447-3744-4091-8eb5-05af6d0a8e17 
## INFO  [06:58:24.296] Evaluating 1 configuration(s) 
## INFO  [06:58:24.476] Result of batch 2: 
## INFO  [06:58:24.478]     cp classif.ce                                uhash 
## INFO  [06:58:24.478]  0.012     0.0303 997a72f6-1741-444b-ab40-33ccb6c98aec 
## INFO  [06:58:24.481] Evaluating 1 configuration(s) 
## INFO  [06:58:24.575] Result of batch 3: 
## INFO  [06:58:24.577]     cp classif.ce                                uhash 
## INFO  [06:58:24.577]  0.023     0.0303 c4ecbcfd-576e-4dda-a0f7-fb01e9737be6 
## INFO  [06:58:24.580] Evaluating 1 configuration(s) 
## INFO  [06:58:24.678] Result of batch 4: 
## INFO  [06:58:24.681]     cp classif.ce                                uhash 
## INFO  [06:58:24.681]  0.001     0.0303 edb1169a-a402-4ac2-a01a-783b2fbf7b02 
## INFO  [06:58:24.683] Evaluating 1 configuration(s) 
## INFO  [06:58:24.774] Result of batch 5: 
## INFO  [06:58:24.777]   cp classif.ce                                uhash 
## INFO  [06:58:24.777]  0.1     0.0303 bda87be1-c07d-4433-9a71-b90ec1382b53 
## INFO  [06:58:24.785] Finished optimizing after 5 evaluation(s) 
## INFO  [06:58:24.786] Result: 
## INFO  [06:58:24.788]     cp learner_param_vals  x_domain classif.ce 
## INFO  [06:58:24.788]  0.089          <list[2]> <list[1]>     0.0303 
## INFO  [06:58:24.836] Starting to optimize 1 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [06:58:24.840] Evaluating 1 configuration(s) 
## INFO  [06:58:24.934] Result of batch 1: 
## INFO  [06:58:24.937]     cp classif.ce                                uhash 
## INFO  [06:58:24.937]  0.001    0.09091 df5af727-a18c-432f-9c96-e9056789865d 
## INFO  [06:58:24.939] Evaluating 1 configuration(s) 
## INFO  [06:58:25.033] Result of batch 2: 
## INFO  [06:58:25.035]     cp classif.ce                                uhash 
## INFO  [06:58:25.035]  0.089    0.09091 da5fa219-7a7f-49bf-9bd1-0d5ce98dd0ff 
## INFO  [06:58:25.038] Evaluating 1 configuration(s) 
## INFO  [06:58:25.134] Result of batch 3: 
## INFO  [06:58:25.136]     cp classif.ce                                uhash 
## INFO  [06:58:25.136]  0.034    0.09091 0fb48bd1-c46b-4448-9a11-16e2602f191c 
## INFO  [06:58:25.139] Evaluating 1 configuration(s) 
## INFO  [06:58:25.233] Result of batch 4: 
## INFO  [06:58:25.235]   cp classif.ce                                uhash 
## INFO  [06:58:25.235]  0.1    0.09091 7431cfd7-f0df-482a-a38d-02c28bfcbabf 
## INFO  [06:58:25.238] Evaluating 1 configuration(s) 
## INFO  [06:58:25.332] Result of batch 5: 
## INFO  [06:58:25.335]     cp classif.ce                                uhash 
## INFO  [06:58:25.335]  0.078    0.09091 ca8f4fdf-ec99-4959-a6ff-b81a99a39951 
## INFO  [06:58:25.341] Finished optimizing after 5 evaluation(s) 
## INFO  [06:58:25.342] Result: 
## INFO  [06:58:25.344]     cp learner_param_vals  x_domain classif.ce 
## INFO  [06:58:25.344]  0.001          <list[2]> <list[1]>    0.09091 
## INFO  [06:58:25.392] Starting to optimize 1 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [06:58:25.396] Evaluating 1 configuration(s) 
## INFO  [06:58:25.491] Result of batch 1: 
## INFO  [06:58:25.493]     cp classif.ce                                uhash 
## INFO  [06:58:25.493]  0.034    0.09091 997a1e09-659b-4f9a-b7ff-434af5046846 
## INFO  [06:58:25.496] Evaluating 1 configuration(s) 
## INFO  [06:58:25.591] Result of batch 2: 
## INFO  [06:58:25.593]     cp classif.ce                                uhash 
## INFO  [06:58:25.593]  0.012    0.09091 92c3fb0d-335a-4a76-9e1f-2ab00a71625d 
## INFO  [06:58:25.596] Evaluating 1 configuration(s) 
## INFO  [06:58:25.694] Result of batch 3: 
## INFO  [06:58:25.696]     cp classif.ce                                uhash 
## INFO  [06:58:25.696]  0.001    0.09091 1de1454e-154a-4a3f-8f76-ba08c3b4c9c4 
## INFO  [06:58:25.699] Evaluating 1 configuration(s) 
## INFO  [06:58:25.793] Result of batch 4: 
## INFO  [06:58:25.795]     cp classif.ce                                uhash 
## INFO  [06:58:25.795]  0.067    0.09091 abe4b5a6-6772-4d03-be7c-25a79d6c6a05 
## INFO  [06:58:25.798] Evaluating 1 configuration(s) 
## INFO  [06:58:25.903] Result of batch 5: 
## INFO  [06:58:25.906]     cp classif.ce                                uhash 
## INFO  [06:58:25.906]  0.056    0.09091 2d84412d-a110-4dd5-95f8-146e3ac5522f 
## INFO  [06:58:25.912] Finished optimizing after 5 evaluation(s) 
## INFO  [06:58:25.914] Result: 
## INFO  [06:58:25.916]     cp learner_param_vals  x_domain classif.ce 
## INFO  [06:58:25.916]  0.034          <list[2]> <list[1]>    0.09091
```

### Evaluation {#nested-resamp-eval}

With the created [`ResampleResult`](https://mlr3.mlr-org.com/reference/ResampleResult.html) we can now inspect the executed resampling iterations more closely.
See the section on [Resampling](#resampling) for more detailed information about [`ResampleResult`](https://mlr3.mlr-org.com/reference/ResampleResult.html) objects.

For example, we can query the aggregated performance result:


```r
rr$aggregate()
```

```
## classif.ce 
##       0.06
```

Check for any errors in the folds during execution (if there is not output, warnings or errors recorded, this is an empty `data.table()`:


```r
rr$errors
```

```
## Empty data.table (0 rows and 2 cols): iteration,msg
```

Or take a look at the confusion matrix of the joined predictions:


```r
rr$prediction()$confusion
```

```
##             truth
## response     setosa versicolor virginica
##   setosa         50          0         0
##   versicolor      0         46         5
##   virginica       0          4        45
```
