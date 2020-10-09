## Hyperparameter Tuning {#tuning}

Hyperparameters are second-order parameters of machine learning models that, while often not explicitly optimized during the model estimation process, can have an important impact on the outcome and predictive performance of a model.
Typically, hyperparameters are fixed before training a model.
However, because the output of a model can be sensitive to the specification of hyperparameters, it is often recommended to make an informed decision about which hyperparameter settings may yield better model performance.
In many cases, hyperparameter settings may be chosen _a priori_, but it can be advantageous to try different settings before fitting your model on the training data.
This process is often called model 'tuning'.

Hyperparameter tuning is supported via the [mlr3tuning](https://mlr3tuning.mlr-org.com) extension package.
Below you can find an illustration of the process:

<img src="images/tuning_process.svg" style="display: block; margin: auto;" />

At the heart of [mlr3tuning](https://mlr3tuning.mlr-org.com) are the R6 classes:

* [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html), [`TuningInstanceMultiCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceMultiCrit.html): These two classes describe the tuning problem and store the results.
* [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html): This class is the base class for implementations of tuning algorithms.

### The `TuningInstance` Classes {#tuning-optimization}

The following sub-section examines the optimization of a simple classification tree on the [`Pima Indian Diabetes`](https://mlr3.mlr-org.com/reference/mlr_tasks_pima.html) data set.


```r
task = tsk("pima")
print(task)
```

```
## <TaskClassif:pima> (768 x 9)
## * Target: diabetes
## * Properties: twoclass
## * Features (8):
##   - dbl (8): age, glucose, insulin, mass, pedigree, pregnant, pressure,
##     triceps
```

We use the classification tree from [rpart](https://cran.r-project.org/package=rpart) and choose a subset of the hyperparameters we want to tune.
This is often referred to as the "tuning space".


```r
learner = lrn("classif.rpart")
learner$param_set
```

```
## <ParamSet>
##                 id    class lower upper      levels        default value
##  1:       minsplit ParamInt     1   Inf                         20      
##  2:      minbucket ParamInt     1   Inf             <NoDefault[3]>      
##  3:             cp ParamDbl     0     1                       0.01      
##  4:     maxcompete ParamInt     0   Inf                          4      
##  5:   maxsurrogate ParamInt     0   Inf                          5      
##  6:       maxdepth ParamInt     1    30                         30      
##  7:   usesurrogate ParamInt     0     2                          2      
##  8: surrogatestyle ParamInt     0     1                          0      
##  9:           xval ParamInt     0   Inf                         10     0
## 10:     keep_model ParamLgl    NA    NA  TRUE,FALSE          FALSE
```

Here, we opt to tune two parameters:

* The complexity `cp`
* The termination criterion `minsplit`

The tuning space needs to be bounded, therefore one has to set lower and upper bounds:


```r
library("paradox")
tune_ps = ParamSet$new(list(
  ParamDbl$new("cp", lower = 0.001, upper = 0.1),
  ParamInt$new("minsplit", lower = 1, upper = 10)
))
tune_ps
```

```
## <ParamSet>
##          id    class lower upper levels        default value
## 1:       cp ParamDbl 0.001   0.1        <NoDefault[3]>      
## 2: minsplit ParamInt 1.000  10.0        <NoDefault[3]>
```

Next, we need to specify how to evaluate the performance.
For this, we need to choose a [`resampling strategy`](https://mlr3.mlr-org.com/reference/Resampling.html) and a [`performance measure`](https://mlr3.mlr-org.com/reference/Measure.html).


```r
hout = rsmp("holdout")
measure = msr("classif.ce")
```

Finally, one has to select the budget available, to solve this tuning instance.
This is done by selecting one of the available [`Terminators`](https://bbotk.mlr-org.com/reference/Terminator.html):

* Terminate after a given time ([`TerminatorClockTime`](https://bbotk.mlr-org.com/reference/mlr_terminators_clock_time.html))
* Terminate after a given amount of iterations ([`TerminatorEvals`](https://bbotk.mlr-org.com/reference/mlr_terminators_evals.html))
* Terminate after a specific performance is reached ([`TerminatorPerfReached`](https://bbotk.mlr-org.com/reference/mlr_terminators_perf_reached.html))
* Terminate when tuning does not improve ([`TerminatorStagnation`](https://bbotk.mlr-org.com/reference/mlr_terminators_stagnation.html))
* A combination of the above in an *ALL* or *ANY* fashion ([`TerminatorCombo`](https://bbotk.mlr-org.com/reference/mlr_terminators_combo.html))

For this short introduction, we specify a budget of 20 evaluations and then put everything together into a [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html):


```r
library("mlr3tuning")

evals20 = trm("evals", n_evals = 20)

instance = TuningInstanceSingleCrit$new(
  task = task,
  learner = learner,
  resampling = hout,
  measure = measure,
  search_space = tune_ps,
  terminator = evals20
)
instance
```

```
## <TuningInstanceSingleCrit>
## * State:  Not optimized
## * Objective: <ObjectiveTuning:classif.rpart_on_pima>
## * Search Space:
## <ParamSet>
##          id    class lower upper levels        default value
## 1:       cp ParamDbl 0.001   0.1        <NoDefault[3]>      
## 2: minsplit ParamInt 1.000  10.0        <NoDefault[3]>      
## * Terminator: <TerminatorEvals>
## * Terminated: FALSE
## * Archive:
## <ArchiveTuning>
## Null data.table (0 rows and 0 cols)
```

To start the tuning, we still need to select how the optimization should take place.
In other words, we need to choose the **optimization algorithm** via the [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html) class.

### The `Tuner` Class

The following algorithms are currently implemented in [mlr3tuning](https://mlr3tuning.mlr-org.com):

* Grid Search ([`TunerGridSearch`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_grid_search.html))
* Random Search ([`TunerRandomSearch`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_random_search.html)) [@bergstra2012]
* Generalized Simulated Annealing ([`TunerGenSA`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_gensa.html))
* Non-Linear Optimization ([`TunerNLoptr`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_nloptr.html))

In this example, we will use a simple grid search with a grid resolution of 5.


```r
tuner = tnr("grid_search", resolution = 5)
```

Since we have only numeric parameters, [`TunerGridSearch`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_grid_search.html) will create an equidistant grid between the respective upper and lower bounds.
As we have two hyperparameters with a resolution of 5, the two-dimensional grid consists of $5^2 = 25$ configurations.
Each configuration serves as a hyperparameter setting for the previously defined [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) and triggers a 3-fold cross validation on the task.
All configurations will be examined by the tuner (in a random order), until either all configurations are evaluated or the [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) signals that the budget is exhausted.

### Triggering the Tuning {#tuning-triggering}

To start the tuning, we simply pass the [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html) to the `$optimize()` method of the initialized [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html).
The tuner proceeds as follows:

1. The [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html) proposes at least one hyperparameter configuration (the [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html) may propose multiple points to improve parallelization, which can be controlled via the setting `batch_size`).
2. For each configuration, the given [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) is fitted on the [`Task`](https://mlr3.mlr-org.com/reference/Task.html) using the provided [`Resampling`](https://mlr3.mlr-org.com/reference/Resampling.html).
   All evaluations are stored in the archive of the [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html).
3. The [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) is queried if the budget is exhausted.
   If the budget is not exhausted, restart with 1) until it is.
4. Determine the configuration with the best observed performance.
5. Store the best configurations as result in the instance object.
   The best hyperparameter settings (`$result_learner_param_vals`) and the corresponding measured performance (`$result_y`) can be accessed from the instance.


```r
tuner$optimize(instance)
```

```
## INFO  [06:58:06.755] Starting to optimize 2 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [06:58:06.801] Evaluating 1 configuration(s) 
## INFO  [06:58:07.060] Result of batch 1: 
## INFO  [06:58:07.063]       cp minsplit classif.ce                                uhash 
## INFO  [06:58:07.063]  0.02575        3     0.2422 76902b89-222f-47f3-b7e4-a9f57fe3d74c 
## INFO  [06:58:07.066] Evaluating 1 configuration(s) 
## INFO  [06:58:07.325] Result of batch 2: 
## INFO  [06:58:07.329]     cp minsplit classif.ce                                uhash 
## INFO  [06:58:07.329]  0.001        1      0.332 9d8c0560-8fb6-433f-8a10-a40ad43a24c3 
## INFO  [06:58:07.331] Evaluating 1 configuration(s) 
## INFO  [06:58:07.454] Result of batch 3: 
## INFO  [06:58:07.456]       cp minsplit classif.ce                                uhash 
## INFO  [06:58:07.456]  0.07525       10     0.2969 5d6ac6a6-a44b-47aa-99b0-5aa5353fb760 
## INFO  [06:58:07.458] Evaluating 1 configuration(s) 
## INFO  [06:58:07.588] Result of batch 4: 
## INFO  [06:58:07.591]       cp minsplit classif.ce                                uhash 
## INFO  [06:58:07.591]  0.07525        5     0.2969 dc69a2db-ef1b-4bfe-bd4a-0d16b4370c47 
## INFO  [06:58:07.593] Evaluating 1 configuration(s) 
## INFO  [06:58:07.706] Result of batch 5: 
## INFO  [06:58:07.708]       cp minsplit classif.ce                                uhash 
## INFO  [06:58:07.708]  0.07525        3     0.2969 d7e74086-e674-4a1c-a134-f8de4eb0aeb2 
## INFO  [06:58:07.711] Evaluating 1 configuration(s) 
## INFO  [06:58:07.827] Result of batch 6: 
## INFO  [06:58:07.830]       cp minsplit classif.ce                                uhash 
## INFO  [06:58:07.830]  0.02575        1     0.2422 3aeca1c1-22e3-456f-ae34-d84f34d105cb 
## INFO  [06:58:07.832] Evaluating 1 configuration(s) 
## INFO  [06:58:07.940] Result of batch 7: 
## INFO  [06:58:07.943]      cp minsplit classif.ce                                uhash 
## INFO  [06:58:07.943]  0.0505        3     0.2969 01620da4-50f3-45d7-ab7f-c138b86c111e 
## INFO  [06:58:07.945] Evaluating 1 configuration(s) 
## INFO  [06:58:08.072] Result of batch 8: 
## INFO  [06:58:08.080]       cp minsplit classif.ce                                uhash 
## INFO  [06:58:08.080]  0.07525        1     0.2969 a67db53b-d7a2-45fa-9c59-e5f15d275e32 
## INFO  [06:58:08.083] Evaluating 1 configuration(s) 
## INFO  [06:58:08.204] Result of batch 9: 
## INFO  [06:58:08.206]       cp minsplit classif.ce                                uhash 
## INFO  [06:58:08.206]  0.02575        5     0.2422 86c1041f-2dec-4ac5-9cae-83affbc29b6f 
## INFO  [06:58:08.209] Evaluating 1 configuration(s) 
## INFO  [06:58:08.326] Result of batch 10: 
## INFO  [06:58:08.329]   cp minsplit classif.ce                                uhash 
## INFO  [06:58:08.329]  0.1       10     0.2969 f65cf5b8-70bc-4b14-b688-8566d07281e3 
## INFO  [06:58:08.331] Evaluating 1 configuration(s) 
## INFO  [06:58:08.446] Result of batch 11: 
## INFO  [06:58:08.448]      cp minsplit classif.ce                                uhash 
## INFO  [06:58:08.448]  0.0505        5     0.2969 5e589a51-2c20-4462-8cb2-f735a00ac523 
## INFO  [06:58:08.451] Evaluating 1 configuration(s) 
## INFO  [06:58:08.564] Result of batch 12: 
## INFO  [06:58:08.567]       cp minsplit classif.ce                                uhash 
## INFO  [06:58:08.567]  0.02575       10     0.2422 563a062a-d320-436c-8d3b-a898932388a8 
## INFO  [06:58:08.570] Evaluating 1 configuration(s) 
## INFO  [06:58:08.676] Result of batch 13: 
## INFO  [06:58:08.679]      cp minsplit classif.ce                                uhash 
## INFO  [06:58:08.679]  0.0505       10     0.2969 abfe0489-36df-44bf-950b-46035c183ab9 
## INFO  [06:58:08.682] Evaluating 1 configuration(s) 
## INFO  [06:58:08.784] Result of batch 14: 
## INFO  [06:58:08.786]       cp minsplit classif.ce                                uhash 
## INFO  [06:58:08.786]  0.02575        8     0.2422 cf90a484-70e6-414e-95a4-3b3a9ffae8d9 
## INFO  [06:58:08.789] Evaluating 1 configuration(s) 
## INFO  [06:58:08.911] Result of batch 15: 
## INFO  [06:58:08.913]      cp minsplit classif.ce                                uhash 
## INFO  [06:58:08.913]  0.0505        8     0.2969 8c216bdc-11e1-4f68-a4d7-f6eeafa8655d 
## INFO  [06:58:08.917] Evaluating 1 configuration(s) 
## INFO  [06:58:09.016] Result of batch 16: 
## INFO  [06:58:09.019]     cp minsplit classif.ce                                uhash 
## INFO  [06:58:09.019]  0.001        3     0.3281 1ec1fe42-7831-4124-be40-44dbdcdaba70 
## INFO  [06:58:09.021] Evaluating 1 configuration(s) 
## INFO  [06:58:09.133] Result of batch 17: 
## INFO  [06:58:09.135]   cp minsplit classif.ce                                uhash 
## INFO  [06:58:09.135]  0.1        1     0.2969 b015d4be-0026-4183-945f-002623765b90 
## INFO  [06:58:09.138] Evaluating 1 configuration(s) 
## INFO  [06:58:09.251] Result of batch 18: 
## INFO  [06:58:09.253]     cp minsplit classif.ce                                uhash 
## INFO  [06:58:09.253]  0.001       10     0.3008 5d20995b-11a3-4ad9-b434-f757982f3c21 
## INFO  [06:58:09.255] Evaluating 1 configuration(s) 
## INFO  [06:58:09.376] Result of batch 19: 
## INFO  [06:58:09.378]     cp minsplit classif.ce                                uhash 
## INFO  [06:58:09.378]  0.001        5     0.2891 200ca768-45c5-4350-ab2e-3f3bbcf337c4 
## INFO  [06:58:09.381] Evaluating 1 configuration(s) 
## INFO  [06:58:09.488] Result of batch 20: 
## INFO  [06:58:09.491]   cp minsplit classif.ce                                uhash 
## INFO  [06:58:09.491]  0.1        8     0.2969 3235ef00-8ee6-4a4f-b5ca-871363f55328 
## INFO  [06:58:09.498] Finished optimizing after 20 evaluation(s) 
## INFO  [06:58:09.500] Result: 
## INFO  [06:58:09.502]       cp minsplit learner_param_vals  x_domain classif.ce 
## INFO  [06:58:09.502]  0.02575        3          <list[3]> <list[2]>     0.2422
```

```
##         cp minsplit learner_param_vals  x_domain classif.ce
## 1: 0.02575        3          <list[3]> <list[2]>     0.2422
```

```r
instance$result_learner_param_vals
```

```
## $xval
## [1] 0
## 
## $cp
## [1] 0.02575
## 
## $minsplit
## [1] 3
```

```r
instance$result_y
```

```
## classif.ce 
##     0.2422
```

One can investigate all resamplings which were undertaken, as they are stored in the archive of the [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html) and can be accessed through `$data()` method:


```r
instance$archive$data()
```

```
##          cp minsplit classif.ce                                uhash  x_domain
##  1: 0.02575        3     0.2422 76902b89-222f-47f3-b7e4-a9f57fe3d74c <list[2]>
##  2: 0.00100        1     0.3320 9d8c0560-8fb6-433f-8a10-a40ad43a24c3 <list[2]>
##  3: 0.07525       10     0.2969 5d6ac6a6-a44b-47aa-99b0-5aa5353fb760 <list[2]>
##  4: 0.07525        5     0.2969 dc69a2db-ef1b-4bfe-bd4a-0d16b4370c47 <list[2]>
##  5: 0.07525        3     0.2969 d7e74086-e674-4a1c-a134-f8de4eb0aeb2 <list[2]>
##  6: 0.02575        1     0.2422 3aeca1c1-22e3-456f-ae34-d84f34d105cb <list[2]>
##  7: 0.05050        3     0.2969 01620da4-50f3-45d7-ab7f-c138b86c111e <list[2]>
##  8: 0.07525        1     0.2969 a67db53b-d7a2-45fa-9c59-e5f15d275e32 <list[2]>
##  9: 0.02575        5     0.2422 86c1041f-2dec-4ac5-9cae-83affbc29b6f <list[2]>
## 10: 0.10000       10     0.2969 f65cf5b8-70bc-4b14-b688-8566d07281e3 <list[2]>
## 11: 0.05050        5     0.2969 5e589a51-2c20-4462-8cb2-f735a00ac523 <list[2]>
## 12: 0.02575       10     0.2422 563a062a-d320-436c-8d3b-a898932388a8 <list[2]>
## 13: 0.05050       10     0.2969 abfe0489-36df-44bf-950b-46035c183ab9 <list[2]>
## 14: 0.02575        8     0.2422 cf90a484-70e6-414e-95a4-3b3a9ffae8d9 <list[2]>
## 15: 0.05050        8     0.2969 8c216bdc-11e1-4f68-a4d7-f6eeafa8655d <list[2]>
## 16: 0.00100        3     0.3281 1ec1fe42-7831-4124-be40-44dbdcdaba70 <list[2]>
## 17: 0.10000        1     0.2969 b015d4be-0026-4183-945f-002623765b90 <list[2]>
## 18: 0.00100       10     0.3008 5d20995b-11a3-4ad9-b434-f757982f3c21 <list[2]>
## 19: 0.00100        5     0.2891 200ca768-45c5-4350-ab2e-3f3bbcf337c4 <list[2]>
## 20: 0.10000        8     0.2969 3235ef00-8ee6-4a4f-b5ca-871363f55328 <list[2]>
##               timestamp batch_nr
##  1: 2020-10-09 06:58:07        1
##  2: 2020-10-09 06:58:07        2
##  3: 2020-10-09 06:58:07        3
##  4: 2020-10-09 06:58:07        4
##  5: 2020-10-09 06:58:07        5
##  6: 2020-10-09 06:58:07        6
##  7: 2020-10-09 06:58:07        7
##  8: 2020-10-09 06:58:08        8
##  9: 2020-10-09 06:58:08        9
## 10: 2020-10-09 06:58:08       10
## 11: 2020-10-09 06:58:08       11
## 12: 2020-10-09 06:58:08       12
## 13: 2020-10-09 06:58:08       13
## 14: 2020-10-09 06:58:08       14
## 15: 2020-10-09 06:58:08       15
## 16: 2020-10-09 06:58:09       16
## 17: 2020-10-09 06:58:09       17
## 18: 2020-10-09 06:58:09       18
## 19: 2020-10-09 06:58:09       19
## 20: 2020-10-09 06:58:09       20
```

In sum, the grid search evaluated 20/25 different configurations of the grid in a random order before the [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) stopped the tuning.

The associated resampling iterations can be accessed in the [`BenchmarkResult`](https://mlr3.mlr-org.com/reference/BenchmarkResult.html):


```r
instance$archive$benchmark_result$data
```

```
## <ResultData>
##   Public:
##     as_data_table: function (view = NULL, reassemble_learners = TRUE, convert_predictions = TRUE, 
##     clone: function (deep = FALSE) 
##     combine: function (rdata) 
##     data: list
##     initialize: function (data = NULL) 
##     iterations: function (view = NULL) 
##     learners: function (view = NULL, states = TRUE, reassemble = TRUE) 
##     logs: function (view = NULL, condition) 
##     prediction: function (view = NULL, predict_sets = "test") 
##     predictions: function (view = NULL, predict_sets = "test") 
##     resamplings: function (view = NULL) 
##     sweep: function () 
##     task_type: active binding
##     tasks: function (view = NULL, reassemble = TRUE) 
##     uhashes: function (view = NULL) 
##   Private:
##     deep_clone: function (name, value) 
##     get_view_index: function (view)
```
The `uhash` column links the resampling iterations to the evaluated configurations stored in `instance$archive$data()`. This allows e.g. to score the included [`ResampleResult`](https://mlr3.mlr-org.com/reference/ResampleResult.html)s on a different measure.

Now the optimized hyperparameters can take the previously created [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html), set the returned hyperparameters and train it on the full dataset.


```r
learner$param_set$values = instance$result_learner_param_vals
learner$train(task)
```

The trained model can now be used to make a prediction on external data.
Note that predicting on observations present in the `task`,  should be avoided.
The model has seen these observations already during tuning and therefore results would be statistically biased.
Hence, the resulting performance measure would be over-optimistic.
Instead, to get statistically unbiased performance estimates for the current task, [nested resampling](#nested-resamling) is required.

### Automating the Tuning {#autotuner}

The [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) wraps a learner and augments it with an automatic tuning for a given set of hyperparameters.
Because the [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) itself inherits from the [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) base class, it can be used like any other learner.
Analogously to the previous subsection, a new classification tree learner is created.
This classification tree learner automatically tunes the parameters `cp` and `minsplit` using an inner resampling (holdout).
We create a terminator which allows 10 evaluations, and use a simple random search as tuning algorithm:


```r
library("paradox")
library("mlr3tuning")

learner = lrn("classif.rpart")
tune_ps = ParamSet$new(list(
  ParamDbl$new("cp", lower = 0.001, upper = 0.1),
  ParamInt$new("minsplit", lower = 1, upper = 10)
))
terminator = trm("evals", n_evals = 10)
tuner = tnr("random_search")

at = AutoTuner$new(
  learner = learner,
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  search_space = tune_ps,
  terminator = terminator,
  tuner = tuner
)
at
```

```
## <AutoTuner:classif.rpart.tuned>
## * Model: -
## * Parameters: xval=0
## * Packages: rpart
## * Predict Type: response
## * Feature types: logical, integer, numeric, factor, ordered
## * Properties: importance, missings, multiclass, selected_features,
##   twoclass, weights
```

We can now use the learner like any other learner, calling the `$train()` and `$predict()` method.
This time however, we pass it to [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html) to compare the tuner to a classification tree without tuning.
This way, the [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) will do its resampling for tuning on the training set of the respective split of the outer resampling.
The learner then undertakes predictions using the test set of the outer resampling.
This yields unbiased performance measures, as the observations in the test set have not been used during tuning or fitting of the respective learner.
This is called [nested resampling](#nested-resampling).

To compare the tuned learner with the learner that uses default values, we can use [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html):


```r
grid = benchmark_grid(
  task = tsk("pima"),
  learner = list(at, lrn("classif.rpart")),
  resampling = rsmp("cv", folds = 3)
)

# avoid console output from mlr3tuning
logger = lgr::get_logger("bbotk")
logger$set_threshold("warn")

bmr = benchmark(grid)
bmr$aggregate(msrs(c("classif.ce", "time_train")))
```

```
##    nr      resample_result task_id          learner_id resampling_id iters
## 1:  1 <ResampleResult[21]>    pima classif.rpart.tuned            cv     3
## 2:  2 <ResampleResult[21]>    pima       classif.rpart            cv     3
##    classif.ce time_train
## 1:     0.2539          0
## 2:     0.2487          0
```

Note that we do not expect any differences compared to the non-tuned approach for multiple reasons:

* the task is too easy
* the task is rather small, and thus prone to overfitting
* the tuning budget (10 evaluations) is small
* [rpart](https://cran.r-project.org/package=rpart) does not benefit that much from tuning

