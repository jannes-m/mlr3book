## Feature Selection / Filtering {#fs}

Often, data sets include a large number of features.
The technique of extracting a subset of relevant features is called "feature selection".

The objective of feature selection is to fit the sparse dependent of a model on a subset of available data features in the most suitable manner.
Feature selection can enhance the interpretability of the model, speed up the learning process and improve the learner performance.
Different approaches exist to identify the relevant features.
Two different approaches are emphasized in the literature:
one is called [Filtering](#fs-filtering) and the other approach is often referred to as feature subset selection or [wrapper methods](#fs-wrapper).

What are the differences [@chandrashekar2014]?

* **Filtering**: An external algorithm computes a rank of the variables (e.g. based on the correlation to the response).
  Then, features are subsetted by a certain criteria, e.g. an absolute number or a percentage of the number of variables.
  The selected features will then be used to fit a model (with optional hyperparameters selected by tuning).
  This calculation is usually cheaper than “feature subset selection” in terms of computation time.
* **Wrapper Methods**: Here, no ranking of features is done.
  Features are selected by a (random) subset of the data.
  Then, we fit a model and subsequently assess the performance.
  This is done for a lot of feature combinations in a cross-validation (CV) setting and the best combination is reported.
  This method is very computationally intensive as a lot of models are fitted.
  Also, strictly speaking all these models would need to be tuned before the performance is estimated.
  This would require an additional nested level in a CV setting.
  After undertaken all of these steps, the selected subset of features is again fitted (with optional hyperparameters selected by tuning).

There is also a third approach which can be attributed to the "filter" family:
The embedded feature-selection methods of some [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html).
Read more about how to use these in section [embedded feature-selection methods](#fs-embedded).

[Ensemble filters](#fs-ensemble) built upon the idea of stacking single filter methods.
These are not yet implemented.

All functionality that is related to feature selection is implemented via the extension package [mlr3filters](https://mlr3filters.mlr-org.com).

### Filters {#fs-filter}

Filter methods assign an importance value to each feature.
Based on these values the features can be ranked.
Thereafter, we are able to select a feature subset.
There is a list of all implemented filter methods in the [Appendix](#list-filters).

### Calculating filter values {#fs-calc}

Currently, only classification and regression tasks are supported.

The first step it to create a new R object using the class of the desired filter method.
Each object of class `Filter` has a `.$calculate()` method which computes the filter values and ranks them in a descending order.


```r
library("mlr3filters")
filter = FilterJMIM$new()

task = tsk("iris")
filter$calculate(task)

as.data.table(filter)
```

```
##         feature  score
## 1: Sepal.Length 1.0401
## 2:  Petal.Width 0.9894
## 3: Petal.Length 0.9881
## 4:  Sepal.Width 0.8314
```

Some filters support changing specific hyperparameters.
This is similar to setting hyperparameters of a [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) using `.$param_set$values`:


```r
filter_cor = FilterCorrelation$new()
filter_cor$param_set
```

```
## <ParamSet>
##        id    class lower upper
## 1:    use ParamFct    NA    NA
## 2: method ParamFct    NA    NA
##                                                                  levels
## 1: everything,all.obs,complete.obs,na.or.complete,pairwise.complete.obs
## 2:                                             pearson,kendall,spearman
##       default value
## 1: everything      
## 2:    pearson
```

```r
# change parameter 'method'
filter_cor$param_set$values = list(method = "spearman")
filter_cor$param_set
```

```
## <ParamSet>
##        id    class lower upper
## 1:    use ParamFct    NA    NA
## 2: method ParamFct    NA    NA
##                                                                  levels
## 1: everything,all.obs,complete.obs,na.or.complete,pairwise.complete.obs
## 2:                                             pearson,kendall,spearman
##       default    value
## 1: everything         
## 2:    pearson spearman
```

Rather than taking the "long" R6 way to create a filter, there is also a built-in shorthand notation for filter creation:


```r
filter = flt("cmim")
filter
```

```
## <FilterCMIM:cmim>
## Task Types: classif, regr
## Task Properties: -
## Packages: praznik
## Feature types: integer, numeric, factor, ordered
```

### Variable Importance Filters {#fs-var-imp-filters}

All [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) with the property "importance" come with integrated feature selection methods.

You can find a list of all learners with this property in the [Appendix](#fs-filter-embedded-list).

For some learners the desired filter method needs to be set during learner creation.
For example, learner `classif.ranger` (in the package [mlr3learners](https://mlr3learners.mlr-org.com)) comes with multiple integrated methods.
See the help page of [`ranger::ranger`](https://www.rdocumentation.org/packages/ranger/topics/ranger).
To use method "impurity", you need to set the filter method during construction.


```r
library("mlr3learners")
lrn = lrn("classif.ranger", importance = "impurity")
```

Now you can use the [`mlr3filters::FilterImportance`](https://mlr3filters.mlr-org.com/reference/FilterImportance.html) class for algorithm-embedded methods to filter a [`Task`](https://mlr3.mlr-org.com/reference/Task.html).


```r
library("mlr3learners")

task = tsk("iris")
filter = flt("importance", learner = lrn)
filter$calculate(task)
head(as.data.table(filter), 3)
```

```
##         feature  score
## 1:  Petal.Width 46.886
## 2: Petal.Length 41.377
## 3: Sepal.Length  8.839
```

### Ensemble Methods {#fs-ensemble}

Work in progress.

### Wrapper Methods {#fs-wrapper}

Wrapper feature selection is supported via the [mlr3fselect](https://mlr3fselect.mlr-org.com) extension package.
At the heart of [mlr3fselect](https://mlr3fselect.mlr-org.com) are the R6 classes:

* [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html), [`FSelectInstanceMultiCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceMultiCrit.html): These two classes describe the feature selection problem and store the results.
* [`FSelector`](https://mlr3fselect.mlr-org.com/reference/FSelector.html): This class is the base class for implementations of feature selection algorithms.

### The `FSelectInstance` Classes {#fs-wrapper-optimization}

The following sub-section examines the feature selection on the [`Pima`](https://mlr3.mlr-org.com/reference/mlr_tasks_sonar.html) data set which is used to predict whether or not a patient has diabetes. 


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
We use the classification tree from [rpart](https://cran.r-project.org/package=rpart).


```r
learner = lrn("classif.rpart")
```

Next, we need to specify how to evaluate the performance of the feature subsets.
For this, we need to choose a [`resampling strategy`](https://mlr3.mlr-org.com/reference/Resampling.html) and a [`performance measure`](https://mlr3.mlr-org.com/reference/Measure.html).


```r
hout = rsmp("holdout")
measure = msr("classif.ce")
```

Finally, one has to choose the available budget for the feature selection.
This is done by selecting one of the available [`Terminators`](https://bbotk.mlr-org.com/reference/Terminator.html):

* Terminate after a given time ([`TerminatorClockTime`](https://bbotk.mlr-org.com/reference/mlr_terminators_clock_time.html))
* Terminate after a given amount of iterations ([`TerminatorEvals`](https://bbotk.mlr-org.com/reference/mlr_terminators_evals.html))
* Terminate after a specific performance is reached ([`TerminatorPerfReached`](https://bbotk.mlr-org.com/reference/mlr_terminators_perf_reached.html))
* Terminate when feature selection does not improve ([`TerminatorStagnation`](https://bbotk.mlr-org.com/reference/mlr_terminators_stagnation.html))
* A combination of the above in an *ALL* or *ANY* fashion ([`TerminatorCombo`](https://bbotk.mlr-org.com/reference/mlr_terminators_combo.html))

For this short introduction, we specify a budget of 20 evaluations and then put everything together into a [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html):


```r
library("mlr3fselect")

evals20 = trm("evals", n_evals = 20)

instance = FSelectInstanceSingleCrit$new(
  task = task,
  learner = learner,
  resampling = hout,
  measure = measure,
  terminator = evals20
)
instance
```

```
## <FSelectInstanceSingleCrit>
## * State:  Not optimized
## * Objective: <ObjectiveFSelect:classif.rpart_on_pima>
## * Search Space:
## <ParamSet>
##          id    class lower upper      levels        default value
## 1:      age ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 2:  glucose ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 3:  insulin ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 4:     mass ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 5: pedigree ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 6: pregnant ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 7: pressure ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 8:  triceps ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## * Terminator: <TerminatorEvals>
## * Terminated: FALSE
## * Archive:
## <ArchiveFSelect>
## Null data.table (0 rows and 0 cols)
```

To start the feature selection, we still need to select an algorithm which are defined via the [`FSelector`](https://mlr3fselect.mlr-org.com/reference/FSelector.html) class

### The `FSelector` Class

The following algorithms are currently implemented in [mlr3fselect](https://mlr3fselect.mlr-org.com):

* Random Search ([`FSelectorRandomSearch`](https://mlr3fselect.mlr-org.com/reference/FSelectorRandomSearch.html))
* Exhaustive Search ([`FSelectorExhaustiveSearch`](https://mlr3fselect.mlr-org.com/reference/FSelectorExhaustiveSearch.html))
* Sequential Search ([`FSelectorSequential`](https://mlr3fselect.mlr-org.com/reference/FSelectorSequential.html))
* Recursive Feature Elimination ([`FSelectorRFE`](https://mlr3fselect.mlr-org.com/reference/FSelectorRFE.html))
* Design Points ([`FSelectorDesignPoints`](https://mlr3fselect.mlr-org.com/reference/FSelectorDesignPoints.html))

In this example, we will use a simple random search.


```r
fselector = fs("random_search")
```

### Triggering the Tuning {#wrapper-selection-triggering}

To start the feature selection, we simply pass the [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html) to the `$optimize()` method of the initialized [`FSelector`](https://mlr3fselect.mlr-org.com/reference/FSelector.html). The algorithm proceeds as follows

1. The [`FSelector`](https://mlr3fselect.mlr-org.com/reference/FSelector.html) proposes at least one feature subset and may propose multiple subsets to improve parallelization, which can be controlled via the setting `batch_size`).
2. For each feature subset, the given [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) is fitted on the [`Task`](https://mlr3.mlr-org.com/reference/Task.html) using the provided [`Resampling`](https://mlr3.mlr-org.com/reference/Resampling.html).
   All evaluations are stored in the archive of the [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html).
3. The [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) is queried if the budget is exhausted.
   If the budget is not exhausted, restart with 1) until it is.
4. Determine the feature subset with the best observed performance.
5. Store the best feature subset as the result in the instance object.
The best featue subset (`$result_feature_set`) and the corresponding measured performance (`$result_y`) can be accessed from the instance.


```r
fselector$optimize(instance)
```

```
## INFO  [06:58:17.680] Starting to optimize 8 parameter(s) with '<FSelectorRandomSearch>' and '<TerminatorEvals>' 
## INFO  [06:58:17.891] Evaluating 10 configuration(s) 
## INFO  [06:58:18.543] Result of batch 1: 
## INFO  [06:58:18.546]    age glucose insulin  mass pedigree pregnant pressure triceps classif.ce 
## INFO  [06:58:18.546]  FALSE    TRUE    TRUE  TRUE    FALSE     TRUE     TRUE    TRUE     0.2461 
## INFO  [06:58:18.546]  FALSE    TRUE    TRUE FALSE    FALSE    FALSE    FALSE    TRUE     0.2500 
## INFO  [06:58:18.546]  FALSE   FALSE    TRUE  TRUE    FALSE    FALSE    FALSE   FALSE     0.3477 
## INFO  [06:58:18.546]  FALSE    TRUE    TRUE FALSE     TRUE     TRUE     TRUE    TRUE     0.2695 
## INFO  [06:58:18.546]   TRUE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2969 
## INFO  [06:58:18.546]  FALSE   FALSE    TRUE FALSE     TRUE     TRUE    FALSE    TRUE     0.3398 
## INFO  [06:58:18.546]   TRUE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2969 
## INFO  [06:58:18.546]   TRUE    TRUE    TRUE  TRUE    FALSE     TRUE     TRUE    TRUE     0.2891 
## INFO  [06:58:18.546]   TRUE   FALSE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.3203 
## INFO  [06:58:18.546]   TRUE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2969 
## INFO  [06:58:18.546]                                 uhash 
## INFO  [06:58:18.546]  c1e2b282-a9ce-45a4-b5e4-846ab5cc91fd 
## INFO  [06:58:18.546]  542cc598-f3af-4c5a-a119-b6e1326d342a 
## INFO  [06:58:18.546]  d232f004-34e2-4c2a-9601-f69df4419ae6 
## INFO  [06:58:18.546]  e5f4aee3-665c-47a7-8222-2d5cf3092494 
## INFO  [06:58:18.546]  1c0fc2bf-5466-45f1-84b0-cbe8b63edd6d 
## INFO  [06:58:18.546]  3841c5f4-26d2-46b9-ad9e-a9f969674142 
## INFO  [06:58:18.546]  260d7d28-9562-440c-b4df-5c5e07c76880 
## INFO  [06:58:18.546]  307e8d5a-24c0-4086-9148-694701bc30a2 
## INFO  [06:58:18.546]  002ddadc-976b-4cd9-9be3-29cd94c15651 
## INFO  [06:58:18.546]  e74270e2-b20c-48e6-aeeb-813c5a0700ec 
## INFO  [06:58:18.550] Evaluating 10 configuration(s) 
## INFO  [06:58:19.031] Result of batch 2: 
## INFO  [06:58:19.034]    age glucose insulin  mass pedigree pregnant pressure triceps classif.ce 
## INFO  [06:58:19.034]  FALSE    TRUE   FALSE FALSE     TRUE    FALSE    FALSE   FALSE     0.2578 
## INFO  [06:58:19.034]   TRUE   FALSE   FALSE FALSE    FALSE    FALSE    FALSE    TRUE     0.3516 
## INFO  [06:58:19.034]   TRUE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2969 
## INFO  [06:58:19.034]   TRUE    TRUE   FALSE  TRUE    FALSE     TRUE    FALSE    TRUE     0.2461 
## INFO  [06:58:19.034]   TRUE   FALSE    TRUE  TRUE     TRUE    FALSE    FALSE    TRUE     0.3281 
## INFO  [06:58:19.034]   TRUE    TRUE    TRUE FALSE     TRUE     TRUE     TRUE    TRUE     0.3086 
## INFO  [06:58:19.034]  FALSE    TRUE    TRUE  TRUE     TRUE     TRUE    FALSE    TRUE     0.2695 
## INFO  [06:58:19.034]   TRUE    TRUE    TRUE FALSE     TRUE     TRUE     TRUE    TRUE     0.3086 
## INFO  [06:58:19.034]   TRUE   FALSE   FALSE  TRUE    FALSE    FALSE     TRUE   FALSE     0.3242 
## INFO  [06:58:19.034]   TRUE   FALSE    TRUE FALSE    FALSE    FALSE    FALSE   FALSE     0.3633 
## INFO  [06:58:19.034]                                 uhash 
## INFO  [06:58:19.034]  2079eb40-8404-46b7-ae40-bfe12a7cd178 
## INFO  [06:58:19.034]  e55b096d-58d8-49f3-b544-df3236d879b1 
## INFO  [06:58:19.034]  fcf3592d-b87d-4ff9-88c4-a49613712571 
## INFO  [06:58:19.034]  7c2c1c3c-025d-4d8b-999e-b6ce22c9151b 
## INFO  [06:58:19.034]  81b7923c-381c-40df-9e10-996e2d05d840 
## INFO  [06:58:19.034]  712052e0-eb94-42d6-b38b-ffd6046f2bce 
## INFO  [06:58:19.034]  1219ba71-5d15-43bd-b76e-aaae3a990c9c 
## INFO  [06:58:19.034]  6e578921-3cb0-44ed-8944-0d512ba004d6 
## INFO  [06:58:19.034]  e2040ac4-b776-4e0d-a41f-c2eeeeb0e1d9 
## INFO  [06:58:19.034]  74629076-4ac9-4c28-ae4c-19a8ddce6e30 
## INFO  [06:58:19.041] Finished optimizing after 20 evaluation(s) 
## INFO  [06:58:19.042] Result: 
## INFO  [06:58:19.045]    age glucose insulin mass pedigree pregnant pressure triceps 
## INFO  [06:58:19.045]  FALSE    TRUE    TRUE TRUE    FALSE     TRUE     TRUE    TRUE 
## INFO  [06:58:19.045]                                        features  x_domain classif.ce 
## INFO  [06:58:19.045]  glucose,insulin,mass,pregnant,pressure,triceps <list[8]>     0.2461
```

```
##      age glucose insulin mass pedigree pregnant pressure triceps
## 1: FALSE    TRUE    TRUE TRUE    FALSE     TRUE     TRUE    TRUE
##                                          features  x_domain classif.ce
## 1: glucose,insulin,mass,pregnant,pressure,triceps <list[8]>     0.2461
```

```r
instance$result_feature_set
```

```
## [1] "glucose"  "insulin"  "mass"     "pregnant" "pressure" "triceps"
```

```r
instance$result_y
```

```
## classif.ce 
##     0.2461
```
One can investigate all resamplings which were undertaken, as they are stored in the archive of the [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html) and can be accessed through `$data()` method:


```r
instance$archive$data()
```

```
##       age glucose insulin  mass pedigree pregnant pressure triceps classif.ce
##  1: FALSE    TRUE    TRUE  TRUE    FALSE     TRUE     TRUE    TRUE     0.2461
##  2: FALSE    TRUE    TRUE FALSE    FALSE    FALSE    FALSE    TRUE     0.2500
##  3: FALSE   FALSE    TRUE  TRUE    FALSE    FALSE    FALSE   FALSE     0.3477
##  4: FALSE    TRUE    TRUE FALSE     TRUE     TRUE     TRUE    TRUE     0.2695
##  5:  TRUE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2969
##  6: FALSE   FALSE    TRUE FALSE     TRUE     TRUE    FALSE    TRUE     0.3398
##  7:  TRUE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2969
##  8:  TRUE    TRUE    TRUE  TRUE    FALSE     TRUE     TRUE    TRUE     0.2891
##  9:  TRUE   FALSE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.3203
## 10:  TRUE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2969
## 11: FALSE    TRUE   FALSE FALSE     TRUE    FALSE    FALSE   FALSE     0.2578
## 12:  TRUE   FALSE   FALSE FALSE    FALSE    FALSE    FALSE    TRUE     0.3516
## 13:  TRUE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2969
## 14:  TRUE    TRUE   FALSE  TRUE    FALSE     TRUE    FALSE    TRUE     0.2461
## 15:  TRUE   FALSE    TRUE  TRUE     TRUE    FALSE    FALSE    TRUE     0.3281
## 16:  TRUE    TRUE    TRUE FALSE     TRUE     TRUE     TRUE    TRUE     0.3086
## 17: FALSE    TRUE    TRUE  TRUE     TRUE     TRUE    FALSE    TRUE     0.2695
## 18:  TRUE    TRUE    TRUE FALSE     TRUE     TRUE     TRUE    TRUE     0.3086
## 19:  TRUE   FALSE   FALSE  TRUE    FALSE    FALSE     TRUE   FALSE     0.3242
## 20:  TRUE   FALSE    TRUE FALSE    FALSE    FALSE    FALSE   FALSE     0.3633
##                                    uhash  x_domain           timestamp batch_nr
##  1: c1e2b282-a9ce-45a4-b5e4-846ab5cc91fd <list[8]> 2020-10-09 06:58:18        1
##  2: 542cc598-f3af-4c5a-a119-b6e1326d342a <list[8]> 2020-10-09 06:58:18        1
##  3: d232f004-34e2-4c2a-9601-f69df4419ae6 <list[8]> 2020-10-09 06:58:18        1
##  4: e5f4aee3-665c-47a7-8222-2d5cf3092494 <list[8]> 2020-10-09 06:58:18        1
##  5: 1c0fc2bf-5466-45f1-84b0-cbe8b63edd6d <list[8]> 2020-10-09 06:58:18        1
##  6: 3841c5f4-26d2-46b9-ad9e-a9f969674142 <list[8]> 2020-10-09 06:58:18        1
##  7: 260d7d28-9562-440c-b4df-5c5e07c76880 <list[8]> 2020-10-09 06:58:18        1
##  8: 307e8d5a-24c0-4086-9148-694701bc30a2 <list[8]> 2020-10-09 06:58:18        1
##  9: 002ddadc-976b-4cd9-9be3-29cd94c15651 <list[8]> 2020-10-09 06:58:18        1
## 10: e74270e2-b20c-48e6-aeeb-813c5a0700ec <list[8]> 2020-10-09 06:58:18        1
## 11: 2079eb40-8404-46b7-ae40-bfe12a7cd178 <list[8]> 2020-10-09 06:58:19        2
## 12: e55b096d-58d8-49f3-b544-df3236d879b1 <list[8]> 2020-10-09 06:58:19        2
## 13: fcf3592d-b87d-4ff9-88c4-a49613712571 <list[8]> 2020-10-09 06:58:19        2
## 14: 7c2c1c3c-025d-4d8b-999e-b6ce22c9151b <list[8]> 2020-10-09 06:58:19        2
## 15: 81b7923c-381c-40df-9e10-996e2d05d840 <list[8]> 2020-10-09 06:58:19        2
## 16: 712052e0-eb94-42d6-b38b-ffd6046f2bce <list[8]> 2020-10-09 06:58:19        2
## 17: 1219ba71-5d15-43bd-b76e-aaae3a990c9c <list[8]> 2020-10-09 06:58:19        2
## 18: 6e578921-3cb0-44ed-8944-0d512ba004d6 <list[8]> 2020-10-09 06:58:19        2
## 19: e2040ac4-b776-4e0d-a41f-c2eeeeb0e1d9 <list[8]> 2020-10-09 06:58:19        2
## 20: 74629076-4ac9-4c28-ae4c-19a8ddce6e30 <list[8]> 2020-10-09 06:58:19        2
```

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

The `uhash` column links the resampling iterations to the evaluated feature subsets stored in `instance$archive$data()`. This allows e.g. to score the included [`ResampleResult`](https://mlr3.mlr-org.com/reference/ResampleResult.html)s on a different measure.

Now the optimized feature subset can be used to subset the task and fit the model on all observations.


```r
task$select(instance$result_feature_set)
learner$train(task)
```

The trained model can now be used to make a prediction on external data.
Note that predicting on observations present in the `task`,  should be avoided.
The model has seen these observations already during feature selection and therefore results would be statistically biased.
Hence, the resulting performance measure would be over-optimistic.
Instead, to get statistically unbiased performance estimates for the current task, [nested resampling](#nested-resampling) is required.

### Automating the Feature Selection {#autofselect}

The [`AutoFSelect`](https://mlr3fselect.mlr-org.com/reference/AutoFSelect.html) wraps a learner and augments it with an automatic feature selection for a given task.
Because the [`AutoFSelect`](https://mlr3fselect.mlr-org.com/reference/AutoFSelect.html) itself inherits from the [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) base class, it can be used like any other learner.
Analogously to the previous subsection, a new classification tree learner is created.
This classification tree learner automatically starts a feature selection on the given task using an inner resampling (holdout).
We create a terminator which allows 10 evaluations, and and uses a simple random search as feature selection algorithm:


```r
library("paradox")
library("mlr3fselect")

learner = lrn("classif.rpart")
terminator = trm("evals", n_evals = 10)
fselector = fs("random_search")

at = AutoFSelect$new(
  learner = learner,
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  terminator = terminator,
  fselector = fselector
)
at
```

```
## <AutoFSelect:classif.rpart.fselector>
## * Model: -
## * Parameters: xval=0
## * Packages: rpart
## * Predict Type: response
## * Feature types: logical, integer, numeric, factor, ordered
## * Properties: importance, missings, multiclass, selected_features,
##   twoclass, weights
```

We can now use the learner like any other learner, calling the `$train()` and `$predict()` method.
This time however, we pass it to [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html) to compare the optimized feature subset to the complete feature set.
This way, the [`AutoFSelect`](https://mlr3fselect.mlr-org.com/reference/AutoFSelect.html) will do its resampling for feature selection on the training set of the respective split of the outer resampling.
The learner then undertakes predictions using the test set of the outer resampling.
This yields unbiased performance measures, as the observations in the test set have not been used during feature selection or fitting of the respective learner.
This is called [nested resampling](#nested-resampling).

To compare the optimized feature subset with the complete feature set, we can use [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html):


```r
grid = benchmark_grid(
  task = tsk("pima"),
  learner = list(at, lrn("classif.rpart")),
  resampling = rsmp("cv", folds = 3)
)

# avoid console output from mlrfselect
logger = lgr::get_logger("bbotk")
logger$set_threshold("warn")

bmr = benchmark(grid, store_models = TRUE)
bmr$aggregate(msrs(c("classif.ce", "time_train")))
```

```
##    nr      resample_result task_id              learner_id resampling_id iters
## 1:  1 <ResampleResult[21]>    pima classif.rpart.fselector            cv     3
## 2:  2 <ResampleResult[21]>    pima           classif.rpart            cv     3
##    classif.ce time_train
## 1:     0.2721          0
## 2:     0.2448          0
```

Note that we do not expect any significant differences since we only evaluated a small fraction of the possible feature subsets.
