## Cost-Sensitive Classification {#cost-sens}

In regular classification the aim is to minimize the misclassification rate and thus all types of misclassification errors are deemed equally severe.
A more general setting is cost-sensitive classification.
Cost sensitive classification does not assume that the costs caused by different kinds of errors are equal.
The objective of cost sensitive classification is to minimize the expected costs.

Imagine you are an analyst for a big credit institution.
Let's also assume that a correct decision of the bank would result in 35% of the profit at the end of a specific period.
A correct decision means that the bank predicts that a customer will pay their bills (hence would obtain a loan), and the customer indeed has good credit.
On the other hand, a wrong decision means that the bank predicts that the customer's credit is in good standing, but the opposite is true.
This would result in a loss of 100% of the given loan.

|                           | Good Customer (truth)       | Bad Customer (truth)       |
| :-----------------------: | :-------------------------: | :------------------------: |
| Good Customer (predicted) | + 0.35                      | - 1.0                      |
| Bad Customer (predicted)  | 0                           | 0                          |


Expressed as costs (instead of profit), we can write down the cost-matrix as follows:


```r
costs = matrix(c(-0.35, 0, 1, 0), nrow = 2)
dimnames(costs) = list(response = c("good", "bad"), truth = c("good", "bad"))
print(costs)
```

```
##         truth
## response  good bad
##     good -0.35   1
##     bad   0.00   0
```
An exemplary data set for such a problem is the [`German Credit`](https://mlr3.mlr-org.com/reference/mlr_tasks_german_credit.html) task:


```r
library("mlr3")
task = tsk("german_credit")
table(task$truth())
```

```
## 
## good  bad 
##  700  300
```

The data has 70% customers who are able to pay back their credit, and 30% bad customers who default on the debt.
A manager, who doesn't have any model, could decide to give either everybody a credit or to give nobody a credit.
The resulting costs for the German credit data are:


```r
# nobody:
(700 * costs[2, 1] + 300 * costs[2, 2]) / 1000
```

```
## [1] 0
```

```r
# everybody
(700 * costs[1, 1] + 300 * costs[1, 2]) / 1000
```

```
## [1] 0.055
```

If the average loan is $20,000, the credit institute would lose more than one million dollar if it would grant everybody a credit:


```r
# average profit * average loan * number of customers
0.055 * 20000 * 1000
```

```
## [1] 1100000
```

Our goal is to find a model which minimizes the costs (and thereby maximizes the expected profit).

### A First Model

For our first model, we choose an ordinary logistic regression (implemented in the add-on package [mlr3learners](https://mlr3learners.mlr-org.com)).
We first create a classification task, then resample the model using a 10-fold cross validation and extract the resulting confusion matrix:


```r
library("mlr3learners")
learner = lrn("classif.log_reg")
rr = resample(task, learner, rsmp("cv"))

confusion = rr$prediction()$confusion
print(confusion)
```

```
##         truth
## response good bad
##     good  606 154
##     bad    94 146
```

To calculate the average costs like above, we can simply multiply the elements of the confusion matrix with the elements of the previously introduced cost matrix, and sum the values of the resulting matrix:


```r
avg_costs = sum(confusion * costs) / 1000
print(avg_costs)
```

```
## [1] -0.0581
```

With an average loan of \$20,000, the logistic regression yields the following costs:


```r
avg_costs * 20000 * 1000
```

```
## [1] -1162000
```

Instead of losing over \$1,000,000, the credit institute now can expect a profit of more than \$1,000,000.

### Cost-sensitive Measure

Our natural next step would be to further improve the modeling step in order to maximize the profit.
For this purpose we first create a cost-sensitive classification measure which calculates the costs based on our cost matrix.
This allows us to conveniently quantify and compare modeling decisions.
Fortunately, there already is a predefined measure [`Measure`](https://mlr3.mlr-org.com/reference/Measure.html) for this purpose: [`MeasureClassifCosts`](https://mlr3.mlr-org.com/reference/mlr_measures_classif.costs.html):


```r
cost_measure = msr("classif.costs", costs = costs)
print(cost_measure)
```

```
## <MeasureClassifCosts:classif.costs>
## * Packages: -
## * Range: [-Inf, Inf]
## * Minimize: TRUE
## * Properties: requires_task
## * Predict type: response
```

If we now call [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) or [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html), the cost-sensitive measures will be evaluated.
We compare the logistic regression to a simple featureless learner and to a random forest from package [ranger](https://cran.r-project.org/package=ranger) :


```r
learners = list(
  lrn("classif.log_reg"),
  lrn("classif.featureless"),
  lrn("classif.ranger")
)
cv3 = rsmp("cv", folds = 3)
bmr = benchmark(benchmark_grid(task, learners, cv3))
bmr$aggregate(cost_measure)
```

```
##    nr      resample_result       task_id          learner_id resampling_id
## 1:  1 <ResampleResult[21]> german_credit     classif.log_reg            cv
## 2:  2 <ResampleResult[21]> german_credit classif.featureless            cv
## 3:  3 <ResampleResult[21]> german_credit      classif.ranger            cv
##    iters classif.costs
## 1:     3      -0.06386
## 2:     3       0.05504
## 3:     3      -0.05165
```

As expected, the featureless learner is performing comparably bad.
The logistic regression and the random forest work equally well.


### Thresholding

Although we now correctly evaluate the models in a cost-sensitive fashion, the models themselves are unaware of the classification costs.
They assume the same costs for both wrong classification decisions (false positives and false negatives).
Some learners natively support cost-sensitive classification (e.g., XXX).
However, we will concentrate on a more generic approach which works for all models which can predict probabilities for class labels: thresholding.

Most learners can calculate the probability $p$ for the positive class.
If $p$ exceeds the threshold $0.5$, they predict the positive class, and the negative class otherwise.

For our binary classification case of the credit data, the we primarily want to minimize the errors where the model predicts "good", but truth is "bad" (i.e., the number of false positives) as this is the more expensive error.
If we now increase the threshold to values $> 0.5$, we reduce the number of false negatives.
Note that we increase the number of false positives simultaneously, or, in other words, we are trading false positives for false negatives.


```r
# fit models with probability prediction
learner = lrn("classif.log_reg", predict_type = "prob")
rr = resample(task, learner, rsmp("cv"))
p = rr$prediction()
print(p)
```

```
## <PredictionClassif> for 1000 observations:
##     row_id truth response prob.good prob.bad
##          3  good     good   0.95809  0.04191
##          4  good     good   0.68398  0.31602
##          6  good     good   0.62417  0.37583
## ---                                         
##        973   bad      bad   0.08402  0.91598
##        976  good     good   0.80744  0.19256
##        995  good     good   0.93632  0.06368
```

```r
# helper function to try different threshold values interactively
with_threshold = function(p, th) {
  p$set_threshold(th)
  list(confusion = p$confusion, costs = p$score(measures = cost_measure, task = task))
}

with_threshold(p, 0.5)
```

```
## $confusion
##         truth
## response good bad
##     good  609 149
##     bad    91 151
## 
## $costs
## classif.costs 
##      -0.06415
```

```r
with_threshold(p, 0.75)
```

```
## $confusion
##         truth
## response good bad
##     good  469  83
##     bad   231 217
## 
## $costs
## classif.costs 
##      -0.08115
```

```r
with_threshold(p, 1.0)
```

```
## $confusion
##         truth
## response good bad
##     good    0   0
##     bad   700 300
## 
## $costs
## classif.costs 
##             0
```

```r
# TODO: include plot of threshold vs performance
```

Instead of manually trying different threshold values, one uses use [`optimize()`](https://www.rdocumentation.org/packages/stats/topics/optimize) to find a good threshold value w.r.t. our performance measure:


```r
# simple wrapper function which takes a threshold and returns the resulting model performance
# this wrapper is passed to optimize() to find its minimum for thresholds in [0.5, 1]
f = function(th) {
  with_threshold(p, th)$costs
}
best = optimize(f, c(0.5, 1))
print(best)
```

```
## $minimum
## [1] 0.8084
## 
## $objective
## classif.costs 
##      -0.09155
```

```r
# optimized confusion matrix:
with_threshold(p, best$minimum)$confusion
```

```
##         truth
## response good bad
##     good  413  53
##     bad   287 247
```

Note that the function [`optimize()`](https://www.rdocumentation.org/packages/stats/topics/optimize) is intended for unimodal functions and therefore may converge to a local optimum here.
See below for better alternatives to find good threshold values.

### Threshold Tuning

More following soon!

Upcoming sections will entail:

* threshold tuning as pipeline operator
* joint hyperparameter optimization
