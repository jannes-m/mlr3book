## Learners {#learners}

Objects of class [`mlr3::Learner`](https://mlr3.mlr-org.com/reference/Learner.html) provide a unified interface to many popular machine learning algorithms in R.
They consist of methods to train and predict a model for a [`mlr3::Task`](https://mlr3.mlr-org.com/reference/Task.html) and provide meta-information about the learners, such as the hyperparameters you can set.

The package ships with a minimal set of classification and regression learners to avoid lots of dependencies:

* [`mlr_learners_classif.featureless`](https://mlr3.mlr-org.com/reference/mlr_learners_classif.featureless.html): Simple baseline classification learner, constantly predicts the label that is most frequent in the training set.
* [`mlr_learners_classif.rpart`](https://mlr3.mlr-org.com/reference/mlr_learners_classif.rpart.html): Single classification tree from [rpart](https://cran.r-project.org/package=rpart).
* [`mlr_learners_regr.featureless`](https://mlr3.mlr-org.com/reference/mlr_learners_regr.featureless.html): Simple baseline regression learner, constantly predicts with the mean.
* [`mlr_learners_regr.rpart`](https://mlr3.mlr-org.com/reference/mlr_learners_regr.rpart.html): Single regression tree from [rpart](https://cran.r-project.org/package=rpart).

Some of the most popular learners are connected via the [mlr3learners](https://mlr3learners.mlr-org.com) package:

* (penalized) linear and logistic regression
* $k$-Nearest Neighbors regression and classification
* Linear and Quadratic Discriminant Analysis
* Naive Bayes
* Support-Vector machines
* Gradient Boosting
* Random Regression Forests and Random Classification Forests
* Kriging

More learners are collected on GitHub in the [mlr3learners organization](https://github.com/mlr3learners/).
Their state is also listed on the [wiki](https://github.com/mlr-org/mlr3learners/wiki) of the [mlr3learners repository](https://github.com/mlr-org/mlr3learners/).
Below a graphical illustration of the role of a learner:


<img src="images/learner.svg" style="display: block; margin: auto;" />


The base class of each learner is [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html), specialized for regression as [`LearnerRegr`](https://mlr3.mlr-org.com/reference/LearnerRegr.html) and for classification as [`LearnerClassif`](https://mlr3.mlr-org.com/reference/LearnerClassif.html).
In contrast to the [`Task`](https://mlr3.mlr-org.com/reference/Task.html), the creation of a custom Learner is usually not required and a more advanced topic.
Hence, we refer the reader to Section \@ref(extending-learners) and proceed with an overview of the interface of already implemented learners.

### Predefined Learners {#learners-predefined}

Similar to [`mlr_tasks`](https://mlr3.mlr-org.com/reference/mlr_tasks.html), the [`Dictionary`](https://mlr3misc.mlr-org.com/reference/Dictionary.html) [`mlr_learners`](https://mlr3.mlr-org.com/reference/mlr_learners.html) can be queried for available learners:


```r
library("mlr3learners")
mlr_learners
```

```
## <DictionaryLearner> with 28 stored values
## Keys: classif.cv_glmnet, classif.debug, classif.featureless,
##   classif.glmnet, classif.kknn, classif.lda, classif.log_reg,
##   classif.multinom, classif.naive_bayes, classif.qda, classif.ranger,
##   classif.rpart, classif.svm, classif.xgboost, regr.cv_glmnet,
##   regr.featureless, regr.glmnet, regr.kknn, regr.km, regr.lm,
##   regr.ranger, regr.rpart, regr.svm, regr.xgboost, surv.cv_glmnet,
##   surv.glmnet, surv.ranger, surv.xgboost
```

Each learner has the following information:

* `feature_types`: the type of features the learner can deal with.
* `packages`: the packages required to train a model with this learner and make predictions.
* `properties`: additional properties and capabilities.
  For example, a learner has the property "missings" if it is able to handle missing feature values, and "importance" if it computes and allows to extract data on the relative importance of the features.
  A complete list of these is available in the mlr3 reference on [regression learners](https://mlr3.mlr-org.com/reference/LearnerRegr.html#construction) and [classification learners](https://mlr3.mlr-org.com/reference/LearnerClassif.html#construction).
* `predict_types`: possible prediction types. For example, a classification learner can predict labels ("response") or probabilities ("prob"). For a complete list of possible predict types see the [mlr3 reference](https://mlr3.mlr-org.com/reference/Learner.html#construction).

For a tabular overview of integrated learners, see Section \@ref(list-learners).

You can get a specific learner using its `id`, listed under `key` in the dictionary:


```r
learner = mlr_learners$get("classif.rpart")
print(learner)
```

```
## <LearnerClassifRpart:classif.rpart>
## * Model: -
## * Parameters: xval=0
## * Packages: rpart
## * Predict Type: response
## * Feature types: logical, integer, numeric, factor, ordered
## * Properties: importance, missings, multiclass, selected_features,
##   twoclass, weights
```

The field `param_set` stores a description of the hyperparameters the learner has, their ranges, defaults, and current values:


```r
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

The set of current hyperparameter values is stored in the `values` field of the `param_set` field.
You can change the current hyperparameter values by assigning a named list to this field:


```r
learner$param_set$values = list(cp = 0.01, xval = 0)
learner
```

```
## <LearnerClassifRpart:classif.rpart>
## * Model: -
## * Parameters: cp=0.01, xval=0
## * Packages: rpart
## * Predict Type: response
## * Feature types: logical, integer, numeric, factor, ordered
## * Properties: importance, missings, multiclass, selected_features,
##   twoclass, weights
```

Note that this operation just overwrites all previously set parameters.
If you just want to add or update hyperparameters, you can use [`mlr3misc::insert_named()`](https://mlr3misc.mlr-org.com/reference/insert_named.html):


```r
learner$param_set$values = mlr3misc::insert_named(
  learner$param_set$values,
  list(cp = 0.02, minsplit = 2)
)
learner
```

```
## <LearnerClassifRpart:classif.rpart>
## * Model: -
## * Parameters: cp=0.02, xval=0, minsplit=2
## * Packages: rpart
## * Predict Type: response
## * Feature types: logical, integer, numeric, factor, ordered
## * Properties: importance, missings, multiclass, selected_features,
##   twoclass, weights
```

This updates `cp` to `0.02`, sets `minsplit` to `2` and keeps the previously set parameter `xval`.

Again, there is an alternative to writing down the lengthy `mlr_learners$get()` part: [`lrn()`](https://mlr3.mlr-org.com/reference/mlr_sugar.html).
This function additionally allows to construct learners with specific hyperparameters or settings of a different identifier in one go:


```r
lrn("classif.rpart", id = "rp", cp = 0.001)
```

```
## <LearnerClassifRpart:rp>
## * Model: -
## * Parameters: xval=0, cp=0.001
## * Packages: rpart
## * Predict Type: response
## * Feature types: logical, integer, numeric, factor, ordered
## * Properties: importance, missings, multiclass, selected_features,
##   twoclass, weights
```

If you pass hyperparameters here, they are added to the parameters in a [`insert_named()`](https://mlr3misc.mlr-org.com/reference/insert_named.html)-fashion.
