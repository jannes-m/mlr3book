## The Building Blocks: PipeOps {#pipe-pipeops}

The building blocks of [mlr3pipelines](https://mlr3pipelines.mlr-org.com) are **PipeOp**-objects (PO).
They can be constructed directly using `PipeOp<NAME>$new()`, but the recommended way is to retrieve them from the `mlr_pipeops` dictionary:


```r
library("mlr3pipelines")
as.data.table(mlr_pipeops)
```

```
##                       key           packages                             tags
##  1:                boxcox      bestNormalize                   data transform
##  2:                branch                                                meta
##  3:                 chunk                                                meta
##  4:        classbalancing                      imbalanced data,data transform
##  5:            classifavg              stats                         ensemble
##  6:          classweights                      imbalanced data,data transform
##  7:              colapply                                      data transform
##  8:       collapsefactors                                      data transform
##  9:              colroles                                      data transform
## 10:                  copy                                                meta
## 11:          datefeatures                                      data transform
## 12:                encode              stats            encode,data transform
## 13:          encodeimpact                               encode,data transform
## 14:            encodelmer        lme4,nloptr            encode,data transform
## 15:          featureunion                                            ensemble
## 16:                filter                    feature selection,data transform
## 17:            fixfactors                            robustify,data transform
## 18:               histbin           graphics                   data transform
## 19:                   ica            fastICA                   data transform
## 20:        imputeconstant                                            missings
## 21:            imputehist           graphics                         missings
## 22:         imputelearner                                            missings
## 23:            imputemean                                            missings
## 24:          imputemedian              stats                         missings
## 25:            imputemode                                            missings
## 26:             imputeoor                                            missings
## 27:          imputesample                                            missings
## 28:             kernelpca            kernlab                   data transform
## 29:               learner                                             learner
## 30:            learner_cv                     learner,ensemble,data transform
## 31:               missind                             missings,data transform
## 32:           modelmatrix              stats                   data transform
## 33:     multiplicityexply                                        multiplicity
## 34:     multiplicityimply                                        multiplicity
## 35:                mutate                                      data transform
## 36:                   nmf           MASS,NMF                   data transform
## 37:                   nop                                                meta
## 38:              ovrsplit                       target transform,multiplicity
## 39:              ovrunite                               multiplicity,ensemble
## 40:                   pca                                      data transform
## 41:                 proxy                                                meta
## 42:           quantilebin              stats                   data transform
## 43:      randomprojection                                      data transform
## 44:        randomresponse                                            abstract
## 45:               regravg                                            ensemble
## 46:       removeconstants                            robustify,data transform
## 47:         renamecolumns                                      data transform
## 48:             replicate                                        multiplicity
## 49:                 scale                                      data transform
## 50:           scalemaxabs                                      data transform
## 51:            scalerange                                      data transform
## 52:                select                    feature selection,data transform
## 53:                 smote        smotefamily   imbalanced data,data transform
## 54:           spatialsign                                      data transform
## 55:             subsample                                      data transform
## 56:          targetinvert                                            abstract
## 57:          targetmutate                                    target transform
## 58: targettrafoscalerange                                    target transform
## 59:        textvectorizer quanteda,stopwords                   data transform
## 60:             threshold                                    target transform
## 61:         tunethreshold              bbotk                 target transform
## 62:              unbranch                                                meta
## 63:                vtreat             vtreat                   data transform
## 64:            yeojohnson      bestNormalize                   data transform
##                       key           packages                             tags
##                                            feature_types input.num output.num
##  1:                                      numeric,integer         1          1
##  2:                                                   NA         1         NA
##  3:                                                   NA         1         NA
##  4: logical,integer,numeric,character,factor,ordered,...         1          1
##  5:                                                   NA        NA          1
##  6: logical,integer,numeric,character,factor,ordered,...         1          1
##  7: logical,integer,numeric,character,factor,ordered,...         1          1
##  8:                                       factor,ordered         1          1
##  9: logical,integer,numeric,character,factor,ordered,...         1          1
## 10:                                                   NA         1         NA
## 11:                                              POSIXct         1          1
## 12:                                       factor,ordered         1          1
## 13:                                       factor,ordered         1          1
## 14:                                       factor,ordered         1          1
## 15:                                                   NA        NA          1
## 16: logical,integer,numeric,character,factor,ordered,...         1          1
## 17:                                       factor,ordered         1          1
## 18:                                      numeric,integer         1          1
## 19:                                      numeric,integer         1          1
## 20: logical,integer,numeric,character,factor,ordered,...         1          1
## 21:                                      integer,numeric         1          1
## 22:                               logical,factor,ordered         1          1
## 23:                                      numeric,integer         1          1
## 24:                                      numeric,integer         1          1
## 25:               factor,integer,logical,numeric,ordered         1          1
## 26:             character,factor,integer,numeric,ordered         1          1
## 27:               factor,integer,logical,numeric,ordered         1          1
## 28:                                      numeric,integer         1          1
## 29:                                                   NA         1          1
## 30: logical,integer,numeric,character,factor,ordered,...         1          1
## 31: logical,integer,numeric,character,factor,ordered,...         1          1
## 32: logical,integer,numeric,character,factor,ordered,...         1          1
## 33:                                                   NA         1         NA
## 34:                                                   NA        NA          1
## 35: logical,integer,numeric,character,factor,ordered,...         1          1
## 36:                                      numeric,integer         1          1
## 37:                                                   NA         1          1
## 38:                                                   NA         1          1
## 39:                                                   NA         1          1
## 40:                                      numeric,integer         1          1
## 41:                                                   NA        NA          1
## 42:                                      numeric,integer         1          1
## 43:                                      numeric,integer         1          1
## 44:                                                   NA         1          1
## 45:                                                   NA        NA          1
## 46: logical,integer,numeric,character,factor,ordered,...         1          1
## 47: logical,integer,numeric,character,factor,ordered,...         1          1
## 48:                                                   NA         1          1
## 49:                                      numeric,integer         1          1
## 50:                                      numeric,integer         1          1
## 51:                                      numeric,integer         1          1
## 52: logical,integer,numeric,character,factor,ordered,...         1          1
## 53: logical,integer,numeric,character,factor,ordered,...         1          1
## 54:                                      numeric,integer         1          1
## 55: logical,integer,numeric,character,factor,ordered,...         1          1
## 56:                                                   NA         2          1
## 57:                                                   NA         1          2
## 58:                                                   NA         1          2
## 59:                                            character         1          1
## 60:                                                   NA         1          1
## 61:                                                   NA         1          1
## 62:                                                   NA        NA          1
## 63: logical,integer,numeric,character,factor,ordered,...         1          1
## 64:                                      numeric,integer         1          1
##                                            feature_types input.num output.num
##     input.type.train  input.type.predict output.type.train output.type.predict
##  1:             Task                Task              Task                Task
##  2:                *                   *                 *                   *
##  3:             Task                Task              Task                Task
##  4:      TaskClassif         TaskClassif       TaskClassif         TaskClassif
##  5:             NULL   PredictionClassif              NULL   PredictionClassif
##  6:      TaskClassif         TaskClassif       TaskClassif         TaskClassif
##  7:             Task                Task              Task                Task
##  8:             Task                Task              Task                Task
##  9:             Task                Task              Task                Task
## 10:                *                   *                 *                   *
## 11:             Task                Task              Task                Task
## 12:             Task                Task              Task                Task
## 13:             Task                Task              Task                Task
## 14:             Task                Task              Task                Task
## 15:             Task                Task              Task                Task
## 16:             Task                Task              Task                Task
## 17:             Task                Task              Task                Task
## 18:             Task                Task              Task                Task
## 19:             Task                Task              Task                Task
## 20:             Task                Task              Task                Task
## 21:             Task                Task              Task                Task
## 22:             Task                Task              Task                Task
## 23:             Task                Task              Task                Task
## 24:             Task                Task              Task                Task
## 25:             Task                Task              Task                Task
## 26:             Task                Task              Task                Task
## 27:             Task                Task              Task                Task
## 28:             Task                Task              Task                Task
## 29:      TaskClassif         TaskClassif              NULL   PredictionClassif
## 30:      TaskClassif         TaskClassif       TaskClassif         TaskClassif
## 31:             Task                Task              Task                Task
## 32:             Task                Task              Task                Task
## 33:              [*]                 [*]                 *                   *
## 34:                *                   *               [*]                 [*]
## 35:             Task                Task              Task                Task
## 36:             Task                Task              Task                Task
## 37:                *                   *                 *                   *
## 38:      TaskClassif         TaskClassif     [TaskClassif]       [TaskClassif]
## 39:           [NULL] [PredictionClassif]              NULL   PredictionClassif
## 40:             Task                Task              Task                Task
## 41:                *                   *                 *                   *
## 42:             Task                Task              Task                Task
## 43:             Task                Task              Task                Task
## 44:             NULL          Prediction              NULL          Prediction
## 45:             NULL      PredictionRegr              NULL      PredictionRegr
## 46:             Task                Task              Task                Task
## 47:             Task                Task              Task                Task
## 48:                *                   *               [*]                 [*]
## 49:             Task                Task              Task                Task
## 50:             Task                Task              Task                Task
## 51:             Task                Task              Task                Task
## 52:             Task                Task              Task                Task
## 53:             Task                Task              Task                Task
## 54:             Task                Task              Task                Task
## 55:             Task                Task              Task                Task
## 56:        NULL,NULL function,Prediction              NULL          Prediction
## 57:             Task                Task         NULL,Task       function,Task
## 58:         TaskRegr            TaskRegr     NULL,TaskRegr   function,TaskRegr
## 59:             Task                Task              Task                Task
## 60:             NULL   PredictionClassif              NULL   PredictionClassif
## 61:             Task                Task              NULL          Prediction
## 62:                *                   *                 *                   *
## 63:             Task                Task              Task                Task
## 64:             Task                Task              Task                Task
##     input.type.train  input.type.predict output.type.train output.type.predict
```

Single POs can be created using `mlr_pipeops$get(<name>)`:


```r
pca = mlr_pipeops$get("pca")
```

or using **syntactic sugar**


```r
pca = po("pca")
```

Some POs require additional arguments for construction:


```r
learner = mlr_pipeops$get("learner")

# Error in as_learner(learner) : argument "learner" is missing, with no default argument "learner" is missing, with no default
```


```r
learner = mlr_pipeops$get("learner", mlr_learners$get("classif.rpart"))
```

or in short `po("learner", lrn("classif.rpart"))`.

Hyperparameters of POs can be set through the `param_vals` argument.
Here we set the fraction of features for a filter:


```r
filter = mlr_pipeops$get("filter",
  filter = mlr3filters::FilterVariance$new(),
  param_vals = list(filter.frac = 0.5))
```

or in short notation:


```r
po("filter", mlr3filters::FilterVariance$new(), filter.frac = 0.5)
```

The figure below shows an exemplary `PipeOp`.
It takes an input, transforms it during `.$train` and `.$predict` and returns data:

<img src="images/po_viz.png" style="display: block; margin: auto;" />
