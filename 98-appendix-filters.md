## Integrated Filter Methods {#list-filters}

### Standalone filter methods {#fs-filter-list}


|Id                                                                                                |Packages                                                                                                           |Task Types    |Feature Types           |
|:-------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------|:-------------|:-----------------------|
|[`anova`](https://mlr3filters.mlr-org.com/reference/mlr_filters_anova.html)                       |[stats](https://cran.r-project.org/package=stats)                                                                  |classif       |int, dbl                |
|[`auc`](https://mlr3filters.mlr-org.com/reference/mlr_filters_auc.html)                           |[mlr3measures](https://cran.r-project.org/package=mlr3measures)                                                    |classif       |int, dbl                |
|[`carscore`](https://mlr3filters.mlr-org.com/reference/mlr_filters_carscore.html)                 |[care](https://cran.r-project.org/package=care)                                                                    |regr          |dbl                     |
|[`cmim`](https://mlr3filters.mlr-org.com/reference/mlr_filters_cmim.html)                         |[praznik](https://cran.r-project.org/package=praznik)                                                              |classif, regr |int, dbl, fct, ord      |
|[`correlation`](https://mlr3filters.mlr-org.com/reference/mlr_filters_correlation.html)           |[stats](https://cran.r-project.org/package=stats)                                                                  |regr          |int, dbl                |
|[`disr`](https://mlr3filters.mlr-org.com/reference/mlr_filters_disr.html)                         |[praznik](https://cran.r-project.org/package=praznik)                                                              |classif       |int, dbl, fct, ord      |
|[`find_correlation`](https://mlr3filters.mlr-org.com/reference/mlr_filters_find_correlation.html) |[stats](https://cran.r-project.org/package=stats)                                                                  |classif, regr |int, dbl                |
|variable_importance                                                                               |[rpart](https://cran.r-project.org/package=rpart)                                                                  |classif       |lgl, int, dbl, fct, ord |
|[`information_gain`](https://mlr3filters.mlr-org.com/reference/mlr_filters_information_gain.html) |[FSelectorRcpp](https://cran.r-project.org/package=FSelectorRcpp)                                                  |classif, regr |int, dbl, fct, ord      |
|[`jmi`](https://mlr3filters.mlr-org.com/reference/mlr_filters_jmi.html)                           |[praznik](https://cran.r-project.org/package=praznik)                                                              |classif       |int, dbl, fct, ord      |
|[`jmim`](https://mlr3filters.mlr-org.com/reference/mlr_filters_jmim.html)                         |[praznik](https://cran.r-project.org/package=praznik)                                                              |classif       |int, dbl, fct, ord      |
|[`kruskal_test`](https://mlr3filters.mlr-org.com/reference/mlr_filters_kruskal_test.html)         |[stats](https://cran.r-project.org/package=stats)                                                                  |classif       |int, dbl                |
|[`mim`](https://mlr3filters.mlr-org.com/reference/mlr_filters_mim.html)                           |[praznik](https://cran.r-project.org/package=praznik)                                                              |classif       |int, dbl, fct, ord      |
|[`mrmr`](https://mlr3filters.mlr-org.com/reference/mlr_filters_mrmr.html)                         |[praznik](https://cran.r-project.org/package=praznik)                                                              |classif       |int, dbl, fct, ord      |
|[`njmim`](https://mlr3filters.mlr-org.com/reference/mlr_filters_njmim.html)                       |[praznik](https://cran.r-project.org/package=praznik)                                                              |classif       |int, dbl, fct, ord      |
|[`performance`](https://mlr3filters.mlr-org.com/reference/mlr_filters_performance.html)           |[mlr3measures](https://cran.r-project.org/package=mlr3measures), [rpart](https://cran.r-project.org/package=rpart) |classif       |lgl, int, dbl, fct, ord |
|[`permutation`](https://mlr3filters.mlr-org.com/reference/mlr_filters_permutation.html)           |[mlr3measures](https://cran.r-project.org/package=mlr3measures), [rpart](https://cran.r-project.org/package=rpart) |classif       |lgl, int, dbl, fct, ord |
|[`variance`](https://mlr3filters.mlr-org.com/reference/mlr_filters_variance.html)                 |[stats](https://cran.r-project.org/package=stats)                                                                  |classif, regr |int, dbl                |

### Learners With Embedded Filter Methods {#fs-filter-embedded-list}


```
##  [1] "classif.featureless" "classif.ranger"      "classif.rpart"      
##  [4] "classif.xgboost"     "regr.featureless"    "regr.ranger"        
##  [7] "regr.rpart"          "regr.xgboost"        "surv.ranger"        
## [10] "surv.rpart"          "surv.xgboost"
```
