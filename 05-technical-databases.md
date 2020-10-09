## Database Backends {#backends}

In mlr3, [`Task`](https://mlr3.mlr-org.com/reference/Task.html)s store their data in an abstract data format, the [`DataBackend`](https://mlr3.mlr-org.com/reference/DataBackend.html).
The default backend uses [data.table](https://cran.r-project.org/package=data.table) via the [`DataBackendDataTable`](https://mlr3.mlr-org.com/reference/DataBackendDataTable.html) as an in-memory data base.

For larger data, or when working with many tasks in parallel, it can be advantageous to interface an out-of-memory data.
We use the excellent R package [dbplyr](https://cran.r-project.org/package=dbplyr) which extends [dplyr](https://cran.r-project.org/package=dplyr) to work on many popular data bases like [MariaDB](https://mariadb.org/), [PostgreSQL](https://www.postgresql.org/) or [SQLite](https://www.sqlite.org).

### Use Case: NYC Flights

To generate a halfway realistic scenario, we use the NYC flights data set from package [nycflights13](https://cran.r-project.org/package=nycflights13):


```r
# load data
requireNamespace("DBI")
```

```
## Loading required namespace: DBI
```

```r
requireNamespace("RSQLite")
```

```
## Loading required namespace: RSQLite
```

```r
requireNamespace("nycflights13")
```

```
## Loading required namespace: nycflights13
```

```r
data("flights", package = "nycflights13")
str(flights)
```

```
## tibble [336,776 × 19] (S3: tbl_df/tbl/data.frame)
##  $ year          : int [1:336776] 2013 2013 2013 2013 2013 2013 2013 2013 2013 2013 ...
##  $ month         : int [1:336776] 1 1 1 1 1 1 1 1 1 1 ...
##  $ day           : int [1:336776] 1 1 1 1 1 1 1 1 1 1 ...
##  $ dep_time      : int [1:336776] 517 533 542 544 554 554 555 557 557 558 ...
##  $ sched_dep_time: int [1:336776] 515 529 540 545 600 558 600 600 600 600 ...
##  $ dep_delay     : num [1:336776] 2 4 2 -1 -6 -4 -5 -3 -3 -2 ...
##  $ arr_time      : int [1:336776] 830 850 923 1004 812 740 913 709 838 753 ...
##  $ sched_arr_time: int [1:336776] 819 830 850 1022 837 728 854 723 846 745 ...
##  $ arr_delay     : num [1:336776] 11 20 33 -18 -25 12 19 -14 -8 8 ...
##  $ carrier       : chr [1:336776] "UA" "UA" "AA" "B6" ...
##  $ flight        : int [1:336776] 1545 1714 1141 725 461 1696 507 5708 79 301 ...
##  $ tailnum       : chr [1:336776] "N14228" "N24211" "N619AA" "N804JB" ...
##  $ origin        : chr [1:336776] "EWR" "LGA" "JFK" "JFK" ...
##  $ dest          : chr [1:336776] "IAH" "IAH" "MIA" "BQN" ...
##  $ air_time      : num [1:336776] 227 227 160 183 116 150 158 53 140 138 ...
##  $ distance      : num [1:336776] 1400 1416 1089 1576 762 ...
##  $ hour          : num [1:336776] 5 5 5 5 6 5 6 6 6 6 ...
##  $ minute        : num [1:336776] 15 29 40 45 0 58 0 0 0 0 ...
##  $ time_hour     : POSIXct[1:336776], format: "2013-01-01 05:00:00" "2013-01-01 05:00:00" ...
```

```r
# add column of unique row ids
flights$row_id = 1:nrow(flights)

# create sqlite database in temporary file
path = tempfile("flights", fileext = ".sqlite")
con = DBI::dbConnect(RSQLite::SQLite(), path)
tbl = DBI::dbWriteTable(con, "flights", as.data.frame(flights))
DBI::dbDisconnect(con)

# remove in-memory data
rm(flights)
```

### Preprocessing with `dplyr`

With the SQLite database in `path`, we now re-establish a connection and switch to [dplyr](https://cran.r-project.org/package=dplyr)/[dbplyr](https://cran.r-project.org/package=dbplyr) for some essential preprocessing.


```r
# establish connection
con = DBI::dbConnect(RSQLite::SQLite(), path)

# select the "flights" table, enter dplyr
library("dplyr")
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
library("dbplyr")
```

```
## 
## Attaching package: 'dbplyr'
```

```
## The following objects are masked from 'package:dplyr':
## 
##     ident, sql
```

```r
tbl = tbl(con, "flights")
```

First, we select a subset of columns to work on:


```r
keep = c("row_id", "year", "month", "day", "hour", "minute", "dep_time",
  "arr_time", "carrier", "flight", "air_time", "distance", "arr_delay")
tbl = select(tbl, keep)
```

Additionally, we remove those observations where the arrival delay (`arr_delay`) has a missing value:


```r
tbl = filter(tbl, !is.na(arr_delay))
```

To keep runtime reasonable for this toy example, we filter the data to only use every second row:


```r
tbl = filter(tbl, row_id %% 2 == 0)
```

The factor levels of the feature `carrier` are merged so that infrequent carriers are replaced by level "other":


```r
tbl = mutate(tbl, carrier = case_when(
    carrier %in% c("OO", "HA", "YV", "F9", "AS", "FL", "VX", "WN") ~ "other",
    TRUE ~ carrier)
)
```

### DataBackendDplyr

The processed table is now used to create a [`mlr3db::DataBackendDplyr`](https://mlr3db.mlr-org.com/reference/DataBackendDplyr.html) from [mlr3db](https://mlr3db.mlr-org.com):


```r
library("mlr3db")
b = as_data_backend(tbl, primary_key = "row_id")
```

We can now use the interface of [`DataBackend`](https://mlr3.mlr-org.com/reference/DataBackend.html) to query some basic information of the data:


```r
b$nrow
```

```
## [1] 163707
```

```r
b$ncol
```

```
## [1] 13
```

```r
b$head()
```

```
##    row_id year month day hour minute dep_time arr_time carrier flight air_time
## 1:      2 2013     1   1    5     29      533      850      UA   1714      227
## 2:      4 2013     1   1    5     45      544     1004      B6    725      183
## 3:      6 2013     1   1    5     58      554      740      UA   1696      150
## 4:      8 2013     1   1    6      0      557      709      EV   5708       53
## 5:     10 2013     1   1    6      0      558      753      AA    301      138
## 6:     12 2013     1   1    6      0      558      853      B6     71      158
##    distance arr_delay
## 1:     1416        20
## 2:     1576       -18
## 3:      719        12
## 4:      229       -14
## 5:      733         8
## 6:     1005        -3
```

Note that the [`DataBackendDplyr`](https://mlr3db.mlr-org.com/reference/DataBackendDplyr.html) does not know about any rows or columns we have filtered out with [dplyr](https://cran.r-project.org/package=dplyr) before, it just operates on the view we provided.

### Model fitting

We create the following [mlr3](https://mlr3.mlr-org.com) objects:

* A [`regression task`](https://mlr3.mlr-org.com/reference/TaskRegr.html), based on the previously created [`mlr3db::DataBackendDplyr`](https://mlr3db.mlr-org.com/reference/DataBackendDplyr.html).
* A regression learner ([`regr.rpart`](https://mlr3.mlr-org.com/reference/mlr_learners_regr.rpart.html)).
* A resampling strategy: 3 times repeated subsampling using 2\% of the observations for training ("[`subsampling`](https://mlr3.mlr-org.com/reference/mlr_resamplings_subsampling.html)")
* Measures "[`mse`](https://mlr3.mlr-org.com/reference/mlr_measures_regr.mse.html)", "[`time_train`](https://mlr3.mlr-org.com/reference/mlr_measures_elapsed_time.html)" and "[`time_predict`](https://mlr3.mlr-org.com/reference/mlr_measures_elapsed_time.html)"


```r
task = TaskRegr$new("flights_sqlite", b, target = "arr_delay")
learner = lrn("regr.rpart")
measures = mlr_measures$mget(c("regr.mse", "time_train", "time_predict"))
resampling = rsmp("subsampling")
resampling$param_set$values = list(repeats = 3, ratio = 0.02)
```

We pass all these objects to [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) to perform a simple resampling with three iterations.
In each iteration, only the required subset of the data is queried from the SQLite data base and passed to [`rpart::rpart()`](https://www.rdocumentation.org/packages/rpart/topics/rpart):


```r
rr = resample(task, learner, resampling)
print(rr)
```

```
## <ResampleResult> of 3 iterations
## * Task: flights_sqlite
## * Learner: regr.rpart
## * Warnings: 0 in 0 iterations
## * Errors: 0 in 0 iterations
```

```r
rr$aggregate(measures)
```

```
##     regr.mse   time_train time_predict 
##         1223            0            0
```

### Cleanup

Finally, we remove the `tbl` object and close the connection.


```r
rm(tbl)
DBI::dbDisconnect(con)
```
