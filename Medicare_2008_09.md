# US CMMS Medical Claims: bucket2009 classification
bdanalytics  

**  **    
**Date: (Tue) Apr 07, 2015**    

# Introduction:  

Data: 
Source: 
    Training:   https://courses.edx.org/c4x/MITx/15.071x_2/asset/ClaimsData.csv.zip  
    New:        <newdt_url>  
Time period: 



# Synopsis:

Based on analysis utilizing <> techniques, <conclusion heading>:  

### ![](<filename>.png)

## Potential next steps include:
- Probability handling for multinomials vs. desired binomial outcome
-   ROCR currently supports only evaluation of binary classification tasks (version 1.0.7)
-   extensions toward multiclass classification are scheduled for the next release

- Skip trControl.method="cv" for dummy classifier ?
- Add custom model to caret for a dummy (baseline) classifier (binomial & multinomial) that generates proba/outcomes which mimics the freq distribution of glb_rsp_var values; Right now glb_dmy_glm_mdl always generates most frequent outcome in training data
- glm_dmy_mdl should use the same method as glm_sel_mdl until custom dummy classifer is implemented

- Prediction accuracy scatter graph:
-   Change shapes to c("+", "x") instead of dots & triangles
-   Add tiles (raw vs. PCA)
-   Use shiny for drop-down of "important" features
-   Use plot.ly for interactive plots ?

- Move glb_analytics_diag_plots to mydsutils.R: (+) Easier to debug (-) Too many glb vars used
- Add print(ggplot.petrinet(glb_analytics_pn) + coord_flip()) at the end of every major chunk
- Parameterize glb_analytics_pn
- Move glb_impute_missing_data to mydsutils.R: (-) Too many glb vars used; glb_<>_df reassigned
- Replicate myrun_mdl_classification features in myrun_mdl_regression
- Do non-glm methods handle interaction terms ?
- f-score computation for classifiers should be summation across outcomes (not just the desired one ?)
- Add accuracy computation to glb_dmy_mdl in predict.data.new chunk
- Why does splitting fit.data.training.all chunk into separate chunks add an overhead of ~30 secs ? It's not rbind b/c other chunks have lower elapsed time. Is it the number of plots ?
- Incorporate code chunks in print_sessionInfo
- Test against 
    - projects in github.com/bdanalytics
    - lectures in jhu-datascience track

# Analysis: 

```r
rm(list=ls())
set.seed(12345)
options(stringsAsFactors=FALSE)
source("~/Dropbox/datascience/R/mydsutils.R")
source("~/Dropbox/datascience/R/myplot.R")
source("~/Dropbox/datascience/R/mypetrinet.R")
# Gather all package requirements here
#suppressPackageStartupMessages(require())
#packageVersion("snow")

#require(sos); findFn("pinv", maxPages=2, sortby="MaxScore")

# Analysis control global variables
glb_trnng_url <- "https://courses.edx.org/c4x/MITx/15.071x_2/asset/ClaimsData.csv.zip"
glb_newdt_url <- "<newdt_url>"
glb_is_separate_newent_dataset <- FALSE    # or TRUE
glb_split_entity_newent_datasets <- TRUE   # or FALSE
glb_split_newdata_method <- "sample"          # "condition" or "sample"
glb_split_newdata_condition <- "<col_name> <condition_operator> <value>"    # or NULL
glb_split_newdata_size_ratio <- 0.4               # > 0 & < 1
glb_split_sample.seed <- 88               # or any integer
glb_max_obs <- 1000 # or NULL

glb_is_regression <- FALSE; glb_is_classification <- TRUE

glb_rsp_var_raw <- "bucket2009"

# for classification, the response variable has to be a factor
#   especially for random forests (method="rf")
glb_rsp_var <- "bucket2009.fctr"    # or glb_rsp_var_raw

# if the response factor is based on numbers e.g (0/1 vs. "A"/"B"), 
#   caret predict(..., type="prob") crashes
glb_map_rsp_raw_to_var <- function(raw) {
    as.factor(paste0("B", raw))
} # or NULL
#glb_map_rsp_raw_to_var(c(1, 2, 3, 4, 5))

glb_map_rsp_var_to_raw <- function(var) {
    as.numeric(var)
} # or NULL
#glb_map_rsp_var_to_raw(glb_map_rsp_raw_to_var(c(1, 2, 3, 4, 5)))

if ((glb_rsp_var != glb_rsp_var_raw) & is.null(glb_map_rsp_raw_to_var))
    stop("glb_map_rsp_raw_to_var function expected")

glb_rsp_var_out <- paste0(glb_rsp_var, ".prediction")
glb_id_vars <- NULL # or c("<id_var>")

glb_exclude_vars_as_features <- union(glb_id_vars, c(glb_rsp_var_raw, ".rnorm"))
# List transformed vars  
glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, 
                                      NULL)     # or c("<col_name>")
# List feats that shd be excluded due to known causation by prediction variable
glb_exclude_vars_as_features <- union(union(glb_exclude_vars_as_features, 
                        paste(glb_rsp_var_out, c("", ".proba"), sep="")),
                                    c("reimbursement2009")) # or NULL

glb_impute_na_data <- FALSE            # or TRUE
glb_mice_complete.seed <- 144               # or any integer

glb_fin_mdl <- glb_sel_mdl <- glb_dmy_mdl <- NULL; 
# Classification
#glb_models_method_vctr <- c("glm", "rpart", "rf")   # Binomials
#glb_models_method_vctr <- c("rpart", "rf")          # Multinomials
glb_models_method_vctr <- c("rpart")          # Multinomials - this exercise

glb_models_lst <- list(); glb_models_df <- data.frame()
glb_loss_mtrx <- matrix(c(0,1,2,3,4,
                          2,0,1,2,3,
                          4,2,0,1,2,
                          6,4,2,0,1,
                          8,6,4,2,0
                          ), byrow=TRUE, nrow=5)    # or NULL
glb_loss_smmry <- function(data, lev=NULL, model=NULL) {
    confusion_df <- mycompute_confusion_df(data, "obs", "pred")    
    confusion_mtrx <- as.matrix(confusion_df[, -1])
    confusion_mtrx <- cbind(confusion_mtrx, 
                              matrix(rep(0, 
        (nrow(confusion_mtrx) - ncol(confusion_mtrx)) * nrow(confusion_mtrx)), 
                                     byrow=TRUE, nrow=nrow(confusion_mtrx)))
    metric <- sum(confusion_mtrx * glb_loss_mtrx) / nrow(data)
    names(metric) <- "loss.error"
    return(metric)
}

glb_tune_models_df <- 
   rbind(
    data.frame(parameter="cp", min=0.00, max=0.04, by=0.01), 
    data.frame(parameter="mtry", min=2, max=4, by=1)
        ) 
# or NULL
glb_n_cv_folds <- 3 # or NULL

glb_clf_proba_threshold <- NULL # 0.5

# Baseline prediction model feature(s)
glb_bsl_mdl_var <- c("bucket2008") # or NULL

# Depict process
glb_analytics_pn <- petrinet(name="glb_analytics_pn",
                        trans_df=data.frame(id=1:6,
    name=c("data.training.all","data.new",
           "model.selected","model.final",
           "data.training.all.prediction","data.new.prediction"),
    x=c(   -5,-5,-15,-25,-25,-35),
    y=c(   -5, 5,  0,  0, -5,  5)
                        ),
                        places_df=data.frame(id=1:4,
    name=c("bgn","fit.data.training.all","predict.data.new","end"),
    x=c(   -0,   -20,                    -30,               -40),
    y=c(    0,     0,                      0,                 0),
    M0=c(   3,     0,                      0,                 0)
                        ),
                        arcs_df=data.frame(
    begin=c("bgn","bgn","bgn",        
            "data.training.all","model.selected","fit.data.training.all",
            "fit.data.training.all","model.final",    
            "data.new","predict.data.new",
            "data.training.all.prediction","data.new.prediction"),
    end  =c("data.training.all","data.new","model.selected",
            "fit.data.training.all","fit.data.training.all","model.final",
            "data.training.all.prediction","predict.data.new",
            "predict.data.new","data.new.prediction",
            "end","end")
                        ))
#print(ggplot.petrinet(glb_analytics_pn))
print(ggplot.petrinet(glb_analytics_pn) + coord_flip())
```

```
## Loading required package: grid
```

![](Medicare_2008_09_files/figure-html/set_global_options-1.png) 

```r
glb_analytics_avl_objs <- NULL

glb_script_tm <- proc.time()
glb_script_df <- data.frame(chunk_label="import_data", 
                            chunk_step_major=1, chunk_step_minor=0,
                            elapsed=(proc.time() - glb_script_tm)["elapsed"])
print(tail(glb_script_df, 2))
```

```
##         chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed import_data                1                0   0.003
```

## Step `1`: import data

```r
glb_entity_df <- myimport_data(
    url=glb_trnng_url, 
    comment="glb_entity_df", force_header=TRUE,
    print_diagn=(glb_is_separate_newent_dataset | 
                !glb_split_entity_newent_datasets))
```

```
## [1] "Reading file ./data/ClaimsData.csv..."
## [1] "dimensions of data in ./data/ClaimsData.csv: 458,005 rows x 16 cols"
```

```r
if (glb_is_separate_newent_dataset) {
    glb_newent_df <- myimport_data(
        url=glb_newdt_url, 
        comment="glb_newent_df", force_header=TRUE, print_diagn=TRUE)
} else {
    if (!glb_split_entity_newent_datasets) {
        stop("Not implemented yet") 
        glb_newent_df <- glb_entity_df[sample(1:nrow(glb_entity_df),
                                          max(2, nrow(glb_entity_df) / 1000)),]                    
    } else      if (glb_split_newdata_method == "condition") {
            glb_newent_df <- do.call("subset", 
                list(glb_entity_df, parse(text=glb_split_newdata_condition)))
            glb_entity_df <- do.call("subset", 
                list(glb_entity_df, parse(text=paste0("!(", 
                                                      glb_split_newdata_condition,
                                                      ")"))))
        } else if (glb_split_newdata_method == "sample") {
                require(caTools)
                
                set.seed(glb_split_sample.seed)
                split <- sample.split(glb_entity_df[, glb_rsp_var_raw], 
                                      SplitRatio=(1-glb_split_newdata_size_ratio))
                glb_newent_df <- glb_entity_df[!split, ] 
                glb_entity_df <- glb_entity_df[split ,]
        } else stop("glb_split_newdata_method should be %in% c('condition', 'sample')")   

    comment(glb_newent_df) <- "glb_newent_df"
    myprint_df(glb_newent_df)
    str(glb_newent_df)

    if (glb_split_entity_newent_datasets) {
        myprint_df(glb_entity_df)
        str(glb_entity_df)        
    }
}         
```

```
## Loading required package: caTools
```

```
##    age alzheimers arthritis cancer copd depression diabetes heart.failure
## 3   67          0         0      0    0          0        0             0
## 5   67          0         0      0    0          0        0             0
## 6   68          0         0      0    0          0        0             0
## 8   70          0         0      0    0          0        0             0
## 9   67          0         0      0    0          0        0             0
## 10  67          0         0      0    0          0        0             0
##    ihd kidney osteoporosis stroke reimbursement2008 bucket2008
## 3    0      0            0      0                 0          1
## 5    0      0            0      0                 0          1
## 6    0      0            0      0                 0          1
## 8    0      0            0      0                 0          1
## 9    0      0            0      0                 0          1
## 10   0      0            0      0                 0          1
##    reimbursement2009 bucket2009
## 3                  0          1
## 5                  0          1
## 6                  0          1
## 8                  0          1
## 9                  0          1
## 10                 0          1
##        age alzheimers arthritis cancer copd depression diabetes
## 43967   57          0         0      0    0          0        0
## 70246   70          0         0      0    0          0        0
## 165755  78          0         0      0    0          0        0
## 208131  73          0         1      1    0          0        0
## 319113  87          0         0      0    0          1        1
## 446073  72          1         0      1    0          1        1
##        heart.failure ihd kidney osteoporosis stroke reimbursement2008
## 43967              0   0      0            0      0                 0
## 70246              0   0      0            0      0                 0
## 165755             0   0      0            0      0               140
## 208131             0   0      0            1      0              5680
## 319113             0   1      0            0      0              2800
## 446073             1   1      0            1      1             16030
##        bucket2008 reimbursement2009 bucket2009
## 43967           1                 0          1
## 70246           1                 0          1
## 165755          1               720          1
## 208131          2              1250          1
## 319113          1              3330          2
## 446073          3             28000          4
##        age alzheimers arthritis cancer copd depression diabetes
## 457996  60          0         1      0    1          1        1
## 457998  87          0         0      0    1          1        1
## 458001  61          1         0      0    1          1        1
## 458002  90          1         0      0    1          1        1
## 458003  76          0         1      0    1          1        1
## 458005  80          1         0      0    1          1        1
##        heart.failure ihd kidney osteoporosis stroke reimbursement2008
## 457996             1   1      0            1      1             11720
## 457998             1   1      1            0      0             27750
## 458001             1   1      1            1      1             15960
## 458002             1   1      1            0      0             26870
## 458003             1   1      1            1      1             89140
## 458005             1   1      1            0      1             38320
##        bucket2008 reimbursement2009 bucket2009
## 457996          3            142960          5
## 457998          4            148600          5
## 458001          3            154000          5
## 458002          4            155010          5
## 458003          5            155810          5
## 458005          4            189930          5
## 'data.frame':	183202 obs. of  16 variables:
##  $ age              : int  67 67 68 70 67 67 56 48 99 68 ...
##  $ alzheimers       : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ arthritis        : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ cancer           : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ copd             : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ depression       : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ diabetes         : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ heart.failure    : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ ihd              : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ kidney           : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ osteoporosis     : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ stroke           : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ reimbursement2008: int  0 0 0 0 0 0 0 0 0 0 ...
##  $ bucket2008       : int  1 1 1 1 1 1 1 1 1 1 ...
##  $ reimbursement2009: int  0 0 0 0 0 0 0 0 0 0 ...
##  $ bucket2009       : int  1 1 1 1 1 1 1 1 1 1 ...
##  - attr(*, "comment")= chr "glb_newent_df"
##    age alzheimers arthritis cancer copd depression diabetes heart.failure
## 1   85          0         0      0    0          0        0             0
## 2   59          0         0      0    0          0        0             0
## 4   52          0         0      0    0          0        0             0
## 7   75          0         0      0    0          0        0             0
## 11  89          0         0      0    0          0        0             0
## 13  74          0         0      0    0          0        0             0
##    ihd kidney osteoporosis stroke reimbursement2008 bucket2008
## 1    0      0            0      0                 0          1
## 2    0      0            0      0                 0          1
## 4    0      0            0      0                 0          1
## 7    0      0            0      0                 0          1
## 11   0      0            0      0                 0          1
## 13   0      0            0      0                 0          1
##    reimbursement2009 bucket2009
## 1                  0          1
## 2                  0          1
## 4                  0          1
## 7                  0          1
## 11                 0          1
## 13                 0          1
##        age alzheimers arthritis cancer copd depression diabetes
## 138659  69          0         0      0    0          0        0
## 168428  74          1         0      0    0          0        1
## 189703  81          0         0      0    0          0        1
## 225640  78          1         0      0    0          1        0
## 382169  77          1         0      0    1          1        1
## 397881  46          1         0      0    0          0        0
##        heart.failure ihd kidney osteoporosis stroke reimbursement2008
## 138659             0   0      0            0      0                 0
## 168428             0   0      0            1      0               720
## 189703             0   1      0            0      0               690
## 225640             0   0      0            1      0              1540
## 382169             1   1      1            1      1             16400
## 397881             1   1      1            0      0              3700
##        bucket2008 reimbursement2009 bucket2009
## 138659          1               380          1
## 168428          1               750          1
## 189703          1              1020          1
## 225640          1              1490          1
## 382169          3              6620          2
## 397881          2              8470          3
##        age alzheimers arthritis cancer copd depression diabetes
## 457991  76          0         0      0    1          1        1
## 457992  84          0         0      0    1          0        1
## 457997  73          0         0      0    1          1        1
## 457999  83          1         1      0    1          0        1
## 458000  56          0         1      0    1          1        1
## 458004  82          1         0      0    1          0        1
##        heart.failure ihd kidney osteoporosis stroke reimbursement2008
## 457991             1   1      1            1      0             53550
## 457992             1   1      1            0      0              8620
## 457997             1   1      1            1      0             53230
## 457999             1   1      1            1      1             62620
## 458000             1   1      1            1      0             62980
## 458004             1   1      1            1      1             20660
##        bucket2008 reimbursement2009 bucket2009
## 457991          4            131960          5
## 457992          3            133500          5
## 457997          4            147760          5
## 457999          5            148860          5
## 458000          5            151880          5
## 458004          4            158800          5
## 'data.frame':	274803 obs. of  16 variables:
##  $ age              : int  85 59 52 75 89 74 81 86 78 67 ...
##  $ alzheimers       : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ arthritis        : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ cancer           : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ copd             : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ depression       : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ diabetes         : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ heart.failure    : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ ihd              : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ kidney           : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ osteoporosis     : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ stroke           : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ reimbursement2008: int  0 0 0 0 0 0 0 0 0 0 ...
##  $ bucket2008       : int  1 1 1 1 1 1 1 1 1 1 ...
##  $ reimbursement2009: int  0 0 0 0 0 0 0 0 0 0 ...
##  $ bucket2009       : int  1 1 1 1 1 1 1 1 1 1 ...
##  - attr(*, "comment")= chr "glb_entity_df"
```

```r
if (!is.null(glb_max_obs)) {
    warning("glb_<ent>_df restricted to glb_max_obs: ", format(glb_max_obs, big.mark=","))
    glb_entity_df <- glb_entity_df[split <- 
        sample.split(glb_entity_df[, glb_rsp_var_raw], SplitRatio=glb_max_obs), ]
    glb_newent_df <- glb_newent_df[split <- 
        sample.split(glb_newent_df[, glb_rsp_var_raw], SplitRatio=glb_max_obs), ]
}
```

```
## Warning: glb_<ent>_df restricted to glb_max_obs: 1,000
```

```r
glb_script_df <- rbind(glb_script_df,
                   data.frame(chunk_label="cleanse_data", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##           chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed   import_data                1                0   0.003
## elapsed1 cleanse_data                2                0   7.395
```

## Step `2`: cleanse data

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="inspectORexplore.data", 
                              chunk_step_major=max(glb_script_df$chunk_step_major), 
                              chunk_step_minor=1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                    chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed1          cleanse_data                2                0   7.395
## elapsed2 inspectORexplore.data                2                1   7.429
```

### Step `2`.`1`: inspect/explore data

```r
#print(str(glb_entity_df))
#View(glb_entity_df)

# List info gathered for various columns
# <col_name>:   <description>; <notes>

# Create new features that help diagnostics
#   Create factors of string variables
str_vars <- sapply(1:length(names(glb_entity_df)), 
    function(col) ifelse(class(glb_entity_df[, names(glb_entity_df)[col]]) == "character",
                         names(glb_entity_df)[col], ""))
if (length(str_vars <- setdiff(str_vars[str_vars != ""], 
                               glb_exclude_vars_as_features)) > 0) {
    warning("Creating factors of string variables:", paste0(str_vars, collapse=", "))
    glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, str_vars)
    for (var in str_vars) {
        glb_entity_df[, paste0(var, ".fctr")] <- factor(glb_entity_df[, var], 
                        as.factor(union(glb_entity_df[, var], glb_newent_df[, var])))
        glb_newent_df[, paste0(var, ".fctr")] <- factor(glb_newent_df[, var], 
                        as.factor(union(glb_entity_df[, var], glb_newent_df[, var])))
    }
}

#   Convert factors to dummy variables
#   Build splines   require(splines); bsBasis <- bs(training$age, df=3)

add_new_diag_feats <- function(obs_df, obs_twin_df) {
    require(plyr)
    
    obs_df <- mutate(obs_df,
#         <col_name>.NA=is.na(<col_name>),

#         <col_name>.fctr=factor(<col_name>, 
#                     as.factor(union(obs_df$<col_name>, obs_twin_df$<col_name>))), 
#         <col_name>.fctr=relevel(factor(<col_name>, 
#                     as.factor(union(obs_df$<col_name>, obs_twin_df$<col_name>))),
#                                   "<ref_val>"), 
#         <col2_name>.fctr=relevel(factor(ifelse(<col1_name> == <val>, "<oth_val>", "<ref_val>")), 
#                               as.factor(c("R", "<ref_val>")),
#                               ref="<ref_val>"),

          # This doesn't work - use sapply instead
#         <col_name>.fctr_num=grep(<col_name>, levels(<col_name>.fctr)), 
#         
#         Date.my=as.Date(strptime(Date, "%m/%d/%y %H:%M")),
#         Year=year(Date.my),
#         Month=months(Date.my),
#         Weekday=weekdays(Date.my)

#         <col_name>.log=log(<col.name>),        
#         <col_name>=<table>[as.character(<col2_name>)],
#         <col_name>=as.numeric(<col2_name>),

        .rnorm=rnorm(n=nrow(obs_df))
                        )

    # If levels of a factor are different across obs_df & glb_newent_df; predict.glm fails  
    # Transformations not handled by mutate
#     obs_df$<col_name>.fctr.num <- sapply(1:nrow(obs_df), 
#         function(row_ix) grep(obs_df[row_ix, "<col_name>"],
#                               levels(obs_df[row_ix, "<col_name>.fctr"])))
    
    print(summary(obs_df))
    print(sapply(names(obs_df), function(col) sum(is.na(obs_df[, col]))))
    return(obs_df)
}

glb_entity_df <- add_new_diag_feats(glb_entity_df, glb_newent_df)
```

```
## Loading required package: plyr
```

```
##       age           alzheimers      arthritis         cancer     
##  Min.   : 26.00   Min.   :0.000   Min.   :0.000   Min.   :0.000  
##  1st Qu.: 67.75   1st Qu.:0.000   1st Qu.:0.000   1st Qu.:0.000  
##  Median : 73.00   Median :0.000   Median :0.000   Median :0.000  
##  Mean   : 72.53   Mean   :0.196   Mean   :0.144   Mean   :0.058  
##  3rd Qu.: 80.00   3rd Qu.:0.000   3rd Qu.:0.000   3rd Qu.:0.000  
##  Max.   :100.00   Max.   :1.000   Max.   :1.000   Max.   :1.000  
##       copd         depression       diabetes     heart.failure  
##  Min.   :0.000   Min.   :0.000   Min.   :0.000   Min.   :0.000  
##  1st Qu.:0.000   1st Qu.:0.000   1st Qu.:0.000   1st Qu.:0.000  
##  Median :0.000   Median :0.000   Median :0.000   Median :0.000  
##  Mean   :0.126   Mean   :0.205   Mean   :0.371   Mean   :0.286  
##  3rd Qu.:0.000   3rd Qu.:0.000   3rd Qu.:1.000   3rd Qu.:1.000  
##  Max.   :1.000   Max.   :1.000   Max.   :1.000   Max.   :1.000  
##       ihd            kidney       osteoporosis       stroke     
##  Min.   :0.000   Min.   :0.000   Min.   :0.000   Min.   :0.000  
##  1st Qu.:0.000   1st Qu.:0.000   1st Qu.:0.000   1st Qu.:0.000  
##  Median :0.000   Median :0.000   Median :0.000   Median :0.000  
##  Mean   :0.428   Mean   :0.182   Mean   :0.178   Mean   :0.036  
##  3rd Qu.:1.000   3rd Qu.:0.000   3rd Qu.:0.000   3rd Qu.:0.000  
##  Max.   :1.000   Max.   :1.000   Max.   :1.000   Max.   :1.000  
##  reimbursement2008   bucket2008    reimbursement2009    bucket2009   
##  Min.   :    0     Min.   :1.000   Min.   :     0.0   Min.   :1.000  
##  1st Qu.:    0     1st Qu.:1.000   1st Qu.:   197.5   1st Qu.:1.000  
##  Median :  945     Median :1.000   Median :  1525.0   Median :1.000  
##  Mean   : 4159     Mean   :1.445   Mean   :  4444.6   Mean   :1.521  
##  3rd Qu.: 3182     3rd Qu.:2.000   3rd Qu.:  4155.0   3rd Qu.:2.000  
##  Max.   :81390     Max.   :5.000   Max.   :131960.0   Max.   :5.000  
##      .rnorm         
##  Min.   :-3.323274  
##  1st Qu.:-0.624792  
##  Median : 0.008896  
##  Mean   : 0.015698  
##  3rd Qu.: 0.674460  
##  Max.   : 3.094892  
##               age        alzheimers         arthritis            cancer 
##                 0                 0                 0                 0 
##              copd        depression          diabetes     heart.failure 
##                 0                 0                 0                 0 
##               ihd            kidney      osteoporosis            stroke 
##                 0                 0                 0                 0 
## reimbursement2008        bucket2008 reimbursement2009        bucket2009 
##                 0                 0                 0                 0 
##            .rnorm 
##                 0
```

```r
glb_newent_df <- add_new_diag_feats(glb_newent_df, glb_entity_df)
```

```
##       age           alzheimers      arthritis         cancer     
##  Min.   : 26.00   Min.   :0.000   Min.   :0.000   Min.   :0.000  
##  1st Qu.: 67.00   1st Qu.:0.000   1st Qu.:0.000   1st Qu.:0.000  
##  Median : 73.00   Median :0.000   Median :0.000   Median :0.000  
##  Mean   : 72.71   Mean   :0.188   Mean   :0.135   Mean   :0.072  
##  3rd Qu.: 81.00   3rd Qu.:0.000   3rd Qu.:0.000   3rd Qu.:0.000  
##  Max.   :100.00   Max.   :1.000   Max.   :1.000   Max.   :1.000  
##       copd         depression       diabetes     heart.failure  
##  Min.   :0.000   Min.   :0.000   Min.   :0.000   Min.   :0.000  
##  1st Qu.:0.000   1st Qu.:0.000   1st Qu.:0.000   1st Qu.:0.000  
##  Median :0.000   Median :0.000   Median :0.000   Median :0.000  
##  Mean   :0.123   Mean   :0.219   Mean   :0.365   Mean   :0.272  
##  3rd Qu.:0.000   3rd Qu.:0.000   3rd Qu.:1.000   3rd Qu.:1.000  
##  Max.   :1.000   Max.   :1.000   Max.   :1.000   Max.   :1.000  
##       ihd            kidney       osteoporosis       stroke     
##  Min.   :0.000   Min.   :0.000   Min.   :0.000   Min.   :0.000  
##  1st Qu.:0.000   1st Qu.:0.000   1st Qu.:0.000   1st Qu.:0.000  
##  Median :0.000   Median :0.000   Median :0.000   Median :0.000  
##  Mean   :0.403   Mean   :0.159   Mean   :0.176   Mean   :0.044  
##  3rd Qu.:1.000   3rd Qu.:0.000   3rd Qu.:0.000   3rd Qu.:0.000  
##  Max.   :1.000   Max.   :1.000   Max.   :1.000   Max.   :1.000  
##  reimbursement2008   bucket2008    reimbursement2009   bucket2009   
##  Min.   :     0    Min.   :1.000   Min.   :    0.0   Min.   :1.000  
##  1st Qu.:     0    1st Qu.:1.000   1st Qu.:  187.5   1st Qu.:1.000  
##  Median :   880    Median :1.000   Median : 1540.0   Median :1.000  
##  Mean   :  3871    Mean   :1.409   Mean   : 4197.2   Mean   :1.521  
##  3rd Qu.:  2925    3rd Qu.:1.000   3rd Qu.: 4067.5   3rd Qu.:2.000  
##  Max.   :128070    Max.   :5.000   Max.   :61330.0   Max.   :5.000  
##      .rnorm        
##  Min.   :-3.08467  
##  1st Qu.:-0.77172  
##  Median :-0.04295  
##  Mean   :-0.01916  
##  3rd Qu.: 0.68672  
##  Max.   : 2.98747  
##               age        alzheimers         arthritis            cancer 
##                 0                 0                 0                 0 
##              copd        depression          diabetes     heart.failure 
##                 0                 0                 0                 0 
##               ihd            kidney      osteoporosis            stroke 
##                 0                 0                 0                 0 
## reimbursement2008        bucket2008 reimbursement2009        bucket2009 
##                 0                 0                 0                 0 
##            .rnorm 
##                 0
```

```r
# Histogram of predictor in glb_entity_df & glb_newent_df
print(myplot_histogram(glb_entity_df, glb_rsp_var_raw))
```

```
## stat_bin: binwidth defaulted to range/30. Use 'binwidth = x' to adjust this.
```

![](Medicare_2008_09_files/figure-html/inspectORexplore_data-1.png) 

```r
if (glb_is_classification)
    print(table(glb_entity_df[, glb_rsp_var_raw]) / nrow(glb_entity_df))
```

```
## 
##     1     2     3     4     5 
## 0.672 0.190 0.089 0.043 0.006
```

```r
print(myplot_histogram(glb_newent_df, glb_rsp_var_raw))
```

```
## stat_bin: binwidth defaulted to range/30. Use 'binwidth = x' to adjust this.
```

![](Medicare_2008_09_files/figure-html/inspectORexplore_data-2.png) 

```r
# Check for duplicates in glb_id_vars
if (length(glb_id_vars) > 0) {
    id_vars_dups_df <- subset(id_vars_df <- mycreate_tbl_df(
        rbind(glb_entity_df[, glb_id_vars, FALSE], glb_newent_df[, glb_id_vars, FALSE]),
            glb_id_vars), .freq > 1)
    if (nrow(id_vars_dups_df) > 0) {
        warning("Duplicates found in glb_id_vars data:", nrow(id_vars_dups_df))
        myprint_df(id_vars_dups_df)
    } else {
        # glb_id_vars are unique across obs in both glb_<>_df
        glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, glb_id_vars)
    }
}

#pairs(subset(glb_entity_df, select=-c(col_symbol)))
# Check for glb_newent_df & glb_entity_df features range mismatches

# Other diagnostics:
# print(subset(glb_entity_df, <col1_name> == max(glb_entity_df$<col1_name>, na.rm=TRUE) & 
#                         <col2_name> <= mean(glb_entity_df$<col1_name>, na.rm=TRUE)))

# print(glb_entity_df[which.max(glb_entity_df$<col_name>),])

# print(<col_name>_freq_glb_entity_df <- mycreate_tbl_df(glb_entity_df, "<col_name>"))
# print(which.min(table(glb_entity_df$<col_name>)))
# print(which.max(table(glb_entity_df$<col_name>)))
# print(which.max(table(glb_entity_df$<col1_name>, glb_entity_df$<col2_name>)[, 2]))
# print(table(glb_entity_df$<col1_name>, glb_entity_df$<col2_name>))
# print(table(is.na(glb_entity_df$<col1_name>), glb_entity_df$<col2_name>))
# print(table(sign(glb_entity_df$<col1_name>), glb_entity_df$<col2_name>))
# print(mycreate_xtab(glb_entity_df, <col1_name>))
# print(mycreate_xtab(glb_entity_df, c(<col1_name>, <col2_name>)))
# print(<col1_name>_<col2_name>_xtab_glb_entity_df <- 
#   mycreate_xtab(glb_entity_df, c("<col1_name>", "<col2_name>")))
# <col1_name>_<col2_name>_xtab_glb_entity_df[is.na(<col1_name>_<col2_name>_xtab_glb_entity_df)] <- 0
# print(<col1_name>_<col2_name>_xtab_glb_entity_df <- 
#   mutate(<col1_name>_<col2_name>_xtab_glb_entity_df, 
#             <col3_name>=(<col1_name> * 1.0) / (<col1_name> + <col2_name>))) 

# print(<col2_name>_min_entity_arr <- 
#    sort(tapply(glb_entity_df$<col1_name>, glb_entity_df$<col2_name>, min, na.rm=TRUE)))
# print(<col1_name>_na_by_<col2_name>_arr <- 
#    sort(tapply(glb_entity_df$<col1_name>.NA, glb_entity_df$<col2_name>, mean, na.rm=TRUE)))

# Other plots:
# print(myplot_box(df=glb_entity_df, ycol_names="<col1_name>"))
# print(myplot_box(df=glb_entity_df, ycol_names="<col1_name>", xcol_name="<col2_name>"))
# print(myplot_line(subset(glb_entity_df, Symbol %in% c("KO", "PG")), 
#                   "Date.my", "StockPrice", facet_row_colnames="Symbol") + 
#     geom_vline(xintercept=as.numeric(as.Date("2003-03-01"))) +
#     geom_vline(xintercept=as.numeric(as.Date("1983-01-01")))        
#         )
# print(myplot_scatter(glb_entity_df, "<col1_name>", "<col2_name>", smooth=TRUE))
# print(myplot_scatter(glb_entity_df, "<col1_name>", "<col2_name>", colorcol_name="<Pred.fctr>"))

glb_script_df <- rbind(glb_script_df, 
    data.frame(chunk_label="manage_missing_data", 
        chunk_step_major=max(glb_script_df$chunk_step_major), 
        chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                    chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed2 inspectORexplore.data                2                1   7.429
## elapsed3   manage_missing_data                2                2   8.898
```

### Step `2`.`2`: manage missing data

```r
# print(sapply(names(glb_entity_df), function(col) sum(is.na(glb_entity_df[, col]))))
# print(sapply(names(glb_newent_df), function(col) sum(is.na(glb_newent_df[, col]))))
# glb_entity_df <- na.omit(glb_entity_df)
# glb_newent_df <- na.omit(glb_newent_df)
# df[is.na(df)] <- 0

# Not refactored into mydsutils.R since glb_*_df might be reassigned
glb_impute_missing_data <- function(entity_df, newent_df) {
    if (!glb_is_separate_newent_dataset) {
        # Combine entity & newent
        union_df <- rbind(mutate(entity_df, .src = "entity"),
                          mutate(newent_df, .src = "newent"))
        union_imputed_df <- union_df[, setdiff(setdiff(names(entity_df), 
                                                       glb_rsp_var), 
                                               glb_exclude_vars_as_features)]
        print(summary(union_imputed_df))
    
        require(mice)
        set.seed(glb_mice_complete.seed)
        union_imputed_df <- complete(mice(union_imputed_df))
        print(summary(union_imputed_df))
    
        union_df[, names(union_imputed_df)] <- union_imputed_df[, names(union_imputed_df)]
        print(summary(union_df))
#         union_df$.rownames <- rownames(union_df)
#         union_df <- orderBy(~.rownames, union_df)
#         
#         imp_entity_df <- myimport_data(
#             url="<imputed_trnng_url>", 
#             comment="imp_entity_df", force_header=TRUE, print_diagn=TRUE)
#         print(all.equal(subset(union_df, select=-c(.src, .rownames, .rnorm)), 
#                         imp_entity_df))
        
        # Partition again
        glb_entity_df <<- subset(union_df, .src == "entity", select=-c(.src, .rownames))
        comment(glb_entity_df) <- "entity_df"
        glb_newent_df <<- subset(union_df, .src == "newent", select=-c(.src, .rownames))
        comment(glb_newent_df) <- "newent_df"
        
        # Generate summaries
        print(summary(entity_df))
        print(sapply(names(entity_df), function(col) sum(is.na(entity_df[, col]))))
        print(summary(newent_df))
        print(sapply(names(newent_df), function(col) sum(is.na(newent_df[, col]))))
    
    } else stop("Not implemented yet")
}

if (glb_impute_na_data) {
    if ((sum(sapply(names(glb_entity_df), 
                    function(col) sum(is.na(glb_entity_df[, col])))) > 0) | 
        (sum(sapply(names(glb_newent_df), 
                    function(col) sum(is.na(glb_newent_df[, col])))) > 0))
        glb_impute_missing_data(glb_entity_df, glb_newent_df)
}    

glb_script_df <- rbind(glb_script_df, 
    data.frame(chunk_label="encode_retype_data", 
        chunk_step_major=max(glb_script_df$chunk_step_major), 
        chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                  chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed3 manage_missing_data                2                2   8.898
## elapsed4  encode_retype_data                2                3   9.336
```

### Step `2`.`3`: encode/retype data

```r
# map_<col_name>_df <- myimport_data(
#     url="<map_url>", 
#     comment="map_<col_name>_df", print_diagn=TRUE)
# map_<col_name>_df <- read.csv(paste0(getwd(), "/data/<file_name>.csv"), strip.white=TRUE)

# glb_entity_df <- mymap_codes(glb_entity_df, "<from_col_name>", "<to_col_name>", 
#     map_<to_col_name>_df, map_join_col_name="<map_join_col_name>", 
#                           map_tgt_col_name="<to_col_name>")
# glb_newent_df <- mymap_codes(glb_newent_df, "<from_col_name>", "<to_col_name>", 
#     map_<to_col_name>_df, map_join_col_name="<map_join_col_name>", 
#                           map_tgt_col_name="<to_col_name>")
    					
# glb_entity_df$<col_name>.fctr <- factor(glb_entity_df$<col_name>, 
#                     as.factor(union(glb_entity_df$<col_name>, glb_newent_df$<col_name>)))
# glb_newent_df$<col_name>.fctr <- factor(glb_newent_df$<col_name>, 
#                     as.factor(union(glb_entity_df$<col_name>, glb_newent_df$<col_name>)))

if (!is.null(glb_map_rsp_raw_to_var)) {
    glb_entity_df[, glb_rsp_var] <- 
        glb_map_rsp_raw_to_var(glb_entity_df[, glb_rsp_var_raw])
    mycheck_map_results(mapd_df=glb_entity_df, 
                        from_col_name=glb_rsp_var_raw, to_col_name=glb_rsp_var)
        
    glb_newent_df[, glb_rsp_var] <- 
        glb_map_rsp_raw_to_var(glb_newent_df[, glb_rsp_var_raw])
    mycheck_map_results(mapd_df=glb_newent_df, 
                        from_col_name=glb_rsp_var_raw, to_col_name=glb_rsp_var)    
}
```

```
## Loading required package: sqldf
## Loading required package: gsubfn
## Loading required package: proto
## Loading required package: RSQLite
## Loading required package: DBI
## Loading required package: tcltk
```

```
##   bucket2009 bucket2009.fctr  .n
## 1          1              B1 672
## 2          2              B2 190
## 3          3              B3  89
## 4          4              B4  43
## 5          5              B5   6
```

![](Medicare_2008_09_files/figure-html/encode_retype_data_1-1.png) 

```
##   bucket2009 bucket2009.fctr  .n
## 1          1              B1 672
## 2          2              B2 190
## 3          3              B3  89
## 4          4              B4  43
## 5          5              B5   6
```

![](Medicare_2008_09_files/figure-html/encode_retype_data_1-2.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="extract_features", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                 chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed4 encode_retype_data                2                3   9.336
## elapsed5   extract_features                3                0  13.454
```

## Step `3`: extract features

```r
# Create new features that help prediction
# <col_name>.lag.2 <- lag(zoo(glb_entity_df$<col_name>), -2, na.pad=TRUE)
# glb_entity_df[, "<col_name>.lag.2"] <- coredata(<col_name>.lag.2)
# <col_name>.lag.2 <- lag(zoo(glb_newent_df$<col_name>), -2, na.pad=TRUE)
# glb_newent_df[, "<col_name>.lag.2"] <- coredata(<col_name>.lag.2)
# 
# glb_newent_df[1, "<col_name>.lag.2"] <- glb_entity_df[nrow(glb_entity_df) - 1, 
#                                                    "<col_name>"]
# glb_newent_df[2, "<col_name>.lag.2"] <- glb_entity_df[nrow(glb_entity_df), 
#                                                    "<col_name>"]
                                                   
# glb_entity_df <- mutate(glb_entity_df,
#     <new_col_name>=
#                     )

# glb_newent_df <- mutate(glb_newent_df,
#     <new_col_name>=
#                     )

# print(summary(glb_entity_df))
# print(summary(glb_newent_df))

# print(sapply(names(glb_entity_df), function(col) sum(is.na(glb_entity_df[, col]))))
# print(sapply(names(glb_newent_df), function(col) sum(is.na(glb_newent_df[, col]))))

# print(myplot_scatter(glb_entity_df, "<col1_name>", "<col2_name>", smooth=TRUE))

replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "data.training.all","data.new")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0
```

![](Medicare_2008_09_files/figure-html/extract_features-1.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="select_features", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##               chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed5 extract_features                3                0  13.454
## elapsed6  select_features                4                0  15.069
```

## Step `4`: select features

```r
print(glb_feats_df <- 
    myselect_features( entity_df=glb_entity_df, 
                       exclude_vars_as_features=glb_exclude_vars_as_features, 
                       rsp_var=glb_rsp_var))
```

```
##                                  id      cor.y  cor.y.abs
## bucket2008               bucket2008 0.44655074 0.44655074
## ihd                             ihd 0.41799716 0.41799716
## diabetes                   diabetes 0.40363086 0.40363086
## reimbursement2008 reimbursement2008 0.39864605 0.39864605
## kidney                       kidney 0.38762968 0.38762968
## heart.failure         heart.failure 0.36864613 0.36864613
## copd                           copd 0.33705795 0.33705795
## alzheimers               alzheimers 0.29777179 0.29777179
## depression               depression 0.27092271 0.27092271
## arthritis                 arthritis 0.25517293 0.25517293
## stroke                       stroke 0.19269419 0.19269419
## osteoporosis           osteoporosis 0.18701176 0.18701176
## cancer                       cancer 0.18571084 0.18571084
## age                             age 0.03524295 0.03524295
```

```r
glb_script_df <- rbind(glb_script_df, 
    data.frame(chunk_label="remove_correlated_features", 
        chunk_step_major=max(glb_script_df$chunk_step_major),
        chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))        
print(tail(glb_script_df, 2))
```

```
##                         chunk_label chunk_step_major chunk_step_minor
## elapsed6            select_features                4                0
## elapsed7 remove_correlated_features                4                1
##          elapsed
## elapsed6  15.069
## elapsed7  15.272
```

### Step `4`.`1`: remove correlated features

```r
print(glb_feats_df <- orderBy(~-cor.y, merge(glb_feats_df, 
          mydelete_cor_features(glb_feats_df, glb_entity_df, glb_rsp_var, 
                                glb_exclude_vars_as_features), 
          all.x=TRUE)))
```

```
## Loading required package: reshape2
```

```
##                   bucket2008        ihd   diabetes reimbursement2008
## bucket2008        1.00000000 0.46958531 0.45126346       0.878312334
## ihd               0.46958531 1.00000000 0.49457184       0.378957172
## diabetes          0.45126346 0.49457184 1.00000000       0.366199213
## reimbursement2008 0.87831233 0.37895717 0.36619921       1.000000000
## kidney            0.55056505 0.42482496 0.46396051       0.453562122
## heart.failure     0.49955978 0.48120259 0.43012377       0.416351106
## copd              0.49268177 0.34149564 0.32596148       0.422330000
## alzheimers        0.39140169 0.37732239 0.37694140       0.323243697
## depression        0.32576952 0.30168154 0.34840528       0.286603171
## arthritis         0.33623054 0.31872843 0.30410058       0.264603789
## stroke            0.30476328 0.18000687 0.17383842       0.299281902
## osteoporosis      0.18349678 0.27906135 0.26497260       0.125133629
## cancer            0.32652869 0.20039143 0.16368061       0.327646692
## age               0.03413982 0.04706689 0.01841107      -0.001423105
##                       kidney heart.failure       copd alzheimers
## bucket2008        0.55056505    0.49955978 0.49268177 0.39140169
## ihd               0.42482496    0.48120259 0.34149564 0.37732239
## diabetes          0.46396051    0.43012377 0.32596148 0.37694140
## reimbursement2008 0.45356212    0.41635111 0.42233000 0.32324370
## kidney            1.00000000    0.45852512 0.35978746 0.37428152
## heart.failure     0.45852512    1.00000000 0.39986877 0.33973694
## copd              0.35978746    0.39986877 1.00000000 0.24522190
## alzheimers        0.37428152    0.33973694 0.24522190 1.00000000
## depression        0.25480518    0.27610822 0.24759621 0.27343608
## arthritis         0.21992226    0.26987077 0.22192295 0.19211953
## stroke            0.21491669    0.19842601 0.15308900 0.18855652
## osteoporosis      0.14637804    0.22615691 0.17782104 0.17196425
## cancer            0.13797748    0.15537815 0.17651697 0.11458281
## age               0.03810528    0.02266969 0.02910471 0.01048956
##                    depression  arthritis      stroke osteoporosis
## bucket2008         0.32576952 0.33623054  0.30476328   0.18349678
## ihd                0.30168154 0.31872843  0.18000687   0.27906135
## diabetes           0.34840528 0.30410058  0.17383842   0.26497260
## reimbursement2008  0.28660317 0.26460379  0.29928190   0.12513363
## kidney             0.25480518 0.21992226  0.21491669   0.14637804
## heart.failure      0.27610822 0.26987077  0.19842601   0.22615691
## copd               0.24759621 0.22192295  0.15308900   0.17782104
## alzheimers         0.27343608 0.19211953  0.18855652   0.17196425
## depression         1.00000000 0.20799307  0.12791598   0.17814933
## arthritis          0.20799307 1.00000000  0.10421295   0.21123442
## stroke             0.12791598 0.10421295  1.00000000   0.07847503
## osteoporosis       0.17814933 0.21123442  0.07847503   1.00000000
## cancer             0.11773731 0.10537998 -0.00202094   0.04111404
## age               -0.01566336 0.02943173  0.08329630  -0.03457520
##                        cancer          age
## bucket2008         0.32652869  0.034139822
## ihd                0.20039143  0.047066887
## diabetes           0.16368061  0.018411075
## reimbursement2008  0.32764669 -0.001423105
## kidney             0.13797748  0.038105278
## heart.failure      0.15537815  0.022669688
## copd               0.17651697  0.029104714
## alzheimers         0.11458281  0.010489562
## depression         0.11773731 -0.015663360
## arthritis          0.10537998  0.029431725
## stroke            -0.00202094  0.083296298
## osteoporosis       0.04111404 -0.034575197
## cancer             1.00000000 -0.012831380
## age               -0.01283138  1.000000000
##                   bucket2008        ihd   diabetes reimbursement2008
## bucket2008        0.00000000 0.46958531 0.45126346       0.878312334
## ihd               0.46958531 0.00000000 0.49457184       0.378957172
## diabetes          0.45126346 0.49457184 0.00000000       0.366199213
## reimbursement2008 0.87831233 0.37895717 0.36619921       0.000000000
## kidney            0.55056505 0.42482496 0.46396051       0.453562122
## heart.failure     0.49955978 0.48120259 0.43012377       0.416351106
## copd              0.49268177 0.34149564 0.32596148       0.422330000
## alzheimers        0.39140169 0.37732239 0.37694140       0.323243697
## depression        0.32576952 0.30168154 0.34840528       0.286603171
## arthritis         0.33623054 0.31872843 0.30410058       0.264603789
## stroke            0.30476328 0.18000687 0.17383842       0.299281902
## osteoporosis      0.18349678 0.27906135 0.26497260       0.125133629
## cancer            0.32652869 0.20039143 0.16368061       0.327646692
## age               0.03413982 0.04706689 0.01841107       0.001423105
##                       kidney heart.failure       copd alzheimers
## bucket2008        0.55056505    0.49955978 0.49268177 0.39140169
## ihd               0.42482496    0.48120259 0.34149564 0.37732239
## diabetes          0.46396051    0.43012377 0.32596148 0.37694140
## reimbursement2008 0.45356212    0.41635111 0.42233000 0.32324370
## kidney            0.00000000    0.45852512 0.35978746 0.37428152
## heart.failure     0.45852512    0.00000000 0.39986877 0.33973694
## copd              0.35978746    0.39986877 0.00000000 0.24522190
## alzheimers        0.37428152    0.33973694 0.24522190 0.00000000
## depression        0.25480518    0.27610822 0.24759621 0.27343608
## arthritis         0.21992226    0.26987077 0.22192295 0.19211953
## stroke            0.21491669    0.19842601 0.15308900 0.18855652
## osteoporosis      0.14637804    0.22615691 0.17782104 0.17196425
## cancer            0.13797748    0.15537815 0.17651697 0.11458281
## age               0.03810528    0.02266969 0.02910471 0.01048956
##                   depression  arthritis     stroke osteoporosis     cancer
## bucket2008        0.32576952 0.33623054 0.30476328   0.18349678 0.32652869
## ihd               0.30168154 0.31872843 0.18000687   0.27906135 0.20039143
## diabetes          0.34840528 0.30410058 0.17383842   0.26497260 0.16368061
## reimbursement2008 0.28660317 0.26460379 0.29928190   0.12513363 0.32764669
## kidney            0.25480518 0.21992226 0.21491669   0.14637804 0.13797748
## heart.failure     0.27610822 0.26987077 0.19842601   0.22615691 0.15537815
## copd              0.24759621 0.22192295 0.15308900   0.17782104 0.17651697
## alzheimers        0.27343608 0.19211953 0.18855652   0.17196425 0.11458281
## depression        0.00000000 0.20799307 0.12791598   0.17814933 0.11773731
## arthritis         0.20799307 0.00000000 0.10421295   0.21123442 0.10537998
## stroke            0.12791598 0.10421295 0.00000000   0.07847503 0.00202094
## osteoporosis      0.17814933 0.21123442 0.07847503   0.00000000 0.04111404
## cancer            0.11773731 0.10537998 0.00202094   0.04111404 0.00000000
## age               0.01566336 0.02943173 0.08329630   0.03457520 0.01283138
##                           age
## bucket2008        0.034139822
## ihd               0.047066887
## diabetes          0.018411075
## reimbursement2008 0.001423105
## kidney            0.038105278
## heart.failure     0.022669688
## copd              0.029104714
## alzheimers        0.010489562
## depression        0.015663360
## arthritis         0.029431725
## stroke            0.083296298
## osteoporosis      0.034575197
## cancer            0.012831380
## age               0.000000000
## [1] "cor(bucket2008, reimbursement2008)=0.8783"
```

![](Medicare_2008_09_files/figure-html/remove_correlated_features-1.png) 

```
## [1] "cor(bucket2009.fctr, bucket2008)=0.4466"
## [1] "cor(bucket2009.fctr, reimbursement2008)=0.3986"
```

```
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
```

```
## Warning in mydelete_cor_features(glb_feats_df, glb_entity_df, glb_rsp_var,
## : Dropping reimbursement2008 as a feature
```

![](Medicare_2008_09_files/figure-html/remove_correlated_features-2.png) 

```
##                          id      cor.y  cor.y.abs
## bucket2008       bucket2008 0.44655074 0.44655074
## ihd                     ihd 0.41799716 0.41799716
## diabetes           diabetes 0.40363086 0.40363086
## kidney               kidney 0.38762968 0.38762968
## heart.failure heart.failure 0.36864613 0.36864613
## copd                   copd 0.33705795 0.33705795
## alzheimers       alzheimers 0.29777179 0.29777179
## depression       depression 0.27092271 0.27092271
## arthritis         arthritis 0.25517293 0.25517293
## stroke               stroke 0.19269419 0.19269419
## osteoporosis   osteoporosis 0.18701176 0.18701176
## cancer               cancer 0.18571084 0.18571084
## age                     age 0.03524295 0.03524295
##               bucket2008        ihd   diabetes     kidney heart.failure
## bucket2008    1.00000000 0.46958531 0.45126346 0.55056505    0.49955978
## ihd           0.46958531 1.00000000 0.49457184 0.42482496    0.48120259
## diabetes      0.45126346 0.49457184 1.00000000 0.46396051    0.43012377
## kidney        0.55056505 0.42482496 0.46396051 1.00000000    0.45852512
## heart.failure 0.49955978 0.48120259 0.43012377 0.45852512    1.00000000
## copd          0.49268177 0.34149564 0.32596148 0.35978746    0.39986877
## alzheimers    0.39140169 0.37732239 0.37694140 0.37428152    0.33973694
## depression    0.32576952 0.30168154 0.34840528 0.25480518    0.27610822
## arthritis     0.33623054 0.31872843 0.30410058 0.21992226    0.26987077
## stroke        0.30476328 0.18000687 0.17383842 0.21491669    0.19842601
## osteoporosis  0.18349678 0.27906135 0.26497260 0.14637804    0.22615691
## cancer        0.32652869 0.20039143 0.16368061 0.13797748    0.15537815
## age           0.03413982 0.04706689 0.01841107 0.03810528    0.02266969
##                     copd alzheimers  depression  arthritis      stroke
## bucket2008    0.49268177 0.39140169  0.32576952 0.33623054  0.30476328
## ihd           0.34149564 0.37732239  0.30168154 0.31872843  0.18000687
## diabetes      0.32596148 0.37694140  0.34840528 0.30410058  0.17383842
## kidney        0.35978746 0.37428152  0.25480518 0.21992226  0.21491669
## heart.failure 0.39986877 0.33973694  0.27610822 0.26987077  0.19842601
## copd          1.00000000 0.24522190  0.24759621 0.22192295  0.15308900
## alzheimers    0.24522190 1.00000000  0.27343608 0.19211953  0.18855652
## depression    0.24759621 0.27343608  1.00000000 0.20799307  0.12791598
## arthritis     0.22192295 0.19211953  0.20799307 1.00000000  0.10421295
## stroke        0.15308900 0.18855652  0.12791598 0.10421295  1.00000000
## osteoporosis  0.17782104 0.17196425  0.17814933 0.21123442  0.07847503
## cancer        0.17651697 0.11458281  0.11773731 0.10537998 -0.00202094
## age           0.02910471 0.01048956 -0.01566336 0.02943173  0.08329630
##               osteoporosis      cancer         age
## bucket2008      0.18349678  0.32652869  0.03413982
## ihd             0.27906135  0.20039143  0.04706689
## diabetes        0.26497260  0.16368061  0.01841107
## kidney          0.14637804  0.13797748  0.03810528
## heart.failure   0.22615691  0.15537815  0.02266969
## copd            0.17782104  0.17651697  0.02910471
## alzheimers      0.17196425  0.11458281  0.01048956
## depression      0.17814933  0.11773731 -0.01566336
## arthritis       0.21123442  0.10537998  0.02943173
## stroke          0.07847503 -0.00202094  0.08329630
## osteoporosis    1.00000000  0.04111404 -0.03457520
## cancer          0.04111404  1.00000000 -0.01283138
## age            -0.03457520 -0.01283138  1.00000000
##               bucket2008        ihd   diabetes     kidney heart.failure
## bucket2008    0.00000000 0.46958531 0.45126346 0.55056505    0.49955978
## ihd           0.46958531 0.00000000 0.49457184 0.42482496    0.48120259
## diabetes      0.45126346 0.49457184 0.00000000 0.46396051    0.43012377
## kidney        0.55056505 0.42482496 0.46396051 0.00000000    0.45852512
## heart.failure 0.49955978 0.48120259 0.43012377 0.45852512    0.00000000
## copd          0.49268177 0.34149564 0.32596148 0.35978746    0.39986877
## alzheimers    0.39140169 0.37732239 0.37694140 0.37428152    0.33973694
## depression    0.32576952 0.30168154 0.34840528 0.25480518    0.27610822
## arthritis     0.33623054 0.31872843 0.30410058 0.21992226    0.26987077
## stroke        0.30476328 0.18000687 0.17383842 0.21491669    0.19842601
## osteoporosis  0.18349678 0.27906135 0.26497260 0.14637804    0.22615691
## cancer        0.32652869 0.20039143 0.16368061 0.13797748    0.15537815
## age           0.03413982 0.04706689 0.01841107 0.03810528    0.02266969
##                     copd alzheimers depression  arthritis     stroke
## bucket2008    0.49268177 0.39140169 0.32576952 0.33623054 0.30476328
## ihd           0.34149564 0.37732239 0.30168154 0.31872843 0.18000687
## diabetes      0.32596148 0.37694140 0.34840528 0.30410058 0.17383842
## kidney        0.35978746 0.37428152 0.25480518 0.21992226 0.21491669
## heart.failure 0.39986877 0.33973694 0.27610822 0.26987077 0.19842601
## copd          0.00000000 0.24522190 0.24759621 0.22192295 0.15308900
## alzheimers    0.24522190 0.00000000 0.27343608 0.19211953 0.18855652
## depression    0.24759621 0.27343608 0.00000000 0.20799307 0.12791598
## arthritis     0.22192295 0.19211953 0.20799307 0.00000000 0.10421295
## stroke        0.15308900 0.18855652 0.12791598 0.10421295 0.00000000
## osteoporosis  0.17782104 0.17196425 0.17814933 0.21123442 0.07847503
## cancer        0.17651697 0.11458281 0.11773731 0.10537998 0.00202094
## age           0.02910471 0.01048956 0.01566336 0.02943173 0.08329630
##               osteoporosis     cancer        age
## bucket2008      0.18349678 0.32652869 0.03413982
## ihd             0.27906135 0.20039143 0.04706689
## diabetes        0.26497260 0.16368061 0.01841107
## kidney          0.14637804 0.13797748 0.03810528
## heart.failure   0.22615691 0.15537815 0.02266969
## copd            0.17782104 0.17651697 0.02910471
## alzheimers      0.17196425 0.11458281 0.01048956
## depression      0.17814933 0.11773731 0.01566336
## arthritis       0.21123442 0.10537998 0.02943173
## stroke          0.07847503 0.00202094 0.08329630
## osteoporosis    0.00000000 0.04111404 0.03457520
## cancer          0.04111404 0.00000000 0.01283138
## age             0.03457520 0.01283138 0.00000000
##                   id      cor.y  cor.y.abs cor.low
## 4         bucket2008 0.44655074 0.44655074       1
## 10               ihd 0.41799716 0.41799716       1
## 8           diabetes 0.40363086 0.40363086       1
## 13 reimbursement2008 0.39864605 0.39864605      NA
## 11            kidney 0.38762968 0.38762968       1
## 9      heart.failure 0.36864613 0.36864613       1
## 6               copd 0.33705795 0.33705795       1
## 2         alzheimers 0.29777179 0.29777179       1
## 7         depression 0.27092271 0.27092271       1
## 3          arthritis 0.25517293 0.25517293       1
## 14            stroke 0.19269419 0.19269419       1
## 12      osteoporosis 0.18701176 0.18701176       1
## 5             cancer 0.18571084 0.18571084       1
## 1                age 0.03524295 0.03524295       1
```

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="run.models", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                         chunk_label chunk_step_major chunk_step_minor
## elapsed7 remove_correlated_features                4                1
## elapsed8                 run.models                5                0
##          elapsed
## elapsed7  15.272
## elapsed8  16.147
```

## Step `5`: run models

```r
max_cor_y_x_var <- subset(glb_feats_df, cor.low == 1)[1, "id"]

#   Regression:
if (glb_is_regression) {
    #   Linear:
    myrun_mdl_fn <- myrun_mdl_lm
}    

#   Classification:
if (glb_is_classification) myrun_mdl_fn <- myrun_mdl_classification 
glb_is_binomial <- (length(unique(glb_entity_df[, glb_rsp_var])) <= 2)
    
# Add dummy model - random variable
ret_lst <- myrun_mdl_fn(indep_vars_vctr=".rnorm",
                         rsp_var=glb_rsp_var, 
                         rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_entity_df, OOB_df=glb_newent_df,
                        method=ifelse(glb_is_binomial, "glm", "rpart"))
```

```
## Loading required package: ROCR
## Loading required package: gplots
## 
## Attaching package: 'gplots'
## 
## The following object is masked from 'package:stats':
## 
##     lowess
## 
## Loading required package: caret
## Loading required package: lattice
## 
## Attaching package: 'caret'
## 
## The following object is masked from 'package:survival':
## 
##     cluster
## 
## Loading required package: rpart
```

```
## + Fold1: cp=0.001334 
## - Fold1: cp=0.001334 
## + Fold2: cp=0.001334 
## - Fold2: cp=0.001334 
## + Fold3: cp=0.001334 
## - Fold3: cp=0.001334 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.00203 on full training set
```

```
## Loading required package: rpart.plot
```

![](Medicare_2008_09_files/figure-html/run.models-1.png) ![](Medicare_2008_09_files/figure-html/run.models-2.png) 

```
## n= 1000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 1000 328 B1 (0.67 0.19 0.089 0.043 0.006) *
```

```r
glb_dmy_mdl <- ret_lst[["model"]]

# Highest cor.y
ret_lst <- myrun_mdl_fn(indep_vars_vctr=max_cor_y_x_var,
                         rsp_var=glb_rsp_var, 
                         rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_entity_df, OOB_df=glb_newent_df,
                        method=ifelse(glb_is_binomial, "glm", "rpart"))                        
```

```
## + Fold1: cp=0 
## - Fold1: cp=0 
## + Fold2: cp=0 
## - Fold2: cp=0 
## + Fold3: cp=0 
## - Fold3: cp=0 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0 on full training set
```

![](Medicare_2008_09_files/figure-html/run.models-3.png) ![](Medicare_2008_09_files/figure-html/run.models-4.png) 

```
## n= 1000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 1000 328 B1 (0.67 0.19 0.089 0.043 0.006)  
##    2) bucket2008< 1.5 741 149 B1 (0.8 0.13 0.042 0.023 0.0013) *
##    3) bucket2008>=1.5 259 169 B2 (0.31 0.35 0.22 0.1 0.019)  
##      6) bucket2008< 2.5 139  80 B1 (0.42 0.3 0.17 0.094 0.0072) *
##      7) bucket2008>=2.5 120  72 B2 (0.17 0.4 0.28 0.11 0.033)  
##       14) bucket2008< 4.5 108  63 B2 (0.18 0.42 0.27 0.1 0.037) *
##       15) bucket2008>=4.5 12   7 B3 (0.17 0.25 0.42 0.17 0) *
```

```r
# Enhance Highest cor.y model with additions of interaction terms that were 
#   dropped due to high correlations
if (nrow(subset(glb_feats_df, is.na(cor.low))) > 0) {
    # Only glm handles interaction terms (checked that rpart does not)
    #   This does not work - why ???
#     indep_vars_vctr <- ifelse(glb_is_binomial, 
#         c(max_cor_y_x_var, paste(max_cor_y_x_var, 
#                         subset(glb_feats_df, is.na(cor.low))[, "id"], sep=":")),
#         union(max_cor_y_x_var, subset(glb_feats_df, is.na(cor.low))[, "id"]))
    if (glb_is_binomial) {
        indep_vars_vctr <- 
            c(max_cor_y_x_var, paste(max_cor_y_x_var, 
                        subset(glb_feats_df, is.na(cor.low))[, "id"], sep=":"))       
    } else {
        indep_vars_vctr <- 
            union(max_cor_y_x_var, subset(glb_feats_df, is.na(cor.low))[, "id"])
    }
    ret_lst <- myrun_mdl_fn(indep_vars_vctr,
                        glb_rsp_var, glb_rsp_var_out,
                            fit_df=glb_entity_df, OOB_df=glb_newent_df,
                        method=ifelse(glb_is_binomial, "glm", "rpart"))                        
}    
```

```
## + Fold1: cp=0.0122 
## - Fold1: cp=0.0122 
## + Fold2: cp=0.0122 
## - Fold2: cp=0.0122 
## + Fold3: cp=0.0122 
## - Fold3: cp=0.0122 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.0122 on full training set
```

![](Medicare_2008_09_files/figure-html/run.models-5.png) ![](Medicare_2008_09_files/figure-html/run.models-6.png) 

```
## n= 1000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 1000 328 B1 (0.67 0.19 0.089 0.043 0.006)  
##    2) reimbursement2008< 1535 608  74 B1 (0.88 0.081 0.026 0.013 0.0016) *
##    3) reimbursement2008>=1535 392 251 B2 (0.35 0.36 0.19 0.089 0.013)  
##      6) reimbursement2008< 5575 237 127 B1 (0.46 0.36 0.11 0.068 0) *
##      7) reimbursement2008>=5575 155  99 B2 (0.18 0.36 0.3 0.12 0.032)  
##       14) reimbursement2008>=9115 110  63 B2 (0.18 0.43 0.25 0.1 0.036) *
##       15) reimbursement2008< 9115 45  26 B3 (0.18 0.2 0.42 0.18 0.022) *
```

```r
# Low correlated X
ret_lst <- myrun_mdl_fn(indep_vars_vctr=subset(glb_feats_df, 
                                               cor.low == 1)[, "id"],
                        glb_rsp_var, glb_rsp_var_out,
                        fit_df=glb_entity_df, OOB_df=glb_newent_df,
                        method=ifelse(glb_is_binomial, "glm", "rpart"))                        
```

```
## + Fold1: cp=0.01829 
## - Fold1: cp=0.01829 
## + Fold2: cp=0.01829 
## - Fold2: cp=0.01829 
## + Fold3: cp=0.01829 
## - Fold3: cp=0.01829 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.0198 on full training set
```

![](Medicare_2008_09_files/figure-html/run.models-7.png) ![](Medicare_2008_09_files/figure-html/run.models-8.png) 

```
## n= 1000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 1000 328 B1 (0.67 0.19 0.089 0.043 0.006)  
##   2) ihd< 0.5 572  78 B1 (0.86 0.093 0.019 0.023 0.0017) *
##   3) ihd>=0.5 428 250 B1 (0.42 0.32 0.18 0.07 0.012)  
##     6) bucket2008< 1.5 210  88 B1 (0.58 0.29 0.1 0.033 0) *
##     7) bucket2008>=1.5 218 141 B2 (0.26 0.35 0.26 0.11 0.023) *
```

```r
# User specified
for (method in glb_models_method_vctr) {
    #print(sprintf("iterating over method:%s", method))

    # All X that is not user excluded
    indep_vars_vctr <- setdiff(names(glb_entity_df), 
        union(glb_rsp_var, glb_exclude_vars_as_features))
    
    # easier to exclude features
#     indep_vars_vctr <- setdiff(names(glb_entity_df), 
#         union(union(glb_rsp_var, glb_exclude_vars_as_features), 
#               c("<feat1_name>", "<feat2_name>")))
    
    # easier to include features
#     indep_vars_vctr <- c("<feat1_name>", "<feat2_name>")

    # User specified bivariate models
#     indep_vars_vctr_lst <- list()
#     for (feat in setdiff(names(glb_entity_df), 
#                          union(glb_rsp_var, glb_exclude_vars_as_features)))
#         indep_vars_vctr_lst[["feat"]] <- feat

    # User specified combinatorial models
#     indep_vars_vctr_lst <- list()
#     combn_mtrx <- combn(c("<feat1_name>", "<feat2_name>", "<featn_name>"), 
#                           <num_feats_to_choose>)
#     for (combn_ix in 1:ncol(combn_mtrx))
#         #print(combn_mtrx[, combn_ix])
#         indep_vars_vctr_lst[[combn_ix]] <- combn_mtrx[, combn_ix]

    set.seed(200)
    ret_lst <- myrun_mdl_fn(indep_vars_vctr=indep_vars_vctr,
                            rsp_var=glb_rsp_var, 
                            rsp_var_out=glb_rsp_var_out,
                            fit_df=glb_entity_df, 
                            OOB_df=glb_newent_df,
                            method=method, 
                            tune_models_df=glb_tune_models_df,
                            n_cv_folds=glb_n_cv_folds)
    glb_sel_nlm_mdl <- ret_lst[["model"]]

    set.seed(201)
    ret_lst <- myrun_mdl_fn(indep_vars_vctr=indep_vars_vctr,
                            rsp_var=glb_rsp_var, 
                            rsp_var_out=glb_rsp_var_out,
                            fit_df=glb_entity_df, 
                            OOB_df=glb_newent_df,
                            method=method, 
                            tune_models_df=glb_tune_models_df,
                            n_cv_folds=glb_n_cv_folds,
                            loss_mtrx=glb_loss_mtrx,
                            summaryFunction=glb_loss_smmry,
                            metric="loss.error",
                            maximize=FALSE)
    glb_sel_mdl <- glb_sel_wlm_mdl <- ret_lst[["model"]]
}
```

```
## + Fold1: cp=0 
## - Fold1: cp=0 
## + Fold2: cp=0 
## - Fold2: cp=0 
## + Fold3: cp=0 
## - Fold3: cp=0 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.03 on full training set
```

![](Medicare_2008_09_files/figure-html/run.models-9.png) ![](Medicare_2008_09_files/figure-html/run.models-10.png) 

```
## n= 1000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 1000 328 B1 (0.67 0.19 0.089 0.043 0.006)  
##    2) reimbursement2008< 1535 608  74 B1 (0.88 0.081 0.026 0.013 0.0016) *
##    3) reimbursement2008>=1535 392 251 B2 (0.35 0.36 0.19 0.089 0.013)  
##      6) reimbursement2008< 5575 237 127 B1 (0.46 0.36 0.11 0.068 0) *
##      7) reimbursement2008>=5575 155  99 B2 (0.18 0.36 0.3 0.12 0.032)  
##       14) copd< 0.5 77  40 B2 (0.17 0.48 0.23 0.1 0.013) *
##       15) copd>=0.5 78  49 B3 (0.19 0.24 0.37 0.14 0.051) *
## + Fold1: cp=0 
## - Fold1: cp=0 
## + Fold2: cp=0 
## - Fold2: cp=0 
## + Fold3: cp=0 
## - Fold3: cp=0 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.01 on full training set
```

![](Medicare_2008_09_files/figure-html/run.models-11.png) ![](Medicare_2008_09_files/figure-html/run.models-12.png) 

```
## n= 1000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 1000 328 B1 (0.67 0.19 0.089 0.043 0.006)  
##    2) reimbursement2008< 1535 608  74 B1 (0.88 0.081 0.026 0.013 0.0016) *
##    3) reimbursement2008>=1535 392 251 B2 (0.35 0.36 0.19 0.089 0.013)  
##      6) reimbursement2008< 5575 237 127 B1 (0.46 0.36 0.11 0.068 0) *
##      7) reimbursement2008>=5575 155  99 B2 (0.18 0.36 0.3 0.12 0.032)  
##       14) copd< 0.5 77  40 B2 (0.17 0.48 0.23 0.1 0.013) *
##       15) copd>=0.5 78  49 B3 (0.19 0.24 0.37 0.14 0.051) *
```

```r
# Simplify a model
# fit_df <- glb_entity_df; glb_mdl <- step(<complex>_mdl)

rownames(glb_models_df) <- seq(1:nrow(glb_models_df))
plot_models_df <- mutate(glb_models_df, feats.label=substr(feats, 1, 30))
if (glb_is_regression) {
    print(orderBy(~ -R.sq.OOB -Adj.R.sq.fit, glb_models_df))
    stop("glb_sel_mdl not selected")
    print(myplot_scatter(plot_models_df, "Adj.R.sq.fit", "R.sq.OOB") + 
          geom_text(aes(label=feats.label), data=plot_models_df, color="NavyBlue", 
                    size=3.5, angle=45))
}    

if (glb_is_classification) {
    # Lower AIC is better
    sel_frmla <- formula(paste("~ ", paste0("-auc.OOB", "-accuracy.fit", "+accuracySD.fit")))
    print(tmp_models_df <- orderBy(sel_frmla, glb_models_df))    
#     print(tmp_models_sav_df <- orderBy(~ -auc.OOB -accuracy.fit +accuracySD.fit, glb_models_df))
    print("Best model using metrics:")
#     print(sprintf("Best model using metrics:", sel_frmla))
    print(summary(bst_mdl <- glb_models_lst[[as.numeric(rownames(tmp_models_df)[1])]]))

    if (!is.null(glb_sel_mdl)) { # Model is user selected
        print("User selected model: ")
        print(summary(glb_sel_mdl))
    } else glb_sel_mdl <- bst_mdl
   
#     glb_sel_mdl <- glb_models_lst[[as.numeric(rownames(tmp_models_df)[1])]]
#     print(summary(glb_sel_mdl))
#     print("Selected model:glb_sel_mdl:")    
#     glb_sel_mdl <- glb_models_lst[[as.numeric(rownames(tmp_models_df)[1])]]
#     print(summary(glb_sel_mdl))    
    
    plot_models_df[, "inv.AIC.fit"] <- 1.0 / plot_models_df[, "AIC.fit"] 
    if (any(!is.na(plot_models_df$inv.AIC.fit)) | any(!is.na(plot_models_df$auc.OOB))) {
        print(myplot_scatter(plot_models_df, "inv.AIC.fit", "auc.OOB") + 
              geom_text(aes(label=feats.label), data=plot_models_df, color="NavyBlue", 
                        size=3.5, angle=45))
    } else warning("All NAs for inv.AIC.fit vs. auc.OOB scatter plot of glb_models_df")
    
    if (any(!is.na(plot_models_df$auc.OOB))) {
        print(myplot_scatter(plot_models_df, "auc.OOB", "accuracy.fit",  
                         colorcol_name="method") + 
          geom_errorbar(aes(x=auc.OOB, ymin=accuracy.fit - accuracySD.fit,
                            ymax=accuracy.fit + accuracySD.fit), 
            width=(max(plot_models_df$auc.OOB)-min(plot_models_df$auc.OOB))/25) +      
          geom_text(aes(label=feats.label), data=plot_models_df, color="NavyBlue", 
                    size=3.5, angle=45))
    } else {
        warning("All NAs for auc.OOB in auc.OOB vs. accuracy.fit scatter plot of glb_models_df")
        print(ggplot(plot_models_df, aes(x=reorder(feats.label, accuracy.fit), y=accuracy.fit, 
                                         fill=factor(method))) +
                  geom_bar(stat="identity", position="dodge") + 
          geom_errorbar(aes(x=feats.label, ymin=accuracy.fit - accuracySD.fit,
                            ymax=accuracy.fit + accuracySD.fit, color=factor(method)), 
            width=0.2) +      
          theme(axis.text.x = element_text(angle = 45,vjust = 1)))        
    }    
    
    # mdl$times plot across models
    print(myplot_scatter(plot_models_df, "inv.elapsedtime.everything", "inv.elapsedtime.final",  
                     colorcol_name="method") + 
          geom_point(aes(size=accuracy.fit)) +      
          geom_text(aes(label=feats.label), data=plot_models_df, color="NavyBlue", 
                    size=3.5, angle=45) + 
        geom_smooth(method="lm"))
}
```

```
##   method
## 2  rpart
## 3  rpart
## 5  rpart
## 4  rpart
## 1  rpart
## 6  rpart
##                                                                                                                                             feats
## 2                                                                                                                                      bucket2008
## 3                                                                                                                   bucket2008, reimbursement2008
## 5 age, alzheimers, arthritis, cancer, copd, depression, diabetes, heart.failure, ihd, kidney, osteoporosis, stroke, reimbursement2008, bucket2008
## 4                    bucket2008, ihd, diabetes, kidney, heart.failure, copd, alzheimers, depression, arthritis, stroke, osteoporosis, cancer, age
## 1                                                                                                                                          .rnorm
## 6 age, alzheimers, arthritis, cancer, copd, depression, diabetes, heart.failure, ihd, kidney, osteoporosis, stroke, reimbursement2008, bucket2008
##   n.fit inv.elapsedtime.everything inv.elapsedtime.final R.sq.fit R.sq.OOB
## 2  1000                  1.1173184              55.55556       NA       NA
## 3  1000                  1.0615711              35.71429       NA       NA
## 5  1000                  0.8833922              17.54386       NA       NA
## 4  1000                  0.9293680              18.51852       NA       NA
## 1  1000                  0.6101281              38.46154       NA       NA
## 6  1000                  0.6150062              17.54386       NA       NA
##   Adj.R.sq.fit SSE.fit SSE.OOB AIC.fit auc.fit auc.OOB accuracy.fit
## 2           NA       0      NA      NA      NA      NA    0.7010447
## 3           NA       0      NA      NA      NA      NA    0.6989864
## 5           NA       0      NA      NA      NA      NA    0.6870463
## 4           NA       0      NA      NA      NA      NA    0.6869984
## 1           NA       0      NA      NA      NA      NA    0.6439452
## 6           NA       0      NA      NA      NA      NA           NA
##   accuracySD.fit
## 2    0.015160702
## 3    0.011863337
## 5    0.040817386
## 4    0.004719059
## 1    0.022505626
## 6             NA
## [1] "Best model using metrics:"
## Call:
## rpart(formula = .outcome ~ ., data = list(bucket2008 = c(1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 
## 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 4, 1, 1, 2, 
## 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 
## 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 
## 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 4, 1, 1, 1, 3, 
## 1, 1, 1, 1, 3, 1, 2, 2, 3, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 
## 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 
## 2, 4, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 3, 1, 1, 1, 1, 1, 1, 2, 1, 
## 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 2, 2, 3, 2, 2, 1, 3, 4, 1, 3, 1, 
## 1, 1, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 3, 1, 2, 3, 1, 1, 1, 1, 1, 
## 2, 2, 3, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 
## 4, 1, 5, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 4, 1, 1, 2, 1, 3, 2, 
## 2, 3, 1, 1, 1, 1, 3, 1, 1, 4, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 
## 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 4, 2, 2, 1, 1, 2, 
## 1, 2, 4, 2, 1, 2, 3, 1, 2, 1, 1, 2, 5, 2, 1, 3, 1, 2, 4, 1, 1, 
## 1, 1, 1, 1, 2, 4, 2, 1, 1, 4, 1, 4, 3, 1, 1, 1, 2, 3, 1, 1, 1, 
## 2, 1, 2, 3, 1, 1, 1, 4, 3, 3, 1, 1, 2, 4, 2, 1, 1, 1, 5, 1, 1, 
## 1, 4, 3, 3, 1, 2, 3, 1, 1, 2, 1, 1, 3, 1, 2, 3, 1, 1, 1, 3, 1, 
## 1, 2, 4, 1, 1, 2, 1, 2, 2, 2, 1, 4, 2, 1, 1, 3, 1, 2, 1, 2, 2, 
## 2, 4, 1, 1, 1, 4, 1, 4, 4, 1, 1, 3, 4, 1, 2, 1, 1, 1, 3, 1, 1, 
## 1, 5, 3, 1, 1, 3, 4, 2, 2, 1, 3, 3, 3, 3, 1, 3, 1, 1, 1, 1, 3, 
## 4, 2, 2, 2, 1, 1, 2, 2, 2, 4, 4, 2, 1, 1, 2, 4, 1, 5, 5, 1, 1, 
## 5, 5, 1, 3, 2, 3, 3, 4, 1, 1, 4, 2, 1, 1, 1, 3, 3, 4, 5, 1, 3, 
## 2, 2, 2, 2, 1, 1, 1, 2, 3, 1, 3, 1, 4, 4, 3, 2, 1, 1, 2, 2, 1, 
## 1, 1, 3, 3, 2, 2, 1, 3, 1, 1, 3, 2, 4, 3, 3, 1, 1, 3, 2, 1, 2, 
## 3, 1, 2, 3, 2, 1, 1, 1, 3, 2, 2, 2, 1, 1, 3, 3, 1, 2, 3, 2, 2, 
## 2, 1, 1, 3, 5, 2, 1, 3, 1, 4, 5, 1, 2, 1, 1, 3, 2, 4, 1, 2, 1, 
## 3, 1, 2, 1, 3, 2, 1, 4, 4, 4, 4), .outcome = c(1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 
## 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
## 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
## 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
## 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
## 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
## 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
## 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
## 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
## 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 
## 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
## 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
## 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
## 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
## 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
## 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
## 4, 4, 5, 5, 5, 5, 5, 5)), control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 1000 
## 
##            CP nsplit rel error
## 1 0.041158537      0 1.0000000
## 2 0.006097561      2 0.9176829
## 3 0.000000000      3 0.9115854
## 
## Variable importance
## bucket2008 
##        100 
## 
## Node number 1: 1000 observations,    complexity param=0.04115854
##   predicted class=B1  expected loss=0.328  P(node) =1
##     class counts:   672   190    89    43     6
##    probabilities: 0.672 0.190 0.089 0.043 0.006 
##   left son=2 (741 obs) right son=3 (259 obs)
##   Primary splits:
##       bucket2008 < 1.5 to the left,  improve=62.33394, (0 missing)
## 
## Node number 2: 741 observations
##   predicted class=B1  expected loss=0.2010796  P(node) =0.741
##     class counts:   592   100    31    17     1
##    probabilities: 0.799 0.135 0.042 0.023 0.001 
## 
## Node number 3: 259 observations,    complexity param=0.04115854
##   predicted class=B2  expected loss=0.6525097  P(node) =0.259
##     class counts:    80    90    58    26     5
##    probabilities: 0.309 0.347 0.224 0.100 0.019 
##   left son=6 (139 obs) right son=7 (120 obs)
##   Primary splits:
##       bucket2008 < 2.5 to the left,  improve=5.471183, (0 missing)
## 
## Node number 6: 139 observations
##   predicted class=B1  expected loss=0.5755396  P(node) =0.139
##     class counts:    59    42    24    13     1
##    probabilities: 0.424 0.302 0.173 0.094 0.007 
## 
## Node number 7: 120 observations,    complexity param=0.006097561
##   predicted class=B2  expected loss=0.6  P(node) =0.12
##     class counts:    21    48    34    13     4
##    probabilities: 0.175 0.400 0.283 0.108 0.033 
##   left son=14 (108 obs) right son=15 (12 obs)
##   Primary splits:
##       bucket2008 < 4.5 to the left,  improve=0.5981481, (0 missing)
## 
## Node number 14: 108 observations
##   predicted class=B2  expected loss=0.5833333  P(node) =0.108
##     class counts:    19    45    29    11     4
##    probabilities: 0.176 0.417 0.269 0.102 0.037 
## 
## Node number 15: 12 observations
##   predicted class=B3  expected loss=0.5833333  P(node) =0.012
##     class counts:     2     3     5     2     0
##    probabilities: 0.167 0.250 0.417 0.167 0.000 
## 
## n= 1000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 1000 328 B1 (0.67 0.19 0.089 0.043 0.006)  
##    2) bucket2008< 1.5 741 149 B1 (0.8 0.13 0.042 0.023 0.0013) *
##    3) bucket2008>=1.5 259 169 B2 (0.31 0.35 0.22 0.1 0.019)  
##      6) bucket2008< 2.5 139  80 B1 (0.42 0.3 0.17 0.094 0.0072) *
##      7) bucket2008>=2.5 120  72 B2 (0.17 0.4 0.28 0.11 0.033)  
##       14) bucket2008< 4.5 108  63 B2 (0.18 0.42 0.27 0.1 0.037) *
##       15) bucket2008>=4.5 12   7 B3 (0.17 0.25 0.42 0.17 0) *
## [1] "User selected model: "
## Call:
## rpart(formula = .outcome ~ ., data = list(age = c(73, 75, 68, 
## 95, 72, 75, 77, 73, 72, 93, 71, 60, 66, 77, 68, 80, 67, 61, 85, 
## 77, 84, 90, 73, 85, 77, 62, 69, 67, 66, 66, 73, 76, 54, 85, 72, 
## 68, 70, 67, 78, 93, 69, 90, 79, 67, 61, 90, 71, 78, 69, 58, 78, 
## 83, 67, 76, 74, 90, 86, 91, 77, 67, 72, 66, 86, 70, 29, 74, 74, 
## 76, 75, 76, 80, 58, 67, 96, 88, 68, 67, 49, 59, 70, 66, 74, 76, 
## 68, 66, 71, 66, 61, 73, 67, 74, 71, 75, 44, 74, 81, 69, 72, 67, 
## 80, 53, 88, 77, 80, 82, 67, 75, 96, 67, 53, 66, 77, 99, 70, 70, 
## 66, 74, 77, 70, 63, 68, 70, 78, 70, 79, 46, 56, 73, 73, 68, 68, 
## 87, 77, 59, 75, 67, 80, 81, 78, 54, 64, 93, 57, 29, 84, 70, 85, 
## 64, 66, 87, 66, 86, 78, 69, 77, 66, 88, 71, 72, 66, 86, 76, 69, 
## 67, 77, 84, 74, 69, 80, 84, 86, 82, 64, 69, 70, 84, 81, 70, 66, 
## 55, 74, 47, 69, 87, 78, 89, 71, 73, 51, 75, 82, 82, 63, 68, 76, 
## 71, 75, 76, 73, 99, 79, 69, 69, 86, 80, 75, 66, 73, 75, 86, 91, 
## 74, 68, 34, 78, 83, 78, 64, 89, 60, 85, 83, 63, 84, 68, 69, 83, 
## 77, 72, 69, 67, 60, 82, 75, 77, 74, 97, 70, 65, 68, 74, 78, 70, 
## 83, 68, 67, 83, 67, 58, 72, 70, 66, 72, 84, 69, 67, 98, 75, 52, 
## 72, 45, 85, 76, 84, 85, 72, 69, 63, 67, 61, 95, 72, 69, 70, 67, 
## 70, 64, 71, 69, 80, 67, 49, 84, 33, 40, 70, 78, 70, 79, 40, 83, 
## 67, 78, 66, 41, 90, 72, 66, 99, 86, 77, 90, 70, 66, 61, 76, 80, 
## 95, 81, 68, 66, 78, 71, 49, 87, 71, 68, 87, 76, 86, 76, 70, 68, 
## 84, 73, 72, 74, 87, 49, 72, 87, 69, 68, 61, 71, 79, 68, 98, 70, 
## 79, 67, 57, 55, 52, 87, 100, 74, 78, 80, 44, 76, 66, 82, 65, 
## 70, 77, 46, 71, 80, 83, 42, 39, 81, 83, 66, 60, 32, 70, 73, 75, 
## 83, 70, 60, 78, 68, 71, 75, 38, 85, 67, 99, 87, 78, 65, 67, 55, 
## 68, 75, 79, 80, 67, 58, 76, 68, 78, 73, 84, 82, 71, 74, 85, 70, 
## 74, 82, 72, 84, 87, 67, 70, 90, 69, 72, 82, 83, 76, 70, 79, 90, 
## 97, 58, 70, 45, 71, 73, 66, 93, 66, 70, 74, 83, 78, 75, 67, 66, 
## 47, 66, 71, 52, 81, 41, 38, 82, 70, 66, 71, 74, 76, 82, 90, 82, 
## 69, 79, 46, 83, 96, 32, 55, 69, 71, 73, 37, 68, 66, 68, 68, 71, 
## 88, 84, 56, 80, 81, 78, 47, 73, 72, 68, 78, 68, 76, 84, 98, 76, 
## 76, 92, 85, 59, 73, 68, 67, 71, 72, 72, 74, 74, 94, 70, 76, 68, 
## 87, 55, 68, 69, 70, 70, 65, 70, 80, 73, 73, 85, 78, 70, 46, 81, 
## 82, 65, 96, 87, 84, 75, 32, 62, 72, 72, 66, 73, 72, 67, 76, 71, 
## 76, 75, 74, 77, 63, 68, 75, 71, 68, 51, 79, 73, 74, 89, 71, 78, 
## 46, 63, 68, 83, 71, 56, 70, 83, 89, 41, 29, 70, 86, 49, 88, 68, 
## 68, 88, 93, 78, 69, 70, 69, 83, 58, 76, 70, 84, 58, 82, 79, 79, 
## 83, 73, 64, 83, 79, 34, 91, 67, 76, 60, 77, 75, 62, 68, 95, 67, 
## 70, 67, 66, 69, 76, 74, 96, 86, 79, 60, 73, 88, 69, 69, 70, 71, 
## 49, 75, 80, 80, 73, 77, 91, 76, 94, 77, 67, 89, 73, 76, 65, 66, 
## 63, 87, 57, 73, 70, 76, 69, 58, 71, 87, 83, 72, 77, 37, 29, 78, 
## 82, 30, 89, 76, 57, 74, 37, 70, 72, 71, 69, 85, 76, 88, 81, 52, 
## 76, 50, 51, 69, 71, 37, 79, 100, 71, 90, 68, 66, 84, 69, 50, 
## 62, 68, 71, 80, 71, 75, 67, 90, 72, 80, 96, 73, 66, 52, 48, 71, 
## 80, 71, 66, 81, 74, 85, 76, 69, 74, 77, 74, 70, 50, 70, 67, 75, 
## 85, 73, 78, 70, 66, 88, 77, 77, 74, 70, 77, 82, 66, 85, 80, 71, 
## 49, 57, 73, 75, 68, 77, 58, 72, 73, 89, 70, 78, 84, 71, 41, 99, 
## 67, 87, 78, 70, 78, 79, 68, 75, 47, 75, 70, 77, 73, 73, 76, 77, 
## 75, 86, 91, 86, 67, 72, 66, 67, 28, 38, 72, 71, 88, 82, 45, 66, 
## 72, 70, 72, 58, 80, 72, 90, 75, 87, 80, 84, 89, 72, 82, 79, 68, 
## 83, 79, 90, 80, 73, 70, 86, 75, 72, 70, 74, 83, 66, 89, 85, 71, 
## 87, 77, 66, 70, 91, 96, 79, 50, 73, 76, 69, 55, 58, 61, 76, 67, 
## 72, 61, 87, 87, 39, 73, 96, 79, 70, 41, 76, 83, 90, 59, 59, 66, 
## 69, 84, 74, 85, 67, 69, 76, 66, 76, 73, 64, 71, 70, 80, 51, 72, 
## 78, 88, 70, 81, 87, 94, 75, 70, 76, 88, 78, 68, 72, 90, 83, 85, 
## 79, 73, 80, 83, 64, 92, 74, 88, 85, 75, 77, 48, 56, 69, 71, 53, 
## 39, 94, 53, 73, 64, 83, 97, 72, 75, 70, 71, 88, 68, 66, 70, 80, 
## 60, 81, 69, 86, 68, 28, 87, 72, 71, 84, 37, 85, 53, 85, 71, 73, 
## 74, 85, 66, 71, 77, 80, 83, 81, 82, 77, 93, 72, 80, 69, 80, 86, 
## 78, 93, 70, 83, 84, 69, 83, 65, 82, 80, 28, 67, 90, 87, 37, 70, 
## 77, 56, 74, 80, 68, 98, 73, 87, 26, 38, 69, 100, 66, 85, 52, 
## 73, 83, 81, 79, 71, 94, 85, 83, 38, 67, 74, 85, 86, 77, 70, 78, 
## 71, 93, 71, 70, 79, 69, 84, 85, 66, 63, 42, 88, 78, 78, 60, 46, 
## 84, 77, 79, 71, 73, 57, 80, 76), alzheimers = c(0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
## 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
## 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
## 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 
## 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 
## 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 
## 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 
## 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 
## 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 
## 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
## 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 
## 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
## 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 
## 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 
## 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 
## 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 
## 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 
## 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 
## 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 
## 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 
## 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 
## 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 
## 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 
## 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 
## 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 
## 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 
## 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 
## 0, 0, 0, 0, 1, 0, 0, 0), arthritis = c(0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
## 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
## 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
## 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 
## 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
## 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 
## 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 
## 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 
## 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 
## 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 
## 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 
## 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 
## 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
## 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 
## 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 
## 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 
## 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 
## 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
## 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 
## 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 
## 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 
## 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 
## 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 
## 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
## 1, 1, 0, 0, 0), cancer = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
## 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 
## 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
## 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
## 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
## 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 
## 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 
## 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 
## 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 
## 0), copd = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 
## 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 
## 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
## 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
## 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
## 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 
## 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 
## 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
## 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
## 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 
## 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 
## 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
## 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 
## 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 
## 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 
## 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 
## 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
## 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 
## 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1), depression = c(0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
## 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
## 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 
## 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 
## 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 
## 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 
## 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
## 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 
## 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 
## 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 
## 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 
## 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 
## 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 
## 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 
## 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 
## 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 
## 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 
## 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 
## 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
## 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 
## 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 
## 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 
## 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 
## 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 
## 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 
## 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1), diabetes = c(0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 
## 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 
## 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 
## 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 
## 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 
## 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 
## 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 
## 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 
## 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 
## 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 
## 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 
## 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 
## 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 
## 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 
## 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 
## 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 
## 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 
## 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 
## 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 
## 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 
## 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 
## 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 
## 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 
## 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 
## 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 
## 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 
## 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 
## 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 
## 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 
## 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 
## 1, 0, 1, 1, 1, 1, 1, 1, 1), heart.failure = c(0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
## 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
## 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 
## 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 
## 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 
## 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 
## 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 
## 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 
## 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 
## 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 
## 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 
## 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 
## 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 
## 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 
## 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 
## 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 
## 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 
## 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 
## 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 
## 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
## 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 
## 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 
## 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 
## 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 
## 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 
## 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 
## 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 
## 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 
## 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 
## 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 
## 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 
## 1, 1, 0, 0, 1, 1, 1, 1), ihd = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
## 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
## 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 
## 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 
## 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 
## 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 
## 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 
## 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 
## 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 
## 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 
## 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 
## 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 
## 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 
## 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 
## 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 
## 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 
## 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 
## 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 
## 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 
## 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 
## 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 
## 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 
## 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 
## 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 
## 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 
## 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 
## 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 
## 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 
## 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
## 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 
## 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 
## 1, 1, 1), kidney = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
## 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 
## 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 
## 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
## 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 
## 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
## 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 
## 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
## 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 
## 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 
## 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 
## 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 
## 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 
## 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 
## 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 
## 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 
## 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
## 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 
## 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 
## 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
## 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 
## 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 
## 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 
## 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1), 
##     osteoporosis = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 
##     0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
##     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
##     0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 
##     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 
##     1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 
##     1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 
##     0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 
##     0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 
##     0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
##     0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 
##     0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
##     1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 
##     0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 
##     0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 
##     0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 
##     0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 
##     0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 
##     1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 
##     0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 
##     1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 
##     1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 
##     1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
##     0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 
##     0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 
##     1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1), stroke = c(0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 
##     0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
##     0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0), reimbursement2008 = c(0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##     0, 0, 20, 20, 40, 40, 50, 70, 100, 120, 140, 140, 240, 250, 
##     260, 280, 480, 0, 0, 0, 0, 0, 70, 0, 0, 0, 0, 80, 0, 0, 0, 
##     0, 10, 90, 320, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 140, 
##     180, 410, 0, 0, 0, 0, 0, 140, 0, 10, 60, 100, 190, 290, 120, 
##     0, 0, 70, 0, 130, 680, 0, 0, 390, 0, 860, 1070, 0, 0, 30, 
##     0, 0, 250, 320, 1100, 0, 0, 60, 70, 90, 0, 0, 1220, 1500, 
##     300, 1390, 0, 0, 0, 2080, 70, 200, 240, 300, 60, 1250, 1440, 
##     0, 920, 1860, 500, 0, 3410, 0, 0, 1100, 3740, 90, 270, 500, 
##     0, 0, 0, 150, 170, 680, 0, 380, 6180, 790, 900, 120, 220, 
##     3130, 0, 780, 40, 280, 580, 0, 50, 360, 600, 0, 200, 370, 
##     410, 730, 760, 0, 0, 1050, 0, 590, 0, 330, 980, 630, 0, 1020, 
##     5330, 0, 1790, 180, 810, 1040, 1120, 5140, 58230, 0, 800, 
##     1490, 0, 0, 830, 1960, 110, 240, 3930, 960, 320, 4500, 230, 
##     360, 660, 21620, 190, 830, 3360, 20, 400, 1310, 3890, 0, 
##     360, 430, 1450, 80, 370, 930, 260, 1260, 1310, 1430, 0, 150, 
##     1760, 400, 1030, 1240, 5410, 750, 50, 1510, 1160, 3690, 1800, 
##     2770, 580, 640, 720, 250, 530, 300, 940, 10, 80, 1530, 0, 
##     2360, 22830, 0, 570, 340, 1110, 250, 470, 1170, 5630, 980, 
##     0, 0, 350, 370, 650, 690, 440, 580, 3470, 710, 310, 280, 
##     310, 450, 660, 1840, 2280, 1190, 820, 660, 820, 2290, 3580, 
##     560, 760, 2880, 5110, 0, 290, 390, 690, 120, 150, 850, 1160, 
##     490, 920, 950, 1250, 2820, 2820, 1170, 1130, 1490, 370, 530, 
##     910, 1780, 1120, 1920, 5090, 1680, 100, 260, 1070, 660, 900, 
##     3700, 530, 810, 1110, 1400, 1410, 670, 2010, 830, 1560, 5480, 
##     1530, 180, 620, 900, 1150, 2490, 1630, 930, 1900, 590, 1440, 
##     2060, 670, 660, 1230, 1530, 1620, 0, 1490, 310, 860, 990, 
##     400, 1380, 1660, 250, 2020, 490, 590, 1010, 540, 590, 4260, 
##     950, 1320, 1240, 360, 1430, 1520, 840, 920, 3930, 880, 530, 
##     28370, 400, 10, 650, 8730, 270, 1270, 2520, 1730, 12840, 
##     2010, 3290, 3510, 9640, 2300, 3540, 2380, 4160, 950, 440, 
##     3420, 3570, 220, 680, 1670, 2970, 540, 5260, 820, 2220, 230, 
##     1040, 1350, 3420, 1160, 1330, 1380, 790, 1080, 2770, 430, 
##     630, 4280, 170, 3400, 5300, 1630, 4540, 20500, 2500, 2990, 
##     3690, 5380, 3190, 310, 1240, 5470, 2380, 3480, 9370, 1780, 
##     260, 2100, 2790, 770, 1220, 3760, 370, 0, 1150, 4190, 1230, 
##     4320, 3230, 2790, 420, 2220, 1750, 6370, 7500, 12590, 4810, 
##     5550, 430, 14880, 30510, 2190, 14750, 950, 1350, 1410, 980, 
##     2010, 1720, 1180, 13570, 6250, 690, 1310, 1850, 2850, 13440, 
##     810, 3700, 10180, 2340, 1730, 2400, 260, 620, 3250, 3750, 
##     12900, 1250, 2280, 1160, 3070, 1590, 4850, 3490, 3060, 470, 
##     1360, 7610, 0, 1220, 3220, 1050, 6160, 4650, 1000, 22910, 
##     1500, 68350, 590, 160, 630, 1680, 2240, 750, 1330, 17370, 
##     620, 1090, 1830, 19090, 840, 1020, 3960, 1380, 14010, 4040, 
##     3790, 18640, 690, 2790, 0, 1750, 10880, 1070, 2380, 32000, 
##     2020, 2360, 3240, 1840, 3070, 3250, 790, 3950, 3210, 1060, 
##     3210, 1870, 2490, 1530, 1930, 1490, 2440, 5240, 20060, 1050, 
##     920, 2780, 2880, 1840, 1810, 1430, 28670, 4720, 6720, 520, 
##     680, 4280, 590, 3070, 38770, 4490, 760, 7040, 12150, 2920, 
##     3450, 1250, 1890, 4090, 72910, 6630, 1580, 10360, 1430, 4210, 
##     19620, 240, 290, 920, 1580, 0, 2290, 5070, 29370, 6320, 1340, 
##     1690, 19570, 1510, 26480, 8390, 2150, 1560, 1690, 5490, 13530, 
##     1250, 2440, 1760, 5350, 1720, 3350, 9550, 2150, 940, 2630, 
##     52130, 10480, 17380, 360, 2170, 6140, 19580, 4910, 670, 2240, 
##     890, 58260, 1860, 300, 1620, 32810, 11480, 14970, 2100, 4470, 
##     13270, 2090, 290, 3330, 210, 700, 11420, 2120, 3070, 12940, 
##     560, 50, 1880, 11140, 1110, 2300, 3850, 45560, 2510, 200, 
##     4810, 1250, 5140, 5630, 3720, 680, 24200, 3590, 2570, 1120, 
##     12600, 1720, 3260, 2970, 7500, 4780, 6040, 39080, 810, 1830, 
##     1880, 29880, 350, 36900, 27900, 1030, 2380, 17470, 23360, 
##     790, 3500, 760, 590, 780, 16100, 1140, 2020, 290, 56960, 
##     11790, 1650, 2020, 11750, 30500, 3390, 3940, 2200, 13570, 
##     18340, 9240, 12320, 830, 16980, 2920, 630, 2800, 1970, 8520, 
##     48110, 4310, 7840, 5990, 2180, 2500, 7160, 7740, 4670, 28790, 
##     41560, 4920, 540, 240, 3120, 41400, 1500, 65590, 75670, 2600, 
##     2290, 68910, 65680, 2570, 12260, 6120, 8230, 18050, 26330, 
##     530, 1150, 51460, 5660, 2870, 1700, 910, 9330, 13340, 38640, 
##     81390, 1550, 14110, 6690, 5060, 4950, 4250, 1260, 2400, 990, 
##     7810, 8990, 2360, 8850, 2550, 25590, 33710, 8430, 5600, 2670, 
##     1140, 5990, 4400, 1200, 1550, 370, 10390, 15300, 6570, 3330, 
##     630, 14440, 2640, 1300, 10550, 3700, 32170, 12260, 18710, 
##     500, 760, 11800, 3180, 1090, 5980, 18160, 2800, 7490, 8420, 
##     3450, 2450, 2540, 1880, 10550, 4060, 4210, 3140, 520, 2000, 
##     10530, 13800, 240, 7010, 18450, 7790, 6280, 3490, 1540, 1860, 
##     8920, 65250, 7250, 260, 13400, 1870, 29820, 62990, 1890, 
##     3410, 80, 0, 10190, 5670, 21700, 0, 5060, 10, 11230, 1140, 
##     7840, 2720, 8560, 7760, 550, 45890, 47990, 40610, 53550), 
##     bucket2008 = c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 
##     1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 
##     2, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 4, 
##     1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 4, 1, 1, 1, 3, 
##     1, 1, 1, 1, 3, 1, 2, 2, 3, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 
##     1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 
##     1, 2, 2, 1, 2, 4, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 3, 1, 1, 
##     1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 2, 2, 3, 
##     2, 2, 1, 3, 4, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 2, 1, 1, 1, 
##     1, 3, 1, 2, 3, 1, 1, 1, 1, 1, 2, 2, 3, 1, 1, 1, 2, 1, 2, 
##     2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 4, 1, 5, 1, 1, 1, 1, 
##     1, 1, 1, 3, 1, 1, 1, 4, 1, 1, 2, 1, 3, 2, 2, 3, 1, 1, 1, 
##     1, 3, 1, 1, 4, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1, 
##     1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 4, 2, 2, 1, 1, 2, 1, 
##     2, 4, 2, 1, 2, 3, 1, 2, 1, 1, 2, 5, 2, 1, 3, 1, 2, 4, 1, 
##     1, 1, 1, 1, 1, 2, 4, 2, 1, 1, 4, 1, 4, 3, 1, 1, 1, 2, 3, 
##     1, 1, 1, 2, 1, 2, 3, 1, 1, 1, 4, 3, 3, 1, 1, 2, 4, 2, 1, 
##     1, 1, 5, 1, 1, 1, 4, 3, 3, 1, 2, 3, 1, 1, 2, 1, 1, 3, 1, 
##     2, 3, 1, 1, 1, 3, 1, 1, 2, 4, 1, 1, 2, 1, 2, 2, 2, 1, 4, 
##     2, 1, 1, 3, 1, 2, 1, 2, 2, 2, 4, 1, 1, 1, 4, 1, 4, 4, 1, 
##     1, 3, 4, 1, 2, 1, 1, 1, 3, 1, 1, 1, 5, 3, 1, 1, 3, 4, 2, 
##     2, 1, 3, 3, 3, 3, 1, 3, 1, 1, 1, 1, 3, 4, 2, 2, 2, 1, 1, 
##     2, 2, 2, 4, 4, 2, 1, 1, 2, 4, 1, 5, 5, 1, 1, 5, 5, 1, 3, 
##     2, 3, 3, 4, 1, 1, 4, 2, 1, 1, 1, 3, 3, 4, 5, 1, 3, 2, 2, 
##     2, 2, 1, 1, 1, 2, 3, 1, 3, 1, 4, 4, 3, 2, 1, 1, 2, 2, 1, 
##     1, 1, 3, 3, 2, 2, 1, 3, 1, 1, 3, 2, 4, 3, 3, 1, 1, 3, 2, 
##     1, 2, 3, 1, 2, 3, 2, 1, 1, 1, 3, 2, 2, 2, 1, 1, 3, 3, 1, 
##     2, 3, 2, 2, 2, 1, 1, 3, 5, 2, 1, 3, 1, 4, 5, 1, 2, 1, 1, 
##     3, 2, 4, 1, 2, 1, 3, 1, 2, 1, 3, 2, 1, 4, 4, 4, 4), .outcome = c(1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
##     1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
##     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
##     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
##     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
##     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
##     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
##     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
##     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
##     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
##     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
##     2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
##     3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
##     3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
##     3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
##     3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
##     4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
##     4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
##     4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5)), control = list(minsplit = 20, 
##     minbucket = 7, cp = 0, maxcompete = 4, maxsurrogate = 5, 
##     usesurrogate = 2, surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 1000 
## 
##           CP nsplit rel error
## 1 0.04268293      0 1.0000000
## 2 0.03048780      2 0.9146341
## 3 0.01000000      3 0.8841463
## 
## Variable importance
## reimbursement2008        bucket2008               ihd     heart.failure 
##                30                20                13                11 
##          diabetes            kidney              copd 
##                11                11                 2 
## 
## Node number 1: 1000 observations,    complexity param=0.04268293
##   predicted class=B1  expected loss=0.328  P(node) =1
##     class counts:   672   190    89    43     6
##    probabilities: 0.672 0.190 0.089 0.043 0.006 
##   left son=2 (608 obs) right son=3 (392 obs)
##   Primary splits:
##       reimbursement2008 < 1535 to the left,  improve=92.07519, (0 missing)
##       ihd               < 0.5  to the left,  improve=68.82295, (0 missing)
##       bucket2008        < 1.5  to the left,  improve=62.33394, (0 missing)
##       diabetes          < 0.5  to the left,  improve=50.11538, (0 missing)
##       heart.failure     < 0.5  to the left,  improve=44.93632, (0 missing)
##   Surrogate splits:
##       bucket2008    < 1.5  to the left,  agree=0.867, adj=0.661, (0 split)
##       ihd           < 0.5  to the left,  agree=0.798, adj=0.485, (0 split)
##       heart.failure < 0.5  to the left,  agree=0.774, adj=0.423, (0 split)
##       diabetes      < 0.5  to the left,  agree=0.771, adj=0.416, (0 split)
##       kidney        < 0.5  to the left,  agree=0.752, adj=0.367, (0 split)
## 
## Node number 2: 608 observations
##   predicted class=B1  expected loss=0.1217105  P(node) =0.608
##     class counts:   534    49    16     8     1
##    probabilities: 0.878 0.081 0.026 0.013 0.002 
## 
## Node number 3: 392 observations,    complexity param=0.04268293
##   predicted class=B2  expected loss=0.6403061  P(node) =0.392
##     class counts:   138   141    73    35     5
##    probabilities: 0.352 0.360 0.186 0.089 0.013 
##   left son=6 (237 obs) right son=7 (155 obs)
##   Primary splits:
##       reimbursement2008 < 5575 to the left,  improve=11.423200, (0 missing)
##       ihd               < 0.5  to the left,  improve= 7.993152, (0 missing)
##       bucket2008        < 2.5  to the left,  improve= 7.468367, (0 missing)
##       copd              < 0.5  to the left,  improve= 5.443367, (0 missing)
##       kidney            < 0.5  to the left,  improve= 4.488303, (0 missing)
##   Surrogate splits:
##       bucket2008 < 2.5  to the left,  agree=0.911, adj=0.774, (0 split)
##       kidney     < 0.5  to the left,  agree=0.719, adj=0.290, (0 split)
##       copd       < 0.5  to the left,  agree=0.717, adj=0.284, (0 split)
##       cancer     < 0.5  to the left,  agree=0.648, adj=0.110, (0 split)
##       alzheimers < 0.5  to the left,  agree=0.630, adj=0.065, (0 split)
## 
## Node number 6: 237 observations
##   predicted class=B1  expected loss=0.535865  P(node) =0.237
##     class counts:   110    85    26    16     0
##    probabilities: 0.464 0.359 0.110 0.068 0.000 
## 
## Node number 7: 155 observations,    complexity param=0.0304878
##   predicted class=B2  expected loss=0.6387097  P(node) =0.155
##     class counts:    28    56    47    19     5
##    probabilities: 0.181 0.361 0.303 0.123 0.032 
##   left son=14 (77 obs) right son=15 (78 obs)
##   Primary splits:
##       copd              < 0.5  to the left,  improve=3.044998, (0 missing)
##       reimbursement2008 < 9115 to the right, improve=2.747540, (0 missing)
##       ihd               < 0.5  to the left,  improve=1.568752, (0 missing)
##       kidney            < 0.5  to the left,  improve=1.289009, (0 missing)
##       bucket2008        < 2.5  to the right, improve=1.132028, (0 missing)
##   Surrogate splits:
##       reimbursement2008 < 7100 to the left,  agree=0.574, adj=0.143, (0 split)
##       age               < 87.5 to the right, agree=0.568, adj=0.130, (0 split)
##       diabetes          < 0.5  to the left,  agree=0.561, adj=0.117, (0 split)
##       bucket2008        < 3.5  to the left,  agree=0.561, adj=0.117, (0 split)
##       depression        < 0.5  to the left,  agree=0.548, adj=0.091, (0 split)
## 
## Node number 14: 77 observations
##   predicted class=B2  expected loss=0.5194805  P(node) =0.077
##     class counts:    13    37    18     8     1
##    probabilities: 0.169 0.481 0.234 0.104 0.013 
## 
## Node number 15: 78 observations
##   predicted class=B3  expected loss=0.6282051  P(node) =0.078
##     class counts:    15    19    29    11     4
##    probabilities: 0.192 0.244 0.372 0.141 0.051 
## 
## n= 1000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 1000 328 B1 (0.67 0.19 0.089 0.043 0.006)  
##    2) reimbursement2008< 1535 608  74 B1 (0.88 0.081 0.026 0.013 0.0016) *
##    3) reimbursement2008>=1535 392 251 B2 (0.35 0.36 0.19 0.089 0.013)  
##      6) reimbursement2008< 5575 237 127 B1 (0.46 0.36 0.11 0.068 0) *
##      7) reimbursement2008>=5575 155  99 B2 (0.18 0.36 0.3 0.12 0.032)  
##       14) copd< 0.5 77  40 B2 (0.17 0.48 0.23 0.1 0.013) *
##       15) copd>=0.5 78  49 B3 (0.19 0.24 0.37 0.14 0.051) *
```

```
## Warning: All NAs for inv.AIC.fit vs. auc.OOB scatter plot of glb_models_df
```

```
## Warning: All NAs for auc.OOB in auc.OOB vs. accuracy.fit scatter plot of
## glb_models_df
```

![](Medicare_2008_09_files/figure-html/run.models-13.png) 

```
## Warning: Removed 1 rows containing missing values (geom_point).
```

![](Medicare_2008_09_files/figure-html/run.models-14.png) 

```r
replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "model.selected")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0 
## 2.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction data.new.prediction 	firing:  model.selected 
## 3.0000 	 3 	 0 2 1 0
```

![](Medicare_2008_09_files/figure-html/run.models-15.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="fit.data.training.all", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                    chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed8            run.models                5                0  16.147
## elapsed9 fit.data.training.all                6                0  31.793
```

## Step `6`: fit.data.training.all

```r
print(mdl_feats_df <- myextract_mdl_feats( sel_mdl=glb_sel_mdl, 
                                           entity_df=glb_entity_df))
```

```
##                   importance                id fit.feat
## reimbursement2008 100.000000 reimbursement2008     TRUE
## ihd                73.776813               ihd     TRUE
## bucket2008         66.764284        bucket2008     TRUE
## diabetes           47.169226          diabetes     TRUE
## heart.failure      42.294628     heart.failure     TRUE
## copd                7.989356              copd     TRUE
## kidney              5.437679            kidney     TRUE
## age                 0.000000               age     TRUE
## alzheimers          0.000000        alzheimers     TRUE
## arthritis           0.000000         arthritis     TRUE
## cancer              0.000000            cancer     TRUE
## depression          0.000000        depression     TRUE
## osteoporosis        0.000000      osteoporosis     TRUE
## stroke              0.000000            stroke     TRUE
```

```r
#     ret_lst <- myrun_mdl_fn(indep_vars_vctr=indep_vars_vctr,
#                             rsp_var=glb_rsp_var, 
#                             rsp_var_out=glb_rsp_var_out,
#                             fit_df=glb_entity_df, 
#                             OOB_df=glb_newent_df,
#                             method=method, 
#                             tune_models_df=glb_tune_models_df,
#                             n_cv_folds=glb_n_cv_folds,
#                             loss_mtrx=glb_loss_mtrx,
#                             summaryFunction=glb_loss_smmry,
#                             metric="loss.error",
#                             maximize=FALSE)
ret_lst <- myrun_mdl_fn(indep_vars_vctr=mdl_feats_df$id,
                        rsp_var=glb_rsp_var, 
                        rsp_var_out=glb_rsp_var_out, 
                        fit_df=glb_entity_df,
                        method=glb_sel_mdl$method,
                        tune_models_df=glb_tune_models_df,
                        n_cv_folds=2, # glb_n_cv_folds,
                        loss_mtrx=glb_loss_mtrx,
                        summaryFunction=glb_sel_mdl$control$summaryFunction,
                        metric=glb_sel_mdl$metric,
                        maximize=glb_sel_mdl$maximize)
```

```
## + Fold1: cp=0 
## - Fold1: cp=0 
## + Fold2: cp=0 
## - Fold2: cp=0 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.01 on full training set
```

![](Medicare_2008_09_files/figure-html/fit.data.training.all_0-1.png) ![](Medicare_2008_09_files/figure-html/fit.data.training.all_0-2.png) 

```
## n= 1000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 1000 328 B1 (0.67 0.19 0.089 0.043 0.006)  
##    2) reimbursement2008< 1535 608  74 B1 (0.88 0.081 0.026 0.013 0.0016) *
##    3) reimbursement2008>=1535 392 251 B2 (0.35 0.36 0.19 0.089 0.013)  
##      6) reimbursement2008< 5575 237 127 B1 (0.46 0.36 0.11 0.068 0) *
##      7) reimbursement2008>=5575 155  99 B2 (0.18 0.36 0.3 0.12 0.032)  
##       14) copd< 0.5 77  40 B2 (0.17 0.48 0.23 0.1 0.013) *
##       15) copd>=0.5 78  49 B3 (0.19 0.24 0.37 0.14 0.051) *
```

```r
glb_fin_mdl <- ret_lst[["model"]]; 

glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="fit.data.training.all", 
    chunk_step_major=glb_script_df[nrow(glb_script_df), "chunk_step_major"], 
    chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                     chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed9  fit.data.training.all                6                0  31.793
## elapsed10 fit.data.training.all                6                1  35.975
```


```r
if (glb_is_regression) {
    glb_entity_df[, glb_rsp_var_out] <- predict(glb_fin_mdl, newdata=glb_entity_df)
    print(myplot_scatter(glb_entity_df, glb_rsp_var, glb_rsp_var_out, 
                         smooth=TRUE))
    glb_entity_df[, paste0(glb_rsp_var_out, ".err")] <- 
        abs(glb_entity_df[, glb_rsp_var_out] - glb_entity_df[, glb_rsp_var])
    print(head(orderBy(reformulate(c("-", paste0(glb_rsp_var_out, ".err"))), 
                       glb_entity_df)))                             
}    

if (glb_is_classification & glb_is_binomial) {
            if (any(class(glb_fin_mdl) %in% c("train"))) {
        glb_entity_df[, paste0(glb_rsp_var_out, ".proba")] <- 
            predict(glb_fin_mdl, newdata=glb_entity_df, type="prob")[, 2]
    } else  if (any(class(glb_fin_mdl) %in% c("rpart", "randomForest"))) {
        glb_entity_df[, paste0(glb_rsp_var_out, ".proba")] <- 
            predict(glb_fin_mdl, newdata=glb_entity_df, type="prob")[, 2]
    } else  if (class(glb_fin_mdl) == "glm") {
        stop("not implemented yet")
        glb_entity_df[, paste0(glb_rsp_var_out, ".proba")] <- 
            predict(glb_fin_mdl, newdata=glb_entity_df, type="response")
    } else  stop("not implemented yet")   

    require(ROCR)
    ROCRpred <- prediction(glb_entity_df[, paste0(glb_rsp_var_out, ".proba")],
                           glb_entity_df[, glb_rsp_var])
    ROCRperf <- performance(ROCRpred, "tpr", "fpr")
    plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0, 1, 0.1), text.adj=c(-0.2,1.7))
    
    thresholds_df <- data.frame(threshold=seq(0.0, 1.0, 0.1))
    thresholds_df$f.score <- sapply(1:nrow(thresholds_df), function(row_ix) 
        mycompute_classifier_f.score(mdl=glb_fin_mdl, obs_df=glb_entity_df, 
                                     proba_threshold=thresholds_df[row_ix, "threshold"], 
                                      rsp_var=glb_rsp_var, 
                                      rsp_var_out=glb_rsp_var_out))
    print(thresholds_df)
    print(myplot_line(thresholds_df, "threshold", "f.score"))
    
    proba_threshold <- thresholds_df[which.max(thresholds_df$f.score), 
                                             "threshold"]
    # This should change to maximize f.score.OOB ???
    print(sprintf("Classifier Probability Threshold: %0.4f to maximize f.score.fit",
                  proba_threshold))
    if (is.null(glb_clf_proba_threshold)) 
        glb_clf_proba_threshold <- proba_threshold else {
        print(sprintf("Classifier Probability Threshold: %0.4f per user specs",
                      glb_clf_proba_threshold))
    }

    if ((class(glb_entity_df[, glb_rsp_var]) != "factor") | 
    	(length(levels(glb_entity_df[, glb_rsp_var])) != 2))
		stop("expecting a factor with two levels:", glb_rsp_var)
	glb_entity_df[, glb_rsp_var_out] <- 
		factor(levels(glb_entity_df[, glb_rsp_var])[
			(glb_entity_df[, paste0(glb_rsp_var_out, ".proba")] >= 
				glb_clf_proba_threshold) * 1 + 1])
             
    print(mycreate_xtab(glb_entity_df, c(glb_rsp_var, glb_rsp_var_out)))
    print(sprintf("f.score=%0.4f", 
        mycompute_classifier_f.score(glb_fin_mdl, glb_entity_df, 
                                     glb_clf_proba_threshold, 
                                     glb_rsp_var, glb_rsp_var_out)))    
}

if (glb_is_classification & !glb_is_binomial) {
    glb_entity_df[, glb_rsp_var_out] <- predict(glb_fin_mdl, newdata=glb_entity_df, type="raw")
}    

print(glb_feats_df <- mymerge_feats_importance(glb_feats_df, glb_fin_mdl, glb_entity_df))
```

```
##                   id      cor.y  cor.y.abs cor.low importance
## 13 reimbursement2008 0.39864605 0.39864605      NA 100.000000
## 10               ihd 0.41799716 0.41799716       1  73.776813
## 4         bucket2008 0.44655074 0.44655074       1  66.764284
## 8           diabetes 0.40363086 0.40363086       1  47.169226
## 9      heart.failure 0.36864613 0.36864613       1  42.294628
## 6               copd 0.33705795 0.33705795       1   7.989356
## 11            kidney 0.38762968 0.38762968       1   5.437679
## 1                age 0.03524295 0.03524295       1   0.000000
## 2         alzheimers 0.29777179 0.29777179       1   0.000000
## 3          arthritis 0.25517293 0.25517293       1   0.000000
## 5             cancer 0.18571084 0.18571084       1   0.000000
## 7         depression 0.27092271 0.27092271       1   0.000000
## 12      osteoporosis 0.18701176 0.18701176       1   0.000000
## 14            stroke 0.19269419 0.19269419       1   0.000000
```

```r
# Most of this code is used again in predict.data.new chunk
glb_analytics_diag_plots <- function(obs_df) {
    for (var in subset(glb_feats_df, !is.na(importance))$id) {
        plot_df <- melt(obs_df, id.vars=var, 
                        measure.vars=c(glb_rsp_var, glb_rsp_var_out))
#         if (var == "<feat_name>") print(myplot_scatter(plot_df, var, "value", 
#                                              facet_colcol_name="variable") + 
#                       geom_vline(xintercept=<divider_val>, linetype="dotted")) else     
            print(myplot_scatter(plot_df, var, "value", 
                                 facet_colcol_name="variable", jitter=TRUE))
    }
    
    if (glb_is_regression) {
        plot_vars_df <- subset(glb_feats_df, Pr.z < 0.1)
        print(myplot_prediction_regression(obs_df, 
                    ifelse(nrow(plot_vars_df) > 1, plot_vars_df$id[2], ".rownames"), 
                                           plot_vars_df$id[1],
                    glb_rsp_var, glb_rsp_var_out)
#               + facet_wrap(reformulate(plot_vars_df$id[2])) # if [1,2] is a factor                                                         
#               + geom_point(aes_string(color="<col_name>.fctr")) #  to color the plot
              )
    }    
    
    if (glb_is_classification) {
        if (nrow(plot_vars_df <- subset(glb_feats_df, !is.na(importance))) == 0)
            warning("No features in selected model are statistically important")
        else print(myplot_prediction_classification(df=obs_df, 
                feat_x=ifelse(nrow(plot_vars_df) > 1, plot_vars_df$id[2], 
                              ".rownames"),
                                               feat_y=plot_vars_df$id[1],
                     rsp_var=glb_rsp_var, 
                     rsp_var_out=glb_rsp_var_out, 
                     id_vars=glb_id_vars)
#               + geom_hline(yintercept=<divider_val>, linetype = "dotted")
                )
    }    
}
glb_analytics_diag_plots(obs_df=glb_entity_df)
```

![](Medicare_2008_09_files/figure-html/fit.data.training.all_1-1.png) ![](Medicare_2008_09_files/figure-html/fit.data.training.all_1-2.png) ![](Medicare_2008_09_files/figure-html/fit.data.training.all_1-3.png) ![](Medicare_2008_09_files/figure-html/fit.data.training.all_1-4.png) ![](Medicare_2008_09_files/figure-html/fit.data.training.all_1-5.png) ![](Medicare_2008_09_files/figure-html/fit.data.training.all_1-6.png) ![](Medicare_2008_09_files/figure-html/fit.data.training.all_1-7.png) ![](Medicare_2008_09_files/figure-html/fit.data.training.all_1-8.png) ![](Medicare_2008_09_files/figure-html/fit.data.training.all_1-9.png) ![](Medicare_2008_09_files/figure-html/fit.data.training.all_1-10.png) ![](Medicare_2008_09_files/figure-html/fit.data.training.all_1-11.png) ![](Medicare_2008_09_files/figure-html/fit.data.training.all_1-12.png) ![](Medicare_2008_09_files/figure-html/fit.data.training.all_1-13.png) ![](Medicare_2008_09_files/figure-html/fit.data.training.all_1-14.png) 

```
##        age alzheimers arthritis cancer copd depression diabetes
## 199     73          0         0      0    0          0        0
## 84011   70          0         0      0    0          0        0
## 141049  99          0         1      0    0          0        1
## 154439  70          0         0      0    1          0        1
## 161619  77          0         0      0    1          0        1
## 179115  85          1         0      0    0          0        1
## 184695  70          1         0      0    0          1        1
## 241349  72          1         0      0    1          1        0
## 243393  72          0         0      0    0          0        1
## 245435  75          0         1      0    0          0        1
## 248052  68          1         0      0    1          0        1
## 277735  79          1         1      1    1          1        1
## 297440  72          0         1      0    0          0        0
## 311429  80          0         1      0    0          0        1
## 312790  75          0         0      0    0          0        0
## 338073  78          0         0      0    0          0        0
## 415851  28          1         1      1    1          1        1
## 451715  84          0         0      0    0          0        0
## 452378  42          0         0      0    0          0        0
##        heart.failure ihd kidney osteoporosis stroke reimbursement2008
## 199                0   0      0            0      0                 0
## 84011              0   1      0            0      0                40
## 141049             0   0      0            0      0              6180
## 154439             0   1      1            0      0             58230
## 161619             0   0      0            0      0             21620
## 179115             1   1      1            0      0             22830
## 184695             0   0      1            0      0              5630
## 241349             1   1      0            0      0             28370
## 243393             1   1      1            1      0              8730
## 245435             0   1      1            1      0             12840
## 248052             1   1      1            0      0              9640
## 277735             0   0      0            0      0             12590
## 297440             1   0      1            1      0              6160
## 311429             0   0      0            1      0               690
## 312790             0   0      0            0      0                 0
## 338073             0   0      0            0      0                 0
## 415851             1   1      0            0      0             81390
## 451715             0   0      0            0      0                 0
## 452378             0   0      0            0      0                 0
##        bucket2008 reimbursement2009 bucket2009      .rnorm bucket2009.fctr
## 199             1                 0          1  0.86235379              B1
## 84011           1                 0          1  0.21639181              B1
## 141049          2               400          1 -0.93199434              B1
## 154439          5               570          1  2.22730511              B1
## 161619          4               660          1 -0.23544580              B1
## 179115          4               880          1 -0.13299033              B1
## 184695          2               950          1  0.51717887              B1
## 241349          4              1710          1 -0.36912210              B1
## 243393          3              1740          1  0.11663450              B1
## 245435          3              1770          1 -0.12437944              B1
## 248052          3              1810          1  0.48365742              B1
## 277735          3              2320          1  0.05150359              B1
## 297440          2              2740          1 -1.47754310              B1
## 311429          1              3110          2  0.72098051              B2
## 312790          1              3150          2 -0.01316559              B2
## 338073          1              4000          2  1.59510180              B2
## 415851          5             11980          3 -0.66424699              B3
## 451715          1             38250          4 -1.37266004              B4
## 452378          1             40180          4  0.29171516              B4
##        bucket2009.fctr.prediction bucket2009.fctr.prediction.accurate
## 199                            B1                                TRUE
## 84011                          B1                                TRUE
## 141049                         B2                               FALSE
## 154439                         B3                               FALSE
## 161619                         B3                               FALSE
## 179115                         B2                               FALSE
## 184695                         B2                               FALSE
## 241349                         B3                               FALSE
## 243393                         B2                               FALSE
## 245435                         B2                               FALSE
## 248052                         B3                               FALSE
## 277735                         B3                               FALSE
## 297440                         B2                               FALSE
## 311429                         B1                               FALSE
## 312790                         B1                               FALSE
## 338073                         B1                               FALSE
## 415851                         B3                                TRUE
## 451715                         B1                               FALSE
## 452378                         B1                               FALSE
##         .label
## 199       .199
## 84011   .84011
## 141049 .141049
## 154439 .154439
## 161619 .161619
## 179115 .179115
## 184695 .184695
## 241349 .241349
## 243393 .243393
## 245435 .245435
## 248052 .248052
## 277735 .277735
## 297440 .297440
## 311429 .311429
## 312790 .312790
## 338073 .338073
## 415851 .415851
## 451715 .451715
## 452378 .452378
```

![](Medicare_2008_09_files/figure-html/fit.data.training.all_1-15.png) 

```r
replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "data.training.all.prediction","model.final")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0 
## 2.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction data.new.prediction 	firing:  model.selected 
## 3.0000 	 3 	 0 2 1 0 
## 3.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  data.training.all.prediction 
## 4.0000 	 5 	 0 1 1 1 
## 4.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  model.final 
## 5.0000 	 4 	 0 0 2 1
```

![](Medicare_2008_09_files/figure-html/fit.data.training.all_1-16.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="predict.data.new", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                     chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed10 fit.data.training.all                6                1  35.975
## elapsed11      predict.data.new                7                0  46.259
```

## Step `7`: predict data.new

```r
if (glb_is_regression)
    glb_newent_df[, glb_rsp_var_out] <- predict(glb_fin_mdl, 
                                        newdata=glb_newent_df, type="response")

if (glb_is_classification & glb_is_binomial) {
    # Compute selected model predictions
            if (any(class(glb_fin_mdl) %in% c("train"))) {
        glb_newent_df[, paste0(glb_rsp_var_out, ".proba")] <- 
            predict(glb_fin_mdl, newdata=glb_newent_df, type="prob")[, 2]
    } else  if (any(class(glb_fin_mdl) %in% c("rpart", "randomForest"))) {
        glb_newent_df[, paste0(glb_rsp_var_out, ".proba")] <- 
            predict(glb_fin_mdl, newdata=glb_newent_df, type="prob")[, 2]
    } else  if (class(glb_fin_mdl) == "glm") {
        stop("not implemented yet")
        glb_newent_df[, paste0(glb_rsp_var_out, ".proba")] <- 
            predict(glb_fin_mdl, newdata=glb_newent_df, type="response")
    } else  stop("not implemented yet")   

    if ((class(glb_newent_df[, glb_rsp_var]) != "factor") | 
		(length(levels(glb_newent_df[, glb_rsp_var])) != 2))
		stop("expecting a factor with two levels:", glb_rsp_var)
	glb_newent_df[, glb_rsp_var_out] <- 
		factor(levels(glb_newent_df[, glb_rsp_var])[
			(glb_newent_df[, paste0(glb_rsp_var_out, ".proba")] >= 
				glb_clf_proba_threshold) * 1 + 1])

    # Compute dummy model predictions
    glb_newent_df[, paste0(glb_rsp_var, ".predictdmy.proba")] <- 
        predict(glb_dmy_mdl, newdata=glb_newent_df, type="prob")[, 2]
    if ((class(glb_newent_df[, glb_rsp_var]) != "factor") | 
    	(length(levels(glb_newent_df[, glb_rsp_var])) != 2))
		stop("expecting a factor with two levels:", glb_rsp_var)
	glb_newent_df[, paste0(glb_rsp_var, ".predictdmy")] <- 
		factor(levels(glb_newent_df[, glb_rsp_var])[
			(glb_newent_df[, paste0(glb_rsp_var, ".predictdmy.proba")] >= 
				glb_clf_proba_threshold) * 1 + 1])
}

if (glb_is_classification & !glb_is_binomial) {
    # Compute baseline predictions
    if (!is.null(glb_bsl_mdl_var)) {
        if (is.null(glb_map_rsp_raw_to_var)) {
        glb_newent_df[, paste0(glb_rsp_var, ".predictbsl")] <- 
                   glb_newent_df[, glb_bsl_mdl_var]
        } else {
        glb_newent_df[, paste0(glb_rsp_var, ".predictbsl")] <- 
                   glb_map_rsp_raw_to_var(glb_newent_df[, glb_bsl_mdl_var])
        }
    }    

    # Compute most frequent outcome predictions
    glb_newent_df[, paste0(glb_rsp_var, ".predictmfo")] <- 
        as.factor(names(sort(table(glb_entity_df[, glb_rsp_var]), decreasing=TRUE))[1])

    # Compute dummy model predictions - different from most frequent outcome - shd be from runif
    glb_newent_df[, paste0(glb_rsp_var, ".predictdmy")] <- 
        predict(glb_dmy_mdl, newdata=glb_newent_df, type="raw")

    # Compute selected_no_loss_matrix model predictions
    glb_newent_df[, paste0(glb_rsp_var, ".predictnlm")] <- 
        predict(glb_sel_nlm_mdl, newdata=glb_newent_df, type="raw")

    # Compute final model predictions
	glb_newent_df[, glb_rsp_var_out] <- predict(glb_fin_mdl, newdata=glb_newent_df, type="raw")
}
    
myprint_df(glb_newent_df[, c(glb_id_vars, glb_rsp_var, glb_rsp_var_out)])
```

```
##      bucket2009.fctr bucket2009.fctr.prediction
## 1332              B1                         B1
## 2225              B1                         B1
## 3084              B1                         B1
## 3705              B1                         B1
## 4007              B1                         B1
## 4140              B1                         B1
##        bucket2009.fctr bucket2009.fctr.prediction
## 165676              B1                         B1
## 306893              B1                         B1
## 338121              B2                         B1
## 353495              B2                         B1
## 414126              B3                         B1
## 416488              B3                         B1
##        bucket2009.fctr bucket2009.fctr.prediction
## 455632              B5                         B1
## 455867              B5                         B1
## 455904              B5                         B1
## 456279              B5                         B1
## 456352              B5                         B1
## 456354              B5                         B1
```

```r
if (glb_is_regression) {
    print(sprintf("Total SSE: %0.4f", 
                  sum((glb_newent_df[, glb_rsp_var_out] - 
                        glb_newent_df[, glb_rsp_var]) ^ 2)))
    print(sprintf("RMSE: %0.4f", 
                  (sum((glb_newent_df[, glb_rsp_var_out] - 
                        glb_newent_df[, glb_rsp_var]) ^ 2) / nrow(glb_newent_df)) ^ 0.5))                        
    print(myplot_scatter(glb_newent_df, glb_rsp_var, glb_rsp_var_out, 
                         smooth=TRUE))
                         
    glb_newent_df[, paste0(glb_rsp_var_out, ".err")] <- 
        abs(glb_newent_df[, glb_rsp_var_out] - glb_newent_df[, glb_rsp_var])
    print(head(orderBy(reformulate(c("-", paste0(glb_rsp_var_out, ".err"))), 
                       glb_newent_df)))                                                      

#     glb_newent_df[, "<Output Pred variable>"] <- func(glb_newent_df[, glb_pred_var_name])                         
}                         

if (glb_is_classification & glb_is_binomial) {
    ROCRpred <- prediction(glb_newent_df[, paste0(glb_rsp_var_out, ".proba")],
                           glb_newent_df[, glb_rsp_var])
    print(sprintf("auc=%0.4f", auc <- as.numeric(performance(ROCRpred, "auc")@y.values)))   
    
    print(sprintf("probability threshold=%0.4f", glb_clf_proba_threshold))
    print(newent_conf_df <- mycreate_xtab(glb_newent_df, 
                                        c(glb_rsp_var, glb_rsp_var_out)))
    print(sprintf("f.score.sel=%0.4f", 
        mycompute_classifier_f.score(mdl=glb_fin_mdl, obs_df=glb_newent_df, 
                                     proba_threshold=glb_clf_proba_threshold, 
                                      rsp_var=glb_rsp_var, 
                                      rsp_var_out=glb_rsp_var_out)))
    print(sprintf("sensitivity=%0.4f", newent_conf_df[2, 3] / 
                      (newent_conf_df[2, 3] + newent_conf_df[2, 2])))
    print(sprintf("specificity=%0.4f", newent_conf_df[1, 2] / 
                      (newent_conf_df[1, 2] + newent_conf_df[1, 3])))
    print(sprintf("accuracy=%0.4f", (newent_conf_df[1, 2] + newent_conf_df[2, 3]) / 
                      (newent_conf_df[1, 2] + newent_conf_df[2, 3] + 
                       newent_conf_df[1, 3] + newent_conf_df[2, 2])))
    
    print(mycreate_xtab(glb_newent_df, c(glb_rsp_var, paste0(glb_rsp_var, ".predictdmy"))))
    print(sprintf("f.score.dmy=%0.4f", 
        mycompute_classifier_f.score(mdl=glb_dmy_mdl, obs_df=glb_newent_df, 
                                     proba_threshold=glb_clf_proba_threshold, 
                                      rsp_var=glb_rsp_var, 
                                      rsp_var_out=paste0(glb_rsp_var, ".predictdmy"))))
}    

if (glb_is_classification & !glb_is_binomial) {
    mycompute_prediction_quality <- function(mdl_type) {    
        print(sprintf("prediction model type: %s", mdl_type))
        print(newent_conf_df <- mycreate_xtab(glb_newent_df, 
            c(glb_rsp_var, 
              ifelse(mdl_type == "fin", glb_rsp_var_out, 
                     paste0(glb_rsp_var, ".predict", mdl_type)))))
        newent_conf_df[is.na(newent_conf_df)] <- 0    
        newent_conf_mtrx <- as.matrix(newent_conf_df[, -1])
        newent_conf_mtrx <- cbind(newent_conf_mtrx, 
            matrix(rep(0, 
                (nrow(newent_conf_mtrx) - ncol(newent_conf_mtrx)) * nrow(newent_conf_mtrx)), 
                    byrow=TRUE, nrow=nrow(newent_conf_mtrx)))
        return(data.frame(mdl_type=mdl_type, 
            accuracy=sum(diag(newent_conf_mtrx)) * 1.0 / sum(newent_conf_mtrx),    
            loss.error=sum(newent_conf_mtrx * glb_loss_mtrx) / nrow(glb_newent_df)))
    }

    predict_qlty_df <- data.frame()
    for (type in c("bsl", "mfo", "dmy", "nlm", "fin"))
        predict_qlty_df <- rbind(predict_qlty_df, mycompute_prediction_quality(type))
    print(predict_qlty_df)
    
    predict_qlty_df$inv.loss.error <- 1.0 / predict_qlty_df$loss.error
    print(myplot_scatter(predict_qlty_df, "accuracy", "inv.loss.error", 
                         colorcol_name="mdl_type"))
}    
```

```
## [1] "prediction model type: bsl"
##   bucket2009.fctr bucket2009.fctr.predictbsl.B1
## 1              B1                           594
## 2              B2                            92
## 3              B3                            48
## 4              B4                            15
## 5              B5                             3
##   bucket2009.fctr.predictbsl.B2 bucket2009.fctr.predictbsl.B3
## 1                            49                            21
## 2                            62                            22
## 3                            19                             9
## 4                            10                             4
## 5                             3                            NA
##   bucket2009.fctr.predictbsl.B4 bucket2009.fctr.predictbsl.B5
## 1                             8                            NA
## 2                            14                            NA
## 3                            11                             2
## 4                             9                             5
## 5                            NA                            NA
## [1] "prediction model type: mfo"
##   bucket2009.fctr bucket2009.fctr.predictmfo.B1
## 1              B1                           672
## 2              B2                           190
## 3              B3                            89
## 4              B4                            43
## 5              B5                             6
## [1] "prediction model type: dmy"
##   bucket2009.fctr bucket2009.fctr.predictdmy.B1
## 1              B1                           672
## 2              B2                           190
## 3              B3                            89
## 4              B4                            43
## 5              B5                             6
## [1] "prediction model type: nlm"
##   bucket2009.fctr bucket2009.fctr.predictnlm.B1
## 1              B1                           633
## 2              B2                           143
## 3              B3                            63
## 4              B4                            22
## 5              B5                             6
##   bucket2009.fctr.predictnlm.B2 bucket2009.fctr.predictnlm.B3
## 1                            22                            17
## 2                            27                            20
## 3                            11                            15
## 4                             7                            14
## 5                            NA                            NA
## [1] "prediction model type: fin"
##   bucket2009.fctr bucket2009.fctr.prediction.B1
## 1              B1                           633
## 2              B2                           143
## 3              B3                            63
## 4              B4                            22
## 5              B5                             6
##   bucket2009.fctr.prediction.B2 bucket2009.fctr.prediction.B3
## 1                            22                            17
## 2                            27                            20
## 3                            11                            15
## 4                             7                            14
## 5                            NA                            NA
##   mdl_type accuracy loss.error
## 1      bsl    0.674      0.779
## 2      mfo    0.672      1.042
## 3      dmy    0.672      1.042
## 4      nlm    0.675      0.872
## 5      fin    0.675      0.872
```

![](Medicare_2008_09_files/figure-html/predict.data.new-1.png) 

```r
glb_analytics_diag_plots(obs_df=glb_newent_df)
```

![](Medicare_2008_09_files/figure-html/predict.data.new-2.png) ![](Medicare_2008_09_files/figure-html/predict.data.new-3.png) ![](Medicare_2008_09_files/figure-html/predict.data.new-4.png) ![](Medicare_2008_09_files/figure-html/predict.data.new-5.png) ![](Medicare_2008_09_files/figure-html/predict.data.new-6.png) ![](Medicare_2008_09_files/figure-html/predict.data.new-7.png) ![](Medicare_2008_09_files/figure-html/predict.data.new-8.png) ![](Medicare_2008_09_files/figure-html/predict.data.new-9.png) ![](Medicare_2008_09_files/figure-html/predict.data.new-10.png) ![](Medicare_2008_09_files/figure-html/predict.data.new-11.png) ![](Medicare_2008_09_files/figure-html/predict.data.new-12.png) ![](Medicare_2008_09_files/figure-html/predict.data.new-13.png) ![](Medicare_2008_09_files/figure-html/predict.data.new-14.png) ![](Medicare_2008_09_files/figure-html/predict.data.new-15.png) 

```
##        age alzheimers arthritis cancer copd depression diabetes
## 1332    95          0         0      0    0          0        0
## 86213   75          0         0      0    0          0        0
## 90317   79          0         1      1    0          1        1
## 126170  67          0         0      1    0          0        0
## 159207  80          0         0      0    1          0        0
## 159225  79          0         0      0    0          1        1
##        heart.failure ihd kidney osteoporosis stroke reimbursement2008
## 1332               0   0      0            0      0                 0
## 86213              0   1      0            0      0               100
## 90317              1   1      0            0      1             11370
## 126170             0   0      0            0      0             17360
## 159207             0   0      1            0      0              9170
## 159225             0   1      0            0      0             17090
##        bucket2008 reimbursement2009 bucket2009     .rnorm bucket2009.fctr
## 1332            1                 0          1  0.6003069              B1
## 86213           1                 0          1  0.1059092              B1
## 90317           3                 0          1  1.3148007              B1
## 126170          3               220          1 -0.1498292              B1
## 159207          3               630          1 -0.9033867              B1
## 159225          3               630          1 -1.2668954              B1
##        bucket2009.fctr.predictbsl bucket2009.fctr.predictmfo
## 1332                           B1                         B1
## 86213                          B1                         B1
## 90317                          B3                         B1
## 126170                         B3                         B1
## 159207                         B3                         B1
## 159225                         B3                         B1
##        bucket2009.fctr.predictdmy bucket2009.fctr.predictnlm
## 1332                           B1                         B1
## 86213                          B1                         B1
## 90317                          B1                         B2
## 126170                         B1                         B2
## 159207                         B1                         B3
## 159225                         B1                         B2
##        bucket2009.fctr.prediction bucket2009.fctr.prediction.accurate
## 1332                           B1                                TRUE
## 86213                          B1                                TRUE
## 90317                          B2                               FALSE
## 126170                         B2                               FALSE
## 159207                         B3                               FALSE
## 159225                         B2                               FALSE
##         .label
## 1332     .1332
## 86213   .86213
## 90317   .90317
## 126170 .126170
## 159207 .159207
## 159225 .159225
##        age alzheimers arthritis cancer copd depression diabetes
## 1332    95          0         0      0    0          0        0
## 90317   79          0         1      1    0          1        1
## 159225  79          0         0      0    0          1        1
## 207371  77          0         0      0    0          0        1
## 216400  58          0         0      0    0          1        0
## 361457  50          0         0      0    0          0        0
##        heart.failure ihd kidney osteoporosis stroke reimbursement2008
## 1332               0   0      0            0      0                 0
## 90317              1   1      0            0      1             11370
## 159225             0   1      0            0      0             17090
## 207371             1   0      1            1      1              7990
## 216400             1   1      0            0      0             12820
## 361457             0   0      0            0      0                 0
##        bucket2008 reimbursement2009 bucket2009     .rnorm bucket2009.fctr
## 1332            1                 0          1  0.6003069              B1
## 90317           3                 0          1  1.3148007              B1
## 159225          3               630          1 -1.2668954              B1
## 207371          2              1240          1 -0.7268655              B1
## 216400          3              1360          1 -0.9557674              B1
## 361457          1              5110          2 -0.2239145              B2
##        bucket2009.fctr.predictbsl bucket2009.fctr.predictmfo
## 1332                           B1                         B1
## 90317                          B3                         B1
## 159225                         B3                         B1
## 207371                         B2                         B1
## 216400                         B3                         B1
## 361457                         B1                         B1
##        bucket2009.fctr.predictdmy bucket2009.fctr.predictnlm
## 1332                           B1                         B1
## 90317                          B1                         B2
## 159225                         B1                         B2
## 207371                         B1                         B2
## 216400                         B1                         B2
## 361457                         B1                         B1
##        bucket2009.fctr.prediction bucket2009.fctr.prediction.accurate
## 1332                           B1                                TRUE
## 90317                          B2                               FALSE
## 159225                         B2                               FALSE
## 207371                         B2                               FALSE
## 216400                         B2                               FALSE
## 361457                         B1                               FALSE
##         .label
## 1332     .1332
## 90317   .90317
## 159225 .159225
## 207371 .207371
## 216400 .216400
## 361457 .361457
##        age alzheimers arthritis cancer copd depression diabetes
## 350328  87          0         0      0    0          0        0
## 361457  50          0         0      0    0          0        0
## 376325  69          0         0      0    0          0        0
## 412626  67          0         0      0    0          0        0
## 446665  40          0         0      0    0          0        0
## 452119  89          1         1      1    1          1        1
##        heart.failure ihd kidney osteoporosis stroke reimbursement2008
## 350328             0   0      0            0      0                 0
## 361457             0   0      0            0      0                 0
## 376325             0   0      0            0      0                 0
## 412626             0   0      0            0      0                 0
## 446665             0   0      0            0      0                 0
## 452119             1   1      1            1      1            128070
##        bucket2008 reimbursement2009 bucket2009     .rnorm bucket2009.fctr
## 350328          1              4530          2 -0.7489170              B2
## 361457          1              5110          2 -0.2239145              B2
## 376325          1              6120          2  0.2477779              B2
## 412626          1             11240          3 -1.2193626              B3
## 446665          1             28710          4 -0.6472890              B4
## 452119          5             39380          4  0.6636506              B4
##        bucket2009.fctr.predictbsl bucket2009.fctr.predictmfo
## 350328                         B1                         B1
## 361457                         B1                         B1
## 376325                         B1                         B1
## 412626                         B1                         B1
## 446665                         B1                         B1
## 452119                         B5                         B1
##        bucket2009.fctr.predictdmy bucket2009.fctr.predictnlm
## 350328                         B1                         B1
## 361457                         B1                         B1
## 376325                         B1                         B1
## 412626                         B1                         B1
## 446665                         B1                         B1
## 452119                         B1                         B3
##        bucket2009.fctr.prediction bucket2009.fctr.prediction.accurate
## 350328                         B1                               FALSE
## 361457                         B1                               FALSE
## 376325                         B1                               FALSE
## 412626                         B1                               FALSE
## 446665                         B1                               FALSE
## 452119                         B3                               FALSE
##         .label
## 350328 .350328
## 361457 .361457
## 376325 .376325
## 412626 .412626
## 446665 .446665
## 452119 .452119
```

![](Medicare_2008_09_files/figure-html/predict.data.new-16.png) 

```r
tmp_replay_lst <- replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "data.new.prediction")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0 
## 2.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction data.new.prediction 	firing:  model.selected 
## 3.0000 	 3 	 0 2 1 0 
## 3.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  data.training.all.prediction 
## 4.0000 	 5 	 0 1 1 1 
## 4.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  model.final 
## 5.0000 	 4 	 0 0 2 1 
## 6.0000 	 6 	 0 0 1 2
```

![](Medicare_2008_09_files/figure-html/predict.data.new-17.png) 

```r
#print(ggplot.petrinet(tmp_replay_lst[["pn"]]) + coord_flip())
```

Null Hypothesis ($\sf{H_{0}}$): mpg is not impacted by am_fctr.  
The variance by am_fctr appears to be independent. 

```r
# print(t.test(subset(cars_df, am_fctr == "automatic")$mpg, 
#              subset(cars_df, am_fctr == "manual")$mpg, 
#              var.equal=FALSE)$conf)
```
We reject the null hypothesis i.e. we have evidence to conclude that am_fctr impacts mpg (95% confidence). Manual transmission is better for miles per gallon versus automatic transmission.


```
##                   chunk_label chunk_step_major chunk_step_minor elapsed
## 10      fit.data.training.all                6                0  31.793
## 12           predict.data.new                7                0  46.259
## 2                cleanse_data                2                0   7.395
## 11      fit.data.training.all                6                1  35.975
## 6            extract_features                3                0  13.454
## 7             select_features                4                0  15.069
## 4         manage_missing_data                2                2   8.898
## 9                  run.models                5                0  16.147
## 5          encode_retype_data                2                3   9.336
## 8  remove_correlated_features                4                1  15.272
## 3       inspectORexplore.data                2                1   7.429
## 1                 import_data                1                0   0.003
##    elapsed_diff
## 10       15.646
## 12       10.284
## 2         7.392
## 11        4.182
## 6         4.118
## 7         1.615
## 4         1.469
## 9         0.875
## 5         0.438
## 8         0.203
## 3         0.034
## 1         0.000
```

```
## [1] "Total Elapsed Time: 46.259 secs"
```

![](Medicare_2008_09_files/figure-html/print_sessionInfo-1.png) 

```
## R version 3.1.3 (2015-03-09)
## Platform: x86_64-apple-darwin13.4.0 (64-bit)
## Running under: OS X 10.10.2 (Yosemite)
## 
## locale:
## [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
## 
## attached base packages:
## [1] tcltk     grid      stats     graphics  grDevices utils     datasets 
## [8] methods   base     
## 
## other attached packages:
##  [1] rpart.plot_1.5.2 rpart_4.1-9      caret_6.0-41     lattice_0.20-31 
##  [5] ROCR_1.0-7       gplots_2.16.0    reshape2_1.4.1   sqldf_0.4-10    
##  [9] RSQLite_1.0.0    DBI_0.3.1        gsubfn_0.6-6     proto_0.3-10    
## [13] plyr_1.8.1       caTools_1.17.1   doBy_4.5-13      survival_2.38-1 
## [17] ggplot2_1.0.1   
## 
## loaded via a namespace (and not attached):
##  [1] bitops_1.0-6        BradleyTerry2_1.0-6 brglm_0.5-9        
##  [4] car_2.0-25          chron_2.3-45        class_7.3-12       
##  [7] codetools_0.2-11    colorspace_1.2-6    compiler_3.1.3     
## [10] digest_0.6.8        e1071_1.6-4         evaluate_0.5.5     
## [13] foreach_1.4.2       formatR_1.1         gdata_2.13.3       
## [16] gtable_0.1.2        gtools_3.4.1        htmltools_0.2.6    
## [19] iterators_1.0.7     KernSmooth_2.23-14  knitr_1.9          
## [22] labeling_0.3        lme4_1.1-7          MASS_7.3-40        
## [25] Matrix_1.2-0        mgcv_1.8-6          minqa_1.2.4        
## [28] munsell_0.4.2       nlme_3.1-120        nloptr_1.0.4       
## [31] nnet_7.3-9          parallel_3.1.3      pbkrtest_0.4-2     
## [34] quantreg_5.11       Rcpp_0.11.5         rmarkdown_0.5.1    
## [37] scales_0.2.4        SparseM_1.6         splines_3.1.3      
## [40] stringr_0.6.2       tools_3.1.3         yaml_2.1.13
```
