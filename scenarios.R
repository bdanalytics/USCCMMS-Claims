model_id_pfx <- "Baseline.tst";     model_method <- "mybaseln_classfr"; indep_vars_vctr <- glb_Baseline_mdl_var
model_id_pfx <- "MFO.tst";          model_method <- "myMFO_classfr";    indep_vars_vctr <- ".rnorm"
model_id_pfx <- "Max.cor.Y.tst";    model_method <- "rpart";            indep_vars_vctr <- max_cor_y_x_var
model_id_pfx <- "Interact.High.cor.y.tst";    model_method <- "rpart";
model_id_pfx <- "Low.cor.X.tst";    model_method <- "rpart";            indep_vars_vctr <- subset(glb_feats_df, cor.low == 1)[, "id"]

model_id_pfx <- "All.X.tst";    model_method <- "rpart";    indep_vars_vctr <- setdiff(names(glb_entity_df), union(glb_rsp_var, glb_exclude_vars_as_features))
ret_lst <- myfit_mdl_fn(model_id=paste0(model_id_pfx, ".cv.0.lser.no"), model_method=model_method,
                        indep_vars_vctr=indep_vars_vctr,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_entity_df, OOB_df=glb_newent_df)
ret_lst <- myfit_mdl_fn(model_id=paste0(model_id_pfx, ".cv.G.lser.no"), model_method=model_method,
                        indep_vars_vctr=indep_vars_vctr,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_entity_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=glb_tune_models_df)
ret_lst <- myfit_mdl_fn(model_id=paste0(model_id_pfx, ".cv.0.lser.ys"), model_method=model_method,
                        indep_vars_vctr=indep_vars_vctr,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_entity_df, OOB_df=glb_newent_df,
                        n_cv_folds=0, tune_models_df=NULL,
                        model_loss_mtrx=glb_model_metric_terms,
                        model_summaryFunction=glb_model_metric_smmry,
                        model_metric=glb_model_metric,
                        model_metric_maximize=glb_model_metric_maximize)
ret_lst <- myfit_mdl_fn(model_id=paste0(model_id_pfx, ".cv.G.lser.ys"), model_method=model_method,
                        indep_vars_vctr=indep_vars_vctr,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_entity_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=glb_tune_models_df,
                        model_loss_mtrx=glb_model_metric_terms,
                        model_summaryFunction=glb_model_metric_smmry,
                        model_metric=glb_model_metric,
                        model_metric_maximize=glb_model_metric_maximize)

from=0.00001; to=0.00105; by=0.00001; seq(from=from, to=to, by=by)
ret_lst <- myfit_mdl_fn(model_id=paste0(model_id_pfx, ".cv.G.lser.no"), model_method=model_method,
                        indep_vars_vctr=indep_vars_vctr,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_entity_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=glb_tune_models_df)

ret_lst <- myfit_mdl_fn(model_id=paste0(model_id_pfx, ".cv.G.lser.no.rnorm.ys"), model_method=model_method,
                        indep_vars_vctr=indep_vars_vctr,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_entity_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=glb_tune_models_df)
caret_mdl <- glb_models_lst[["All.X.tst.cv.G.lser.no.rnorm.ys.rpart"]]

min=0.000600, max=0.001000, by=0.000002
min=0.0001; max=0.0024; by=0.00012; seq(from=min, to=max, by=by)
print(glb_tune_models_df <-
    rbind(
        data.frame(parameter="cp", min=from, max=to, by=by),
        data.frame(parameter="mtry", min=2, max=4, by=1)))
glb_n_cv_folds <- 7
model_id_pfx <- "All.X.tst";    model_method <- "rpart";
ret_lst <- myfit_mdl_fn(model_id=paste0(model_id_pfx, ".cv.G.lser.no.rnorm.no"), model_method=model_method,
#                        indep_vars_vctr=setdiff(indep_vars_vctr, ".rnorm"),
                        indep_vars_vctr=setdiff(indep_vars_vctr, NULL),
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_entity_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=glb_tune_models_df)
caret_mdl <- glb_models_lst[["All.X.tst.cv.G.lser.no.rnorm.no.rpart"]]
print(ggplot(caret_mdl) + geom_vline(xintercept=caret_mdl$bestTune[1, 1], linetype="dotted"))
#prp(caret_mdl$finalModel)
print(            mypredict_mdl(caret_mdl, glb_newent_df, glb_rsp_var,
                                glb_rsp_var_out, model_id, "OOB",
                                glb_model_metric_smmry, glb_model_metric,
                                glb_model_metric_maximize, ret_type="stats"))

print(indep_vars_vctr <- setdiff(names(glb_entity_df), union(glb_rsp_var, glb_exclude_vars_as_features)))
ntv_mdl <- rpart(bucket2009 ~ age + alzheimers + arthritis + cancer + copd + depression + diabetes + heart.failure + ihd + kidney + osteoporosis + stroke + bucket2008 + reimbursement2008, data=glb_entity_df, method="class", cp=0.00005)

ntv_mdl <- rpart(reformulate(indep_vars_vctr, glb_rsp_var), data=glb_entity_df, method="class", cp=0.00005)
ntv_mdl <- rpart(reformulate(setdiff(indep_vars_vctr, ".rnorm"), glb_rsp_var), data=glb_entity_df, method="class", cp=0.00005)
prp(ntv_mdl)
dev.off()

# This is too much for Preview
ntv_mdl <- rpart(reformulate(setdiff(indep_vars_vctr, ".rnorm"), glb_rsp_var), data=glb_entity_df, method="class", cp=0.00010)
png("ntv_mdl_cp_0.00010.png", width=480*500, height=480*5)
prp(ntv_mdl)
dev.off()

# This works but still too big
ntv_mdl <- rpart(reformulate(setdiff(indep_vars_vctr, ".rnorm"), glb_rsp_var), data=glb_entity_df, method="class", cp=0.00050)
png("ntv_mdl_cp_0.00050.png", width=480*5, height=480*5)
prp(ntv_mdl)
dev.off()

# Working on this right now
ntv_mdl <- rpart(reformulate(setdiff(indep_vars_vctr, ".rnorm"), glb_rsp_var), data=glb_entity_df, method="class", cp=0.00100)
png("ntv_mdl_cp_0.00100.png", width=480*5, height=480*5)
prp(ntv_mdl)
dev.off()
myControl <- trainControl(method="none", number=0, verboseIter=TRUE)
cp1Grid = expand.grid(cp=0.00100)
crt_mdl <- train(reformulate(setdiff(indep_vars_vctr, ".rnorm"), glb_rsp_var), data=glb_entity_df,
                 method="rpart", trControl=myControl, tuneGrid=cp1Grid, tuneLength=nrow(cp1Grid))
png("crt_mdl_cp_0.00100.png", width=480*5, height=480*5)
prp(crt_mdl$finalModel)
dev.off()
print(confusionMatrix(predict(crt_mdl, glb_newent_df, "raw"),
                      predict(ntv_mdl, glb_newent_df, "class"))$table)

mycvControl <- trainControl(method="cv", number=7, verboseIter=TRUE)
print(cpGrid <- expand.grid(cp=seq(0.0001, 0.0020, 0.0001)))
crtcv_mdl <- train(reformulate(setdiff(indep_vars_vctr, ".rnorm"), glb_rsp_var), data=glb_entity_df,
                 method="rpart", trControl=mycvControl, tuneGrid=cpGrid, tuneLength=nrow(cpGrid))
ggplot(crtcv_mdl) + geom_vline(xintercept=crtcv_mdl$bestTune[1, 1], linetype="dotted")
train(reformulate(setdiff(indep_vars_vctr, ".rnorm"), glb_rsp_var), data=glb_entity_df,
      method= "rpart", trControl=mycvControl, tuneGrid=cpGrid)
print(crtcv_mdl$results)
png("crtcv_mdl_cp_0.00180.png", width=480*5, height=480*5)
prp(crtcv_mdl$finalModel)
dev.off()
print(confusionMatrix(predict(crt_mdl, glb_newent_df, "raw"),
                      predict(ntv_mdl, glb_newent_df, "class"))$table)

ntv_mdl <- rpart(reformulate(union(setdiff(indep_vars_vctr, ".rnorm"), ".rnorm"), glb_rsp_var), data=glb_entity_df, method="class", cp=0.00005)
print(ntv_mdl)
#prp(ntv_mdl)

ntv_mdl <- rpart(reformulate(c("bucket2008", "reimbursement2008"), glb_rsp_var), data=glb_entity_df, method="class", cp=0.00112)
print(ntv_mdl)
prp(ntv_mdl)

ntv_mdl <- rpart(reformulate(c("bucket2008", "reimbursement2008"), glb_rsp_var), data=glb_entity_df, method="class", cp=0.00112)
print(ntv_mdl)
prp(ntv_mdl)

min=0.000600, max=0.001000, by=0.000002
min=0.0001, max=0.0012, by=0.00012
min=0.00004, max=0.00006, by=0.00002
min=0.00005; max=0.00005; by=0.000005; seq(from=min, to=max, by=by)

ddive_crt <- function(crt_mdl_id) {
    crt_mdl <- glb_models_lst[[crt_mdl_id]]
    ggplot(crt_mdl) +  geom_vline(xintercept=crt_mdl$bestTune[1, 1], linetype="dotted")
    prp(crt_mdl$finalModel)
    print(crt_mdl$finalModel)
    print(confusion_mtrx <- t(confusionMatrix(predict(crt_mdl, glb_newent_df, "raw") ,glb_newent_df[, glb_rsp_var])$table))
    print(sprintf("min.loss.error.OOB: %0.4f", sum(confusion_mtrx * glb_model_metric_terms) / nrow(glb_newent_df)))
    print(sprintf("max.Accuracy.OOB: %0.4f", sum(diag(confusion_mtrx)) / nrow(glb_newent_df)))
    print(sprintf("%s: %0.5f", names(crt_mdl$bestTune), crt_mdl$bestTune[, 1]))
}

ddive_rpart <- function(crt_mdl_id) {
    crt_mdl <- glb_models_lst[[crt_mdl_id]]
}
ddive_crt("Interact.High.cor.y.rpart")
ddive_crt("All.X.lser.no.rpart")
ddive_crt("All.X.lser.ys.cp.opt.rpart")

print(indep_vars_vctr <- setdiff(names(glb_entity_df),
                           union(union(glb_rsp_var, glb_exclude_vars_as_features), ".rnorm")))
print(cp_val <- glb_models_lst[["All.X.lser.no.rpart"]]$bestTune[1, 1])
print(cp_val <- glb_models_lst[["Interact.High.cor.y.rpart"]]$bestTune[1, 1])
ntv_mdl <- rpart(reformulate(indep_vars_vctr, glb_rsp_var), data=glb_entity_df, method="class", cp=cp_val)
print(ntv_mdl)
prp(ntv_mdl)
print(confusion_mtrx <- t(confusionMatrix(predict(ntv_mdl, glb_newent_df, "class") ,glb_newent_df[, glb_rsp_var])$table))
print(sprintf("min.loss.error.OOB: %0.4f", sum(confusion_mtrx * glb_model_metric_terms) / nrow(glb_newent_df)))
print(sprintf("max.Accuracy.OOB: %0.4f", sum(diag(confusion_mtrx)) / nrow(glb_newent_df)))

print(cp_val <- glb_models_lst[["Max.cor.Y.cv.G.rpart"]]$bestTune[1, 1])
print(cp_val <- glb_models_lst[["Interact.High.cor.y.rpart"]]$bestTune[1, 1])
print(cp_val <- glb_models_lst[["All.X.lser.ys.cp.opt.rpart"]]$bestTune[1, 1])
print(cp_val <- glb_models_lst[["All.X.lser.no.cp.4015.rpart"]]$bestTune[1, 1])
ggplot(crt_mdl <- glb_models_lst[["All.X.lser.no.cp.4015.rpart"]]) +  geom_vline(xintercept=crt_mdl$bestTune[1, 1], linetype="dotted")
print(cp_val <- glb_models_lst[["All.X.lser.ys.cp.4015.rpart"]]$bestTune[1, 1])
print(cp_val <- glb_models_lst[["All.X.lser.no.cp.opt.rpart"]]$bestTune[1, 1])
ggplot(crt_mdl <- glb_models_lst[["All.X.lser.ys.cp.4015.rpart"]]) +  geom_vline(xintercept=crt_mdl$bestTune[1, 1], linetype="dotted")

print(confusion_mtrx <- t(confusionMatrix(predict(glb_models_lst[["Baseline.mybaseln_classfr"]], glb_newent_df, "raw") ,glb_newent_df[, glb_rsp_var])$table))
print(confusion_mtrx <- t(confusionMatrix(glb_newent_df[, glb_Baseline_mdl_var], glb_newent_df[, glb_rsp_var])$table))