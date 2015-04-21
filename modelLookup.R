tmp_df <- modelLookup()
print(nrow(subset(modelLookup(), model == "rpart")))
print(nrow(subset(modelLookup(), model == "arbit")))
