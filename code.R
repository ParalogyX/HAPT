###########################################################
#   HarvardX: PH125.9x  Data Science: Capstone            #
#         Smartphone-Based Recognition of                 #
#    Human Activities and Postural Transitions            #
###########################################################

##########################################################
#            READ THIS BEFORE RUN THE CODE!!!            #
#                                                        #
#  1. This code is made for R version 4.1.1 (2021-08-10) #
#      Check your R version (type "R.version.string"     #
#                       in console)                      #
#  2.                                                      #
#                                                        #
#  2.    Run complete code with console output            #
#     can take 30-60 minutes, depends on a computer      #
#                                                        #
# Code itself is well commented as code, for theoretical #
#     background please look in report.pdf               #
##########################################################


# clear console output
cat("\014")

###########################################################
#                        LIBRARIES                        #
###########################################################
# library installations if needed:
if(!require(rstudioapi)) install.packages("rstudioapi")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(gridExtra)) install.packages("gridExtra")
if(!require(grid)) install.packages("grid")
if(!require(caret)) install.packages("caret")
if(!require(corrplot)) install.packages("corrplot")
if(!require(xgboost)) install.packages("xgboost")
if(!require(imbalance)) install.packages("imbalance")
if(!require(UBL)) install.packages("UBL")
if(!require(ROSE)) install.packages("ROSE")
if(!require(smotefamily)) install.packages("smotefamily")



# Load libraries
library(rstudioapi)
library(tidyverse)
library(gridExtra)
library(grid) 
library(caret)
library(corrplot)
library(xgboost)
library(imbalance)
library(UBL)
library(ROSE)
library(smotefamily)

# Set working directory to source file location
setwd(dirname(getActiveDocumentContext()$path))

# Program controls
RETRAIN <- F       # TRUE: models will be retrained; FALSE: trained models will be loaded from files
PRINT_DEBUG <- T   # TRUE: debug information and training functions output will be printed out to the console; 
#                    FALSE: no or only minimum of debug information will be printed out to the console

set.seed(11, sample.kind="Rounding")          # Set seed for reproducibility

###########################################################
#         Download dataset if not downloaded              #
#  Load dataset for analysis (x_train + y_train = df)   #
#         Load hold-out test set for final              #
#           validation (x_test and y_test)                #
###########################################################

# If "Data" folder is not exist, create it
if (!dir.exists("./data")) {
  dir.create("./data")
}

# Download file if it is not downloaded yet
if (!file.exists("./data//HAPT Data Set.zip")) {
  download.file("http://archive.ics.uci.edu/ml/machine-learning-databases/00341/HAPT%20Data%20Set.zip", "./data//HAPT Data Set.zip")
}

# Read data from all files in the archive and assign to variables

# First read features labels, to be able to name columns in the features dataframe
features <- read.csv(unzip("./data//HAPT Data Set.zip", "features.txt"), header = FALSE)
features <- as.vector(features[,1])

# Replace "-" to "_" from the features names, because it replaces by "." when apply to column names
features <- str_replace_all(features, "-", "_")
# And remove spaces
features <- str_replace_all(features, " ", "")

# Read activity labels to replace activity number by activity label in the outcome vector
activity_labels <- read.csv(unzip("./data//HAPT Data Set.zip", "activity_labels.txt"), sep = " ", header = FALSE)
colnames(activity_labels) <- c("Activity_number", "Activity")
activity_labels <- activity_labels %>% select("Activity_number", "Activity")

# Unzip and read training data
x_train <- read.csv(unzip("./data//HAPT Data Set.zip", "Train/X_train.txt"), sep = " ", header = FALSE, col.names = features)
y_train <- read.csv(unzip("./data//HAPT Data Set.zip", "Train/y_train.txt"), sep = " ", header = FALSE, col.names = "Activity_number")

# Merge outcome vector and activity labels to have labels instead of numbers
y_train <- y_train %>% left_join(activity_labels, by = "Activity_number") %>% select(-"Activity_number")


# Unzip and read testing data (feature vector is the same that for training data)
x_test <- read.csv(unzip("./data//HAPT Data Set.zip", "Test/X_test.txt"), sep = " ", header = FALSE, col.names = features)
y_test <- read.csv(unzip("./data//HAPT Data Set.zip", "Test/y_test.txt"), sep = " ", header = FALSE, col.names = "Activity_number")

# Merge outcome vector and activity labels to have labels instead of numbers
y_test <- y_test %>% left_join(activity_labels, by = "Activity_number") %>% select(-"Activity_number")

# Delete unzipped files and folders
unlink(c("Test", "Train", "activity_labels.txt", "features.txt"), recursive = T)

# Change Activity column type to factor
y_train <- y_train %>% mutate(Activity = factor(Activity))
y_test <- y_test %>% mutate(Activity = factor(Activity))


# x_test and y_test will be considered as unknown data and will not be used till the end of the project as final validation.
# Data analysis, model training and selection will be performed entirely on the train dataset (x_train and y_train)

# combine x_train and y_train to one dataframe for EDA

df <- cbind(x_train, y_train)


###########################################################
#       Functions
##########################################################

plot_confusion <- function(truth, pred, name = "Confusion matrix"){

  xtab <- table(pred, truth)
  cm <- confusionMatrix(xtab)
  plt <- as.data.frame(cm$table)
  colnames(plt) <- c("Prediction", "Reference", "Freq")
  plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
  
  ggplot(plt, aes(Prediction,Reference, fill= Freq)) +
    geom_tile() + geom_text(aes(label=Freq)) +
    scale_fill_gradient(low="white", high="#009194") +
    labs(x = "Reference",y = "Prediction") +
    scale_x_discrete(labels=levels(plt$Prediction)) +
    scale_y_discrete(labels=rev(levels(plt$Prediction))) +
    ggtitle(name) +
    theme(axis.text.x = element_text(angle = 45, vjust = 1.0, hjust=1))
}


###########################################################
#                         Analysis                        #
###########################################################

# dimensions
dim(df)

# features names
df %>% colnames()

# Head and tail of dataframe with only first five features and the outcome column
head(df[c(1:5, 562)])
tail(df[c(1:5, 562)])

# NAs in dataset
df %>% is.na() %>% sum()
# no NAs


# distribution of outcomes in the training dataset
df %>% group_by(Activity) %>% mutate(n = n()) %>%
  ggplot(aes(reorder(Activity, -n))) +
  geom_bar() + 
  xlab("Activity") + 
  theme(axis.text.x=element_text(angle = -90, hjust = 0))

# dataset is unbalanced, less data points for transitions between activities compare to continuous activities  
# Will take it into account when build a model
# Metric will be Cohen's kappa coefficient

# All features statistics summary 
features_stat <- as.data.frame(df %>% select(-Activity) %>% summary()) %>% select(-Var1)
colnames(features_stat) <-(c("feature_name", "parameter"))

features_stat <- features_stat %>% mutate(param_name = map(strsplit(parameter, ":"),1), 
                                          value = as.numeric(unlist(map(strsplit(parameter, ":"),2)))) %>% 
  select(-parameter)

features_stat <- features_stat %>% pivot_wider(names_from = param_name, values_from = value)

# rename columns for readability and easier access
colnames(features_stat) <- c("feature", "min", "first_q", "median", "mean", "third_q", "max")

plot_min <- features_stat %>% 
  ggplot(aes(min)) +
  geom_histogram() + 
  xlab("Minimal values of all features")

plot_max <- features_stat %>% 
  ggplot(aes(max)) +
  geom_histogram() + 
  xlab("Maximum values of all features")

plot_median <- features_stat %>% 
  ggplot(aes(median)) +
  geom_histogram() + 
  xlab("Median values of all features")

plot_mean <- features_stat %>% 
  ggplot(aes(mean)) +
  geom_histogram() + 
  xlab("Mean values of all features")

grid.arrange(plot_min, plot_max, plot_median, plot_mean, ncol=2)

# remove unnecessary variables:
rm(plot_max, plot_mean, plot_median, plot_min)

# 
# trying to balance
imbalanced_ratio <- imbalanceRatio(df, "Activity")
#df_balanced <- racog(df, numInstances = 100, classAttr = "Activity")
df_oversampled <- RandOverClassif(Activity ~., df, C.perc = "balance")
oversampled_ratio <- imbalanceRatio(df_oversampled, "Activity")


# distribution of outcomes in the training dataset
df_oversampled %>% group_by(Activity) %>% mutate(n = n()) %>%
  ggplot(aes(Activity)) +
  geom_bar() + 
  xlab("Activity") + 
  theme(axis.text.x=element_text(angle = -90, hjust = 0))


# Undersampling+oversampling both (SMOTE)

df_SMOTE <- SmoteClassif(Activity ~., df, C.perc = "balance")
SMOTE_ratio <- imbalanceRatio(df_SMOTE, "Activity")

df_SMOTE %>% group_by(Activity) %>% mutate(n = n()) %>%
  ggplot(aes(Activity)) +
  geom_bar() + 
  xlab("Activity") + 
  theme(axis.text.x=element_text(angle = -90, hjust = 0))


#Compare three df by lda with 10-fold cross-val
control <- trainControl(method="cv", number=10, verbose = PRINT_DEBUG)

time_start <- unclass(Sys.time())
imbalanced_lda_model <- train(Activity ~ ., method = "lda", data = df, trControl = control)
time_end <- unclass(Sys.time())
time_imbalanced_lda <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))

time_start <- unclass(Sys.time())
oversampled_lda_model <- train(Activity ~ ., method = "lda", data = df_oversampled, trControl = control)
time_end <- unclass(Sys.time())
time_oversampled_lda <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))


time_start <- unclass(Sys.time())
SMOTE_lda_model <- train(Activity ~ ., method = "lda", data = df_SMOTE, trControl = control)
time_end <- unclass(Sys.time())
time_SMOTE_lda <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))


# Test with test data (to create another)

imbalance_lda_test <- predict(imbalanced_lda_model, x_test)
plot_confusion(factor(imbalance_lda_test), factor(y_test$Activity), "Imbalance lda test")

oversampled_lda_test <- predict(oversampled_lda_model, x_test)
plot_confusion(factor(oversampled_lda_test), factor(y_test$Activity), "Oversampled lda test")

SMOTE_lda_test <- predict(SMOTE_lda_model, x_test)
plot_confusion(factor(SMOTE_lda_test), factor(y_test$Activity), "SMOTE lda test")


# stats
xtab <- table(factor(imbalance_lda_test), factor(y_test$Activity))
cm_lda_unbalanced <- confusionMatrix(xtab)
cm_lda_unbalanced$overall
xtab <- table(factor(oversampled_lda_test), factor(y_test$Activity))
cm_lda_oversampled <- confusionMatrix(xtab)
cm_lda_oversampled$overall
xtab <- table(factor(SMOTE_lda_test), factor(y_test$Activity))
cm_lda_SMOTE <- confusionMatrix(xtab)
cm_lda_SMOTE$overall




#Compare three df by knn with 10-fold cross-val
control <- trainControl(method="cv", number=10, verbose = PRINT_DEBUG)

time_start <- unclass(Sys.time())
imbalanced_knn_model <- train(Activity ~ ., method = "knn", data = df, trControl = control)
time_end <- unclass(Sys.time())
time_imbalanced_knn <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))

time_start <- unclass(Sys.time())
oversampled_knn_model <- train(Activity ~ ., method = "knn", data = df_oversampled, trControl = control)
time_end <- unclass(Sys.time())
time_oversampled_knn <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))


time_start <- unclass(Sys.time())
SMOTE_knn_model <- train(Activity ~ ., method = "knn", data = df_SMOTE, trControl = control)
time_end <- unclass(Sys.time())
time_SMOTE_knn <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))


# Test with test data (to create another)

imbalance_knn_test <- predict(imbalanced_knn_model, x_test)
plot_confusion(factor(imbalance_knn_test), factor(y_test$Activity), "Imbalance knn test")

oversampled_knn_test <- predict(oversampled_knn_model, x_test)
plot_confusion(factor(oversampled_knn_test), factor(y_test$Activity), "Oversampled knn test")

SMOTE_knn_test <- predict(SMOTE_knn_model, x_test)
plot_confusion(factor(SMOTE_knn_test), factor(y_test$Activity), "SMOTE knn test")


# stats
xtab <- table(factor(imbalance_knn_test), factor(y_test$Activity))
cm_knn_unbalanced <- confusionMatrix(xtab)
cm_knn_unbalanced$overall
xtab <- table(factor(oversampled_knn_test), factor(y_test$Activity))
cm_knn_oversampled <- confusionMatrix(xtab)
cm_knn_oversampled$overall
xtab <- table(factor(SMOTE_knn_test), factor(y_test$Activity))
cm_knn_SMOTE <- confusionMatrix(xtab)
cm_knn_SMOTE$overall




#Compare three df by svmLinear with 10-fold cross-val
control <- trainControl(method="cv", number=10, verbose = PRINT_DEBUG)

time_start <- unclass(Sys.time())
imbalanced_svmLinear_model <- train(Activity ~ ., method = "svmLinear", data = df, trControl = control)
time_end <- unclass(Sys.time())
time_imbalanced_svmLinear <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))

time_start <- unclass(Sys.time())
oversampled_svmLinear_model <- train(Activity ~ ., method = "svmLinear", data = df_oversampled, trControl = control)
time_end <- unclass(Sys.time())
time_oversampled_svmLinear <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))


time_start <- unclass(Sys.time())
SMOTE_svmLinear_model <- train(Activity ~ ., method = "svmLinear", data = df_SMOTE, trControl = control)
time_end <- unclass(Sys.time())
time_SMOTE_svmLinear <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))


# Test with test data (to create another)

imbalance_svmLinear_test <- predict(imbalanced_svmLinear_model, x_test)
plot_confusion(factor(imbalance_svmLinear_test), factor(y_test$Activity), "Imbalance svmLinear test")

oversampled_svmLinear_test <- predict(oversampled_svmLinear_model, x_test)
plot_confusion(factor(oversampled_svmLinear_test), factor(y_test$Activity), "Oversampled svmLinear test")

SMOTE_svmLinear_test <- predict(SMOTE_svmLinear_model, x_test)
plot_confusion(factor(SMOTE_svmLinear_test), factor(y_test$Activity), "SMOTE svmLinear test")


# stats
xtab <- table(factor(imbalance_svmLinear_test), factor(y_test$Activity))
cm_svmLinear_unbalanced <- confusionMatrix(xtab)
cm_svmLinear_unbalanced$overall
xtab <- table(factor(oversampled_svmLinear_test), factor(y_test$Activity))
cm_svmLinear_oversampled <- confusionMatrix(xtab)
cm_svmLinear_oversampled$overall
xtab <- table(factor(SMOTE_svmLinear_test), factor(y_test$Activity))
cm_svmLinear_SMOTE <- confusionMatrix(xtab)
cm_svmLinear_SMOTE$overall


# Dimension reduction


# calculate correlation matrix
correlationMatrix <- cor(x_train)
# summarize the correlation matrix
print(correlationMatrix)
corrplot(correlationMatrix, method = "color", tl.pos='n')
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.9)
highlyCorrelated2 <- findCorrelation(correlationMatrix, cutoff=0.75)
# print indexes of highly correlated attributes
#print(highlyCorrelated)
#print(highlyCorrelated2)

# remove highly correlated features
df_reduced_90 <- df %>% select(-all_of(highlyCorrelated))
df_reduced_75 <- df %>% select(-all_of(highlyCorrelated2))

corrplot(cor(df_reduced_90 %>% select(-Activity)), method = "color", tl.pos='n')
corrplot(cor(df_reduced_75 %>% select(-Activity)), method = "color", tl.pos='n')

# prepare test data
x_test_corr90 <- x_test %>% select(-all_of(highlyCorrelated))
x_test_corr75 <- x_test %>% select(-all_of(highlyCorrelated2))

# Applying PCA


df_PCA <- prcomp(x_train, scale = TRUE)

plot(df_PCA)

df_PCA_15 <- as.data.frame(df_PCA$x[,1:15]) %>% cbind(y_train)
df_PCA_100 <- as.data.frame(df_PCA$x[,1:100]) %>% cbind(y_train)
df_PCA_200 <- as.data.frame(df_PCA$x[,1:200]) %>% cbind(y_train)
df_PCA_300 <- as.data.frame(df_PCA$x[,1:300]) %>% cbind(y_train)




# Transform test_set
col_means <- colMeans(x_test)
x_test_PCA <- sweep(as.matrix(x_test), 2, col_means) %*% df_PCA$rotation
x_test_15 <- as.data.frame(x_test_PCA[,1:15])
x_test_100 <- as.data.frame(x_test_PCA[,1:100])
x_test_200 <- as.data.frame(x_test_PCA[,1:200])
x_test_300 <- as.data.frame(x_test_PCA[,1:300])

# oversampled 100

df_PCA_overs <- prcomp(df_oversampled[,1:561], scale = TRUE)
df_PCA_SMOTE <- prcomp(df_SMOTE[,1:561], scale = TRUE)


df_PCA_100_overs <- as.data.frame(df_PCA_overs$x[,1:100]) %>% cbind(Activity = df_oversampled$Activity)
df_PCA_100_SMOTE <- as.data.frame(df_PCA_SMOTE$x[,1:100]) %>% cbind(Activity = df_SMOTE$Activity)

#Compare three df by lda with 10-fold cross-val
control <- trainControl(method="cv", number=10, verbose = PRINT_DEBUG)

time_start <- unclass(Sys.time())
corr90_lda_model <- train(Activity ~ ., method = "lda", data = df_reduced_90, trControl = control)
time_end <- unclass(Sys.time())
time_corr90_lda <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))

time_start <- unclass(Sys.time())
corr75_lda_model <- train(Activity ~ ., method = "lda", data = df_reduced_75, trControl = control)
time_end <- unclass(Sys.time())
time_corr75_lda <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))


time_start <- unclass(Sys.time())
PCA15_lda_model <- train(Activity ~ ., method = "lda", data = df_PCA_15, trControl = control)
time_end <- unclass(Sys.time())
time_PCA15_lda <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))

time_start <- unclass(Sys.time())
PCA100_lda_model <- train(Activity ~ ., method = "lda", data = df_PCA_100, trControl = control)
time_end <- unclass(Sys.time())
time_PCA100_lda <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))

time_start <- unclass(Sys.time())
PCA200_lda_model <- train(Activity ~ ., method = "lda", data = df_PCA_200, trControl = control)
time_end <- unclass(Sys.time())
time_PCA200_lda <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))

time_start <- unclass(Sys.time())
PCA300_lda_model <- train(Activity ~ ., method = "lda", data = df_PCA_300, trControl = control)
time_end <- unclass(Sys.time())
time_PCA300_lda <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))


time_start <- unclass(Sys.time())
PCA100_overs_lda_model <- train(Activity ~ ., method = "lda", data = df_PCA_100_overs, trControl = control)
time_end <- unclass(Sys.time())
time_PCA100_overs_lda <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))

time_start <- unclass(Sys.time())
PCA100_SMOTE_lda_model <- train(Activity ~ ., method = "lda", data = df_PCA_100_SMOTE, trControl = control)
time_end <- unclass(Sys.time())
time_PCA100_SMOTE_lda <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))


# Test with test data (to create another)

corr90_lda_test <- predict(corr90_lda_model, x_test_corr90)
plot_confusion(factor(corr90_lda_test), factor(y_test$Activity), "Corr > 0.9 lda test")

corr75_lda_test <- predict(corr75_lda_model, x_test_corr75)
plot_confusion(factor(corr75_lda_test), factor(y_test$Activity), "Corr > 0.75 lda test")

PCA15_lda_test <- predict(PCA15_lda_model, x_test_15)
plot_confusion(factor(PCA15_lda_test), factor(y_test$Activity), "15 PCA lda test")



PCA100_lda_test <- predict(PCA100_lda_model, x_test_100)
plot_confusion(factor(PCA100_lda_test), factor(y_test$Activity), "100 PCA lda test")

PCA200_lda_test <- predict(PCA200_lda_model, x_test_200)
plot_confusion(factor(PCA200_lda_test), factor(y_test$Activity), "200 PCA lda test")

PCA300_lda_test <- predict(PCA300_lda_model, x_test_300)
plot_confusion(factor(PCA300_lda_test), factor(y_test$Activity), "300 PCA lda test")


PCA100_overs_lda_test <- predict(PCA100_overs_lda_model, x_test_100)
plot_confusion(factor(PCA100_overs_lda_test), factor(y_test$Activity), "100 PCA oversampled lda test")


# stats
xtab <- table(factor(corr90_lda_test), factor(y_test$Activity))
cm_lda_corr90 <- confusionMatrix(xtab)
cm_lda_corr90$overall
xtab <- table(factor(corr75_lda_test), factor(y_test$Activity))
cm_lda_corr75 <- confusionMatrix(xtab)
cm_lda_corr75$overall
xtab <- table(factor(PCA15_lda_test), factor(y_test$Activity))
cm_lda_PCA15 <- confusionMatrix(xtab)
cm_lda_PCA15$overall
xtab <- table(factor(PCA100_lda_test), factor(y_test$Activity))
cm_lda_PCA100 <- confusionMatrix(xtab)
cm_lda_PCA100$overall
xtab <- table(factor(PCA200_lda_test), factor(y_test$Activity))
cm_lda_PCA200 <- confusionMatrix(xtab)
cm_lda_PCA200$overall
xtab <- table(factor(PCA300_lda_test), factor(y_test$Activity))
cm_lda_PCA300 <- confusionMatrix(xtab)
cm_lda_PCA300$overall








# 
# # remove unnecessary variables:
# rm(correlationMatrix, highlyCorrelated)
# 
# # Model building
# # Metric will kappa-value, because our dataset is imbalanced 





#df_ttt <- df %>% sample_n(500)

#fit <- train(Activity ~ ., method = "gamLoess", data = df)
#df.cols <- df[, 1:ncol(df)-1]

#fit <- train(x = df.cols, y = df$Activity, method = "rf")



# if (RETRAIN) {
#   # models <- c("lda", "naive_bayes", "svmLinear", "knn", "gamLoess", "multinom", "qda", "rf", "adaboost")
#   models <- c("lda", "naive_bayes", "svmLinear", "knn")
#   
#   fits <- lapply(models, function(model){ 
#     print(model)
#     time_start <- unclass(Sys.time())
#     fit <- train(Activity ~ ., method = model, data = df_balanced)
#     time_end <- unclass(Sys.time())
#     time <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))
#     c(fit, time)
#   }) 
#   
#   names(fits) <- models
#   
#   
#   saveRDS(fits, "./models/ensemble_imba.rds")
# } else {
#     if (!file.exists("./models//ensemble_imba.rds")) {
#       stop("File not found. Rerun code with RETRAIN = TRUE")
#     } else {
#       fits <- readRDS("./models//ensemble_imba.rds")
#     }
#   }

# https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00349-y
# https://towardsdatascience.com/machine-learning-multiclass-classification-with-imbalanced-data-set-29f6a177c1a
# 
# # try Multinomial Logistic Regression on full dataset
# df_ttt <- df[c(1:10, 562)]
# 
# if (RETRAIN) {
#   control <- trainControl(method="cv", number=10, verbose = PRINT_DEBUG)
#   time_start <- unclass(Sys.time())
#   model <- train(Activity~., data = df_ttt, method="multinom", MaxNWts = 10000, metric = "Kappa", trControl = control)
#   time_end <- unclass(Sys.time())
#   MLR_full_time <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))
#   if (PRINT_DEBUG) {print(paste("Time for first MLR is: ", MLR_full_time))}
#   # If "models" folder is not exist, create it
#   if (!dir.exists("./models")) {
#     dir.create("./models")
#   }
#   saveRDS(model, "./models//MLR_full_model.rds")
# } else {
#   if (!file.exists("./models//MLR_full_model.rds")) {
#     stop("File not found. Rerun code with RETRAIN = TRUE")
#   } else {
#     model <- readRDS("./models//MLR_full_model.rds")
#   }
# }
# 
# $
# 
# 
# truth <- factor(y_test$Activity)
# pred <- predict(model, newdata = x_test)
# 
# xtab <- table(pred, truth)
# cm_full <- confusionMatrix(xtab)
# 
# 
# print(cm_full)
# 


# 
# 
# plot_confusion(factor(y_test$Activity),factor(y_test$Activity))
# 
# xtab <- table(factor(y_test$Activity),factor(y_test$Activity))
# cm <- confusionMatrix(xtab, mode = "everything")
# 
# 
# 
# # try Multinomial Logistic Regression on the reduced  1 dataset
# if (RETRAIN) {
#   control <- trainControl(method="cv", number=3, verbose = PRINT_DEBUG)
#   time_start <- unclass(Sys.time())
#   model <- train(Activity~., data = df_reduced, method="multinom", MaxNWts = 10000, trControl = control)
#   time_end <- unclass(Sys.time())
#   MLR_reduced1_time <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))
# 
#   if (PRINT_DEBUG) {print(paste("Time for first MLR is: ", ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))))}
#   # If "models" folder is not exist, create it
#   if (!dir.exists("./models")) {
#     dir.create("./models")
#   }
#   saveRDS(model, "./models//MLR_reduced1_model.rds")
# } else {
#   if (!file.exists("./models//MLR_reduced1_model.rds")) {
#     stop("File not found. Rerun code with RETRAIN = TRUE")
#   } else {
#     model <- readRDS("./models//MLR_reduced1_model.rds")
#   }
# }
# 
# 
# 
# truth <- factor(y_test$Activity)
# pred <- predict(model, newdata = x_test)
# 
# xtab <- table(pred, truth)
# cm_reduced1 <- confusionMatrix(xtab)
# 
# 
# print(cm_reduced1)
# 
# 
# # try Multinomial Logistic Regression on the reduced  2 dataset
# if (RETRAIN) {
#   control <- trainControl(method="cv", number=3, verbose = PRINT_DEBUG)
#   time_start <- unclass(Sys.time())
#   model <- train(Activity~., data = df_reduced2, method="multinom", MaxNWts = 10000, trControl = control)
#   time_end <- unclass(Sys.time())
#   MLR_reduced2_time <- ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))
#   
#   if (PRINT_DEBUG) {print(paste("Time for first MLR is: ", ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))))}
#   # If "models" folder is not exist, create it
#   if (!dir.exists("./models")) {
#     dir.create("./models")
#   }
#   saveRDS(model, "./models//MLR_reduced2_model.rds")
# } else {
#   if (!file.exists("./models//MLR_reduced2_model.rds")) {
#     stop("File not found. Rerun code with RETRAIN = TRUE")
#   } else {
#     model <- readRDS("./models//MLR_reduced2_model.rds")
#   }
# }
# 
# 
# truth <- factor(y_test$Activity)
# pred <- predict(model, newdata = x_test)
# 
# xtab <- table(pred, truth)
# cm_reduced2 <- confusionMatrix(xtab)
# 
# 
# print(cm_reduced2)
# 
# 
# 
# # try QDA
# train_qda <- train(Activity ~., method = "lda", data = df)
# # Obtain predictors and accuracy
# truth <- factor(y_test$Activity)
# pred <- predict(train_qda, newdata = x_test)
# 
# xtab <- table(pred, truth)
# cm <- confusionMatrix(xtab)
# 
# 
# print(cm)
# 
# 
# # try gBoost
# control <- trainControl(method="cv", number=10, verbose = PRINT_DEBUG)
# train_xgb <- train(Activity ~., method = "xgbTree", data = df, trControl = control)
# # Obtain predictors and accuracy
# truth <- factor(y_test$Activity)
# pred <- predict(train_xgb, newdata = x_test)
# 
# xtab <- table(pred, truth)
# cm <- confusionMatrix(xtab)
# 
# 
# print(cm)
# # Best tuning parameter
# train_xgb$bestTune
# 
# 
# # truth <- factor(y_test$Activity)
# # pred <- predict(model, newdata = x_test)
# # 
# # xtab <- table(pred, truth)
# # cm <- confusionMatrix(xtab)
# # 
# # 
# # print(cm)
