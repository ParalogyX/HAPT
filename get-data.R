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
if(!require(caret)) install.packages("caret")

# Load libraries
library(rstudioapi)
library(tidyverse)
library(gridExtra)
library(caret)

# Set working directory to source file location
setwd(dirname(getActiveDocumentContext()$path))

# Program controls
RETRAIN <- F       # TRUE: models will be retrained; FALSE: trained models will be loaded from files
set.seed(11, sample.kind="Rounding")          # Set seed for reproducibility

###########################################################
#         Download dataset if not downloaded              #
#  Create dataset for analysis (x_train + y_train = df)   #
#         Create hold-out test set for final              #
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


# x_test and y_test will be considered as unknown data and will not be used till the end of the project as final validation.
# Data analysis, model training and selection will be performed entirely on the train dataset (x_train and y_train)

# combine x_train and y_train to one dataframe for EDA

df <- cbind(x_train, y_train)


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

# distribution of outcomes in the training dataset
df %>% group_by(Activity) %>% mutate(n = n()) %>%
  ggplot(aes(reorder(Activity, -n))) +
  geom_bar() + 
  xlab("Activity") + 
  theme(axis.text.x=element_text(angle = -90, hjust = 0))

# dataset is unbalanced, less data points for transitions between activities compare to continuous activities  
# Will take it into account when build a model

# First feature statistics
df$tBodyAcc_Mean_1 %>% summary()

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
# many features has very low median value. That can mean, that they are not changing a lot and therefore are not informative
# But it doesn't mean that we can just throw them away: our dataset is unbalanced, these features can indicate small outcome classes

# check features with median value < -0.99 (half of the values are -1)
#s_features <- features_stat %>% filter(median < -0.99 & mean < -0.95)

#df$s_features$feature[1]

# Features selection:
# https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/

# Dimension reduction

# calculate correlation matrix
correlationMatrix <- cor(x_train)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
# print indexes of highly correlated attributes
print(highlyCorrelated)

df_reduced <- df %>% select(-all_of(highlyCorrelated))


# repeat analysis for reduced data
features_stat <- as.data.frame(df_reduced %>% select(-Activity) %>% summary()) %>% select(-Var1)
colnames(features_stat) <-(c("feature_name", "parameter"))

features_stat <- features_stat %>% mutate(param_name = map(strsplit(parameter, ":"),1), 
                                          value = as.numeric(unlist(map(strsplit(parameter, ":"),2)))) %>% 
  select(-parameter)

features_stat <- features_stat %>% pivot_wider(names_from = param_name, values_from = value)

# rename columns for readablity and easier access
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












# Dimension reduction using random forest selection with 3-fold cross validation

# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=3, verbose = T)
# run the RFE algorithm
if (RETRAIN) {
  time_before_rfe <- Sys.time()
  print(time_before_rfe)
  results <- rfe(df_reduced[,1:(length(df_reduced)-1)], as.factor(df_reduced[,length(df_reduced)]), sizes=c(1:(length(df_reduced)-1)), rfeControl=control)
  # If "models" folder is not exist, create it
  if (!dir.exists("./models")) {
    dir.create("./models")
  }
  saveRDS(results, "./models//rf_dr_result.rds")
  time_after_rfe <- Sys.time()
  print(time_after_rfe)
} else {
  if (!file.exists("./models//rf_dr_result.rds")) {
    stop("File not found. Rerun code with RETRAIN = TRUE")
  } else {
    results <- readRDS("./models//rf_dr_result.rds")
  }
}

# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))


df_reduced_2 <- df %>% select(all_of(predictors(results)), Activity)

# repeat analysis for reduced data second time
features_stat <- as.data.frame(df_reduced_2 %>% summary()) %>% select(-Var1)
colnames(features_stat) <-(c("feature_name", "parameter"))

features_stat <- features_stat %>% mutate(param_name = map(strsplit(parameter, ":"),1), 
                                          value = as.numeric(unlist(map(strsplit(parameter, ":"),2)))) %>% 
  select(-parameter)

features_stat <- features_stat %>% pivot_wider(names_from = param_name, values_from = value)

# rename columns for readablity and easier access
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

# Check which predictors can define transitions (outcomes which are minority)
reduced_predictors = predictors(results)
df %>% filter(Activity %in% c("SIT_TO_STAND", "STAND_TO_SIT", "LIE_TO_STAND")) %>% select(all_of(reduced_predictors), Activity) %>%
  pivot_longer(., cols = reduced_predictors, names_to = "Var", values_to = "Val") %>% ggplot(aes(x = Var, y = Val, col = Activity)) +
  geom_boxplot()


# Trying with Neural Network
# Learning Vector Quantization

# make activity column categorical (if works put in to the beginning)
df_reduced_2 <- df_reduced_2 %>% mutate(Activity = factor(Activity))


# prepare training scheme
control <- trainControl(method="repeatedcv", number=3, repeats=1, verbose = T)
# train the model
time_before_ranking <- Sys.time()
print(time_before_ranking)
model <- train(Activity~., data=df_reduced_2, method="lvq", preProcess="scale", trControl=control)
time_after_ranking <- Sys.time()
print(time_after_ranking)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
impDF <- data.frame(importance$importance)
Stats <- summarize_all(impDF, mean)
impDF$Mean <- apply(impDF, 1, mean)
impDF <- impDF %>% mutate(mean = mean())
top20pred <- impDF %>% arrange(desc(Mean)) %>% head(20) %>% rownames()
top50pred <- impDF %>% arrange(desc(Mean)) %>% head(50) %>% rownames()
top80pred <- impDF %>% arrange(desc(Mean)) %>% head(80) %>% rownames()
# summarize importance
print(importance)
# plot importance
plot(importance, top = 50)


df_reduced_3 <- df %>% select(all_of(top80pred), Activity)
df_reduced_3 <- df_reduced_3 %>% mutate(Activity = factor(Activity))

# try linear regression
model <- train(Activity~., data = df_reduced_3, method="multinom")


truth <- factor(y_test$Activity)
pred <- predict(model, newdata = x_test)

xtab <- table(pred, truth)
confusionMatrix(xtab)

# try linear with 154 vars: if improvement compare to 80?
df_reduced <- df_reduced %>% mutate(Activity = factor(Activity))
# try linear regression
model <- train(Activity~., data = df_reduced, method="multinom", MaxNWts = 10000)


truth <- factor(y_test$Activity)
pred <- predict(model, newdata = x_test)

xtab <- table(pred, truth)
confusionMatrix(xtab)

# Metric selection:
# https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd


