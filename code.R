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


# Load libraries
library(rstudioapi)
library(tidyverse)
library(gridExtra)
library(grid) 
library(caret)
library(corrplot)

# Set working directory to source file location
setwd(dirname(getActiveDocumentContext()$path))

# Program controls
RETRAIN <- T       # TRUE: models will be retrained; FALSE: trained models will be loaded from files
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

# Dimension reduction

# calculate correlation matrix
correlationMatrix <- cor(x_train)
# summarize the correlation matrix
print(correlationMatrix)
corrplot(correlationMatrix, method = "color", tl.pos='n')
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
# print indexes of highly correlated attributes
print(highlyCorrelated)

# remove highly correlated features
df_reduced <- df %>% select(-all_of(highlyCorrelated))

corrplot(cor(df_reduced %>% select(-Activity)), method = "color", tl.pos='n')

# remove unnecessary variables:
rm(correlationMatrix, highlyCorrelated)

# Model building
# Metric will kappa-value, because our dataset is imbalanced 

# try Multinomial Logistic Regression 
if (RETRAIN) {
  control <- trainControl(method="cv", number=3, verbose = PRINT_DEBUG)
  time_start <- unclass(Sys.time())
  model <- train(Activity~., data = df_reduced, method="multinom", MaxNWts = 10000, trControl = control)
  time_end <- unclass(Sys.time())
  if (PRINT_DEBUG) {print(paste("Time for first MLR is: ", ifelse((time_end - time_start)/60 > 180, paste((time_end - time_start)/3600, "hours"), paste((time_end - time_start)/60, "minutes"))))}
  # If "models" folder is not exist, create it
  if (!dir.exists("./models")) {
    dir.create("./models")
  }
  saveRDS(results, "./models//MLR_first_model.rds")
} else {
  if (!file.exists("./models//MLR_first_model.rds")) {
    stop("File not found. Rerun code with RETRAIN = TRUE")
  } else {
    results <- readRDS("./models//MLR_first_model.rds")
  }
}



truth <- factor(y_test$Activity)
pred <- predict(model, newdata = x_test)

xtab <- table(pred, truth)
cm <- confusionMatrix(xtab)


print(cm)
