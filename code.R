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
#                      LIBRARIES                          #
###########################################################
# library installations if needed:
if(!require(rstudioapi)) install.packages("rstudioapi")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(gridExtra)) install.packages("gridExtra")
if(!require(grid)) install.packages("grid")

# Load libraries
library(rstudioapi)
library(tidyverse)
library(gridExtra)
library(grid)

###########################################################
#                      PROGRAM CONTROL                    #
###########################################################

# Set working directory to source file location
setwd(dirname(getActiveDocumentContext()$path))

# Program controls
RETRAIN <- F       # TRUE: models will be retrained; FALSE: trained models will be loaded from files
PRINT_DEBUG <- T   # TRUE: debug information and training functions output will be printed out to the console; 
#                    FALSE: no or only minimum of debug information will be printed out to the console

set.seed(11, sample.kind="Rounding")          # Set seed for reproducibility


###########################################################
#         Download dataset if not downloaded              #
#   Load dataset for analysis (x_train + y_train = df)    #
#          Load hold-out test set for final               #
#     validation (x_test + y_test = df_validation)        #
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
df_validation <- cbind(x_test, y_test)

rm(x_train, y_train, x_test, y_test)


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

# plot outcomes distribution
df %>% group_by(Activity) %>% mutate(n = n()) %>%
  ggplot(aes(reorder(Activity, -n))) +
  geom_bar(col=rgb(0.1,0.4,0.5,0.7), fill=rgb(0.1,0.4,0.5,0.7)) + 
  xlab("Activity") + ylab("Count") +
  scale_y_continuous(breaks = seq(0,1500,100)) +
  ggtitle("Distribution of activities") +
  theme_bw() +
  theme(axis.text.x=element_text(angle = -60, hjust = 0))

# how many samples of each class in df
sort(table(df$Activity),decreasing=TRUE)

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
  geom_histogram(binwidth = 0.02, col=rgb(0.1,0.4,0.5,0.7), fill=rgb(0.1,0.4,0.5,0.7)) + 
  theme_bw() +
  xlab("Minimal values of all features")

plot_max <- features_stat %>% 
  ggplot(aes(max)) +
  geom_histogram(binwidth = 0.02, col=rgb(0.1,0.4,0.5,0.7), fill=rgb(0.1,0.4,0.5,0.7)) + 
  theme_bw() +
  xlab("Maximum values of all features")

plot_median <- features_stat %>% 
  ggplot(aes(median)) +
  geom_histogram(binwidth = 0.05, col=rgb(0.1,0.4,0.5,0.7), fill=rgb(0.1,0.4,0.5,0.7)) + 
  theme_bw() +
  xlab("Median values of all features")

plot_mean <- features_stat %>% 
  ggplot(aes(mean)) +
  geom_histogram(binwidth = 0.05, col=rgb(0.1,0.4,0.5,0.7), fill=rgb(0.1,0.4,0.5,0.7)) + 
  theme_bw() +
  xlab("Mean values of all features")

grid.arrange(plot_min, plot_max, plot_median, plot_mean, ncol=2, top = textGrob("Features summary",gp=gpar(fontsize=13,font=1), x = 0.07, hjust = 0))

# remove unnecessary variables:
rm(plot_max, plot_mean, plot_median, plot_min)


# five features with minimum mean value
low_5 <- features_stat %>% arrange(mean) %>% head(5) %>% .$feature
low_5
# select only low variance features and outcome from dataset
low_5_df <- df %>% select(c(all_of(low_5), Activity))
# build features distributions for each class
trends <- lapply(activity_labels$Activity, FUN = function(label) {
  low_5_df %>% filter(Activity == label) %>% ggplot() +
    geom_density(aes_string(x = as.character(low_5[1])), colour = "blue") + 
    geom_density(aes_string(x = as.character(low_5[2])), colour = "red") + 
    geom_density(aes_string(x = as.character(low_5[3])), colour = "green") + 
    geom_density(aes_string(x = as.character(low_5[4])), colour = "yellow") + 
    geom_density(aes_string(x = as.character(low_5[5])), colour = "magenta") +
    xlab("Value") +
    ggtitle(paste("class", label)) +
    theme_bw() +
    theme(plot.title = element_text(size = 15))
  
})
# add legend
# dummy trend to create legend
invisible(dummy_plot <- data.frame(feature = low_5, var1 = c(1,2,3,4,5), var2 = c(1,2,3,4,5)) %>% 
            ggplot(aes(var1, var2, color = feature)) +
            geom_point() + theme(legend.position="right") + 
            scale_color_manual(values=c("blue", "red", "green", "yellow", "magenta")))

## Function to extract legend
g_legend <- function(a.gplot){ 
  tmp <- ggplot_gtable(ggplot_build(a.gplot)) 
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box") 
  legend <- tmp$grobs[[leg]] 
  legend
} 
invisible(legend <- g_legend(dummy_plot))
trends[[nrow(activity_labels)+1]] <- legend

grid.arrange(grobs = trends, ncol=3, top = textGrob("Distribution of low variance features"))

# remove unnecessary variables:
rm(dummy_plot, legend, low_5_df, trends, low_5, g_legend)



###########################################################
#                      Model building                     #
###########################################################

library(caret)

# linear

control <- trainControl(method="cv", number=3, savePredictions = TRUE, classProbs = TRUE, verbose = PRINT_DEBUG)

library(yardstick)
#metrics = yardstick::metric_set(yardstick::roc_auc, yardstick::sens, yardstick::spec)
# Linear Discriminant Analysis
#library(MASS)
lda_fit <- train(Activity ~ ., data = df, method = "lda", trControl = control, metric = "ROC")
plot_confusion(lda_fit$pred$obs, lda_fit$pred$pred)

multiclass.roc(lda_fit$pred$obs, lda_fit$pred[3:14])
#roc_auc(lda_fit$pred$obs, SIT_TO_LIE, STAND_TO_LIE)
#roc_auc(lda_fit$pred$obs, lda_fit$pred$pred)

# Penalized Discriminant Analysis
library(mda)
pdaGrid <- expand.grid(lambda = seq(0.04, 0.2, 0.02))
pda_fit <- train(Activity ~ ., data = df, method = "pda", trControl = control, tuneGrid = pdaGrid, metric = "ROC")

plot_confusion(pda_fit$pred$obs, pda_fit$pred$pred)

multiclass.roc(pda_fit$pred$obs, pda_fit$pred[3:14])


#final_pda <- train(Activity ~ ., data = df, method = "pda", tuneGrid = c(lambda = 0.08))
pda_test <- predict(pda_fit$finalModel, df_validation[1:561])

plot_confusion(pda_test, df_validation$Activity)

lda_test <- predict(lda_fit, df_validation[1:561])
plot_confusion(lda_test, df_validation$Activity)
