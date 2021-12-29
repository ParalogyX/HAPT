###########################################################
#   HarvardX: PH125.9x  Data Science: Capstone            #
#         Smartphone-Based Recognition of                 #
#    Human Activities and Postural Transitions            #
###########################################################

##########################################################
#            READ THIS BEFORE RUN THE CODE!!!            #
#                                                        #
#  1. This code is made for R version 4.1.2 (2021-11-01) #
#      Check your R version (type "R.version.string"     #
#           in console) and update if needed             #
#                                                        #
#  2. Check Program Control section in the code:         #
#      it contains RETRAIN flag, if it set to TRUE,      #
#     all models will be retrained and saved to folder   #
#     "models". Running time will be ~10-14 hours,       #
#                 depends on computer;                   #    
#    If RETRAIN set to False, then, models will not be   #
#   retrained, but loaded from "models" folder, or,      #
#   if not found, downloaded from GitHub. Total size     #
#                     is ~1GB                            #
#                                                        #
#  3. Program checks if dataset is in the "data" folder  #
#          and downloads it if it is not a case.         #
#                Dataset size is 75.9MB.                 #
#                                                        #
#  4. Because program works with files in different      #
#         folders, it is important to set correct        #
#       working directory. It is done automatically,     #
#                   by using function:                   #
#      setwd(dirname(getActiveDocumentContext()$path))   #
#    Please, keep cursor in the source window during     #
#       code running to avoid wrong setting WD           #
#                                                        #
# Code itself is well commented as code, for theoretical #
#        background please look in report.pdf            #
##########################################################


# clear console output
cat("\014")

total_time_start <- unclass(Sys.time())

###########################################################
#                      LIBRARIES                          #
###########################################################
# library installations if needed:
if(!require(rstudioapi)) install.packages("rstudioapi")
if(!require(dplyr)) install.packages("dplyr")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(gridExtra)) install.packages("gridExtra")
if(!require(grid)) install.packages("grid")
if(!require(conflicted)) install.packages("conflicted")
if(!require(plyr)) install.packages("plyr")
if(!require(UBL)) install.packages("UBL")


# Load libraries
library(rstudioapi)
library(dplyr)
library(tidyverse)
library(gridExtra)
library(grid)
library(conflicted)
library(UBL)

###########################################################
#                      PROGRAM CONTROL                    #
###########################################################

# Set working directory to source file location
setwd(dirname(getActiveDocumentContext()$path))


# functions are masked in other package, but we need only this one
conflict_prefer("mutate", "dplyr")
conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")
conflict_prefer("arrange", "dplyr")

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
#                         Functions                       #
###########################################################

# Function to extract legend from ggplot
g_legend <- function(a.gplot){ 
  tmp <- ggplot_gtable(ggplot_build(a.gplot)) 
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box") 
  legend <- tmp$grobs[[leg]] 
  legend
} 

# Function for confusion plot output
# set prop = TRUE to get percentage instead of class output
plot_confusion <- function(truth, pred, name = "Confusion matrix", prop = FALSE){
  
  if (prop){
    xtab <- prop.table(table(pred, truth))
  }
  else {
    xtab <- table(pred, truth)
  }
  cm <- confusionMatrix(xtab)
  plt <- as.data.frame(cm$table)
  colnames(plt) <- c("Prediction", "Reference", "Freq")
  plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
  
  if (prop){
    plt <- plt %>% mutate(Freq = round(Freq * 100, 2))
  }
  
  
  ggplot(plt, aes(Prediction,Reference, fill= Freq)) +
    geom_tile() + geom_text(aes(label=Freq), size = 2.8) +
    scale_fill_gradient(low="white", high="#009194") +
    labs(x = "Reference",y = "Prediction") +
    scale_x_discrete(labels=levels(plt$Prediction)) +
    scale_y_discrete(labels=rev(levels(plt$Prediction))) +
    ggtitle(ifelse(prop, paste(name, "%"), name)) +
    theme(axis.text.x = element_text(angle = 45, vjust = 1.0, hjust=1))
}


###########################################################
#                          EDA                            #
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

# types of features and outcome
unique(lapply(df, class))
sum(lapply(df, class) == "numeric")
sum(lapply(df, class) == "factor")


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
  xlab("Minimal values of all features") + ylab("Count")

plot_max <- features_stat %>% 
  ggplot(aes(max)) +
  geom_histogram(binwidth = 0.02, col=rgb(0.1,0.4,0.5,0.7), fill=rgb(0.1,0.4,0.5,0.7)) + 
  theme_bw() +
  xlab("Maximum values of all features") + ylab("Count")

plot_median <- features_stat %>% 
  ggplot(aes(median)) +
  geom_histogram(binwidth = 0.05, col=rgb(0.1,0.4,0.5,0.7), fill=rgb(0.1,0.4,0.5,0.7)) + 
  theme_bw() +
  xlab("Median values of all features") + ylab("Count")

plot_mean <- features_stat %>% 
  ggplot(aes(mean)) +
  geom_histogram(binwidth = 0.05, col=rgb(0.1,0.4,0.5,0.7), fill=rgb(0.1,0.4,0.5,0.7)) + 
  theme_bw() +
  xlab("Mean values of all features") + ylab("Count")

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
    geom_density(aes_string(x = as.character(low_5[4])), colour = "orange") + 
    geom_density(aes_string(x = as.character(low_5[5])), colour = "magenta") +
    xlab("Value") + ylab("Density") +
    ggtitle(paste("class", label)) +
    theme_bw() +
    theme(plot.title = element_text(size = 15))
  
})
# add legend
# dummy trend to create legend
invisible(dummy_plot <- data.frame(feature = low_5, var1 = c(1,2,3,4,5), var2 = c(1,2,3,4,5)) %>% 
            ggplot(aes(var1, var2, color = feature)) +
            geom_point() + theme(legend.position="right") + 
            scale_color_manual(values=c("blue", "red", "green", "orange", "magenta")))

invisible(legend <- g_legend(dummy_plot))
trends[[nrow(activity_labels)+1]] <- legend

grid.arrange(grobs = trends, ncol=3, top = textGrob("Distribution of low variance features", x = 0.023, hjust = 0))

# remove unnecessary variables:
rm(dummy_plot, legend, low_5_df, trends, low_5, features_stat)


# Apply SMOTE before PCA
df_smote <- SmoteClassif(Activity ~ ., dat = df)
# plot outcomes distribution
df_smote %>% group_by(Activity) %>% mutate(n = n()) %>%
  ggplot(aes(reorder(Activity, -n))) +
  geom_bar(col=rgb(0.1,0.4,0.5,0.7), fill=rgb(0.1,0.4,0.5,0.7)) +
  xlab("Activity") + ylab("Count") +
  scale_y_continuous(breaks = seq(0,1500,100)) +
  ggtitle("Distribution of activities after SMOTE is applied") +
  theme_bw() +
  theme(axis.text.x=element_text(angle = -60, hjust = 0))

# how many samples of each class in df
sort(table(df_smote$Activity),decreasing=TRUE)


# visualization with PCA

pca <- prcomp(df_smote[-ncol(df_smote)], scale. = TRUE)

var_explained <- data.frame(PC= paste0("PC",1:(ncol(df_smote)-1)),
                            var_explained=(pca$sdev)^2/sum((pca$sdev)^2))

var_explained[1:20,] %>%
  ggplot(aes(x=factor(PC, levels = PC),y=var_explained * 100)) +
  geom_col(col=rgb(0.1,0.4,0.5,0.7), fill=rgb(0.1,0.4,0.5,0.7)) +
  xlab("Principal component") + ylab("% of variance explained") +
  ggtitle("Variance explained by Principal Components") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.0))


var_explained[1:400,] %>%
  ggplot(aes(x=as.numeric(factor(PC, levels = PC)), y=cumsum(var_explained * 100), group = 1)) +
  geom_point(alpha = 0.5, size = 1) +
  geom_line() +
  xlab("Principal component") + ylab("% of variance explained") +
  scale_y_continuous(breaks = seq(30, 100, 5)) +
  #scale_x_discrete(labels = c(1, 50, 100)) + 
  ggtitle("Cumulative variance explained by Principal Components") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.0))


# plot with two first PC's
df_pca <- as.data.frame(pca$x) %>% cbind(df_smote[ncol(df_smote)])

# visualize classification of two PC

df_pca %>% ggplot(aes(x = PC1, y = PC2, color = Activity)) +
  geom_point() +
  ggtitle("First two principal components classification plot")+
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.0))

# for better visualization we will introduce some combined classes to distinguish activities
static <- c("LAYING", "SITTING", "STANDING")
moving <- c("WALKING", "WALKING_DOWNSTAIRS", "WALKING_UPSTAIRS")
postural_trans <- activity_labels$Activity[!activity_labels$Activity %in% c(static, moving)]

df_pca %>% mutate(Activity_type = ifelse(Activity %in% static, "Static", 
                                  ifelse(Activity %in% moving, "Moving", 
                                         "Postural trans"))) %>%
  ggplot(aes(x = PC1, y = PC2, color = Activity_type)) +
  geom_point() +
  ggtitle("First two principal components for activity type classification")+
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.0))

# check classification inside groups
# Static activities
p1 <- df_pca %>% filter(Activity %in% static) %>%
  ggplot(aes(x = PC1, y = PC2, color = Activity)) +
  geom_point() +
  theme_bw() +
  theme(legend.position = "none")

p2 <- df_pca %>% filter(Activity %in% static) %>%
  ggplot(aes(x = PC3, y = PC4, color = Activity)) +
  geom_point() +
  theme_bw() +
  theme(legend.position = "none")

p3 <- df_pca %>% filter(Activity %in% static) %>%
  ggplot(aes(x = PC5, y = PC6, color = Activity)) +
  geom_point() +
  theme_bw() +
  theme(legend.position = "none")

p4 <- df_pca %>% filter(Activity %in% static) %>%
  ggplot(aes(x = PC7, y = PC8, color = Activity)) +
  geom_point() +
  theme_bw() +
  theme(legend.position = "none")

# dummy plot for legend
invisible(dummy_plot <- df_pca %>% filter(Activity %in% static) %>%
            ggplot(aes(x = PC1, y = PC2, color = Activity)) +
            geom_point() +
            theme_bw() +
            theme(legend.position = "bottom"))
invisible(legend <- g_legend(dummy_plot))


grid.arrange(arrangeGrob(p1, p2, p3, p4, ncol=2), legend, nrow = 2, heights = c(10, 1),  top = textGrob("Static activities on eight PC's", x = 0.023, hjust = 0))


# Moving activities
p1 <- df_pca %>% filter(Activity %in% moving) %>%
  ggplot(aes(x = PC1, y = PC2, color = Activity)) +
  geom_point() +
  theme_bw() +
  theme(legend.position = "none")

p2 <- df_pca %>% filter(Activity %in% moving) %>%
  ggplot(aes(x = PC3, y = PC4, color = Activity)) +
  geom_point() +
  theme_bw() +
  theme(legend.position = "none")

p3 <- df_pca %>% filter(Activity %in% moving) %>%
  ggplot(aes(x = PC5, y = PC6, color = Activity)) +
  geom_point() +
  theme_bw() +
  theme(legend.position = "none")

p4 <- df_pca %>% filter(Activity %in% moving) %>%
  ggplot(aes(x = PC7, y = PC8, color = Activity)) +
  geom_point() +
  theme_bw() +
  theme(legend.position = "none")

# dummy plot for legend
invisible(dummy_plot <- df_pca %>% filter(Activity %in% moving) %>%
            ggplot(aes(x = PC1, y = PC2, color = Activity)) +
            geom_point() +
            theme_bw() +
            theme(legend.position = "bottom"))
invisible(legend <- g_legend(dummy_plot))


grid.arrange(arrangeGrob(p1, p2, p3, p4, ncol=2), legend, nrow = 2, heights = c(10, 1),  top = textGrob("Moving activities on eight PC's", x = 0.023, hjust = 0))

# Postural transitions
p1 <- df_pca %>% filter(Activity %in% postural_trans) %>%
  ggplot(aes(x = PC1, y = PC2, color = Activity)) +
  geom_point() +
  theme_bw() +
  theme(legend.position = "none")

p2 <- df_pca %>% filter(Activity %in% postural_trans) %>%
  ggplot(aes(x = PC3, y = PC4, color = Activity)) +
  geom_point() +
  theme_bw() +
  theme(legend.position = "none")

p3 <- df_pca %>% filter(Activity %in% postural_trans) %>%
  ggplot(aes(x = PC5, y = PC6, color = Activity)) +
  geom_point() +
  theme_bw() +
  theme(legend.position = "none")

p4 <- df_pca %>% filter(Activity %in% postural_trans) %>%
  ggplot(aes(x = PC7, y = PC8, color = Activity)) +
  geom_point() +
  theme_bw() +
  theme(legend.position = "none")

# dummy plot for legend
invisible(dummy_plot <- df_pca %>% filter(Activity %in% postural_trans) %>%
            ggplot(aes(x = PC1, y = PC2, color = Activity)) +
            geom_point() +
            theme_bw() +
            theme(legend.position = "bottom"))
invisible(legend <- g_legend(dummy_plot))


grid.arrange(arrangeGrob(p1, p2, p3, p4, ncol=2), legend, nrow = 2, heights = c(10, 1),  top = textGrob("Postural transitions on eight PC's", x = 0.023, hjust = 0))


# Check higher dimensions
p1 <- df_pca %>% filter(Activity %in% postural_trans) %>%
  ggplot(aes(x = PC9, y = PC10, color = Activity)) +
  geom_point() +
  theme_bw() +
  theme(legend.position = "none")

p2 <- df_pca %>% filter(Activity %in% postural_trans) %>%
  ggplot(aes(x = PC11, y = PC12, color = Activity)) +
  geom_point() +
  theme_bw() +
  theme(legend.position = "none")

p3 <- df_pca %>% filter(Activity %in% postural_trans) %>%
  ggplot(aes(x = PC13, y = PC14, color = Activity)) +
  geom_point() +
  theme_bw() +
  theme(legend.position = "none")

p4 <- df_pca %>% filter(Activity %in% postural_trans) %>%
  ggplot(aes(x = PC15, y = PC16, color = Activity)) +
  geom_point() +
  theme_bw() +
  theme(legend.position = "none")

# dummy plot for legend
invisible(dummy_plot <- df_pca %>% filter(Activity %in% postural_trans) %>%
            ggplot(aes(x = PC1, y = PC2, color = Activity)) +
            geom_point() +
            theme_bw() +
            theme(legend.position = "bottom"))
invisible(legend <- g_legend(dummy_plot))


grid.arrange(arrangeGrob(p1, p2, p3, p4, ncol=2), legend, nrow = 2, heights = c(10, 1),  top = textGrob("Postural transitions on PC's 9-16", x = 0.023, hjust = 0))

# remove unnecessary variables:
rm(p1, p2, p3, p4, dummy_plot, legend)






###########################################################
#                      Model building                     #
###########################################################
# apply the same pca to non SMOTE df
# keep 100 features from df_pca
df_pca_or <- as.data.frame(predict(pca, newdata = df[1:ncol(df)-1]))
df_pca_or <- df_pca_or[1:100] %>% cbind(df[ncol(df)])

# If "Data" folder is not exist, create it
if (!dir.exists("./models")) {
  dir.create("./models")
}


if(!require(caret)) install.packages("caret", dependencies = TRUE)
library(caret)

# load libraries only if retrain, to save time: it can take up to 1 hour to install it on clean R
if(RETRAIN){
  if(!require(kknn)) install.packages("kknn", dependencies = TRUE)
  if(!require(mda)) install.packages("mda", dependencies = TRUE)
  if(!require(nnet)) install.packages("nnet", dependencies = TRUE)
  if(!require(gbm)) install.packages("gbm", dependencies = TRUE)
  if(!require(plyr)) install.packages("plyr", dependencies = TRUE)
  conflict_prefer("mutate", "dplyr")
  conflict_prefer("arrange", "dplyr")
  if(!require(xgboost)) install.packages("xgboost", dependencies = TRUE)
  if(!require(e1071)) install.packages("e1071", dependencies = TRUE)
  if(!require(randomForest)) install.packages("randomForest", dependencies = TRUE)
  if(!require(foreach)) install.packages("foreach", dependencies = TRUE)
  if(!require(import)) install.packages("import", dependencies = TRUE)
  
  
  library(kknn)
  library(mda)
  library(nnet)
  library(gbm)
  library(xgboost)
  library(randomForest)
}


# Train control and metric for all models to test

control <- trainControl(method="cv", number = 5, classProbs= TRUE, summaryFunction = multiClassSummary, 
                        savePredictions = "final",
                        search="grid",
                        verbose = PRINT_DEBUG)


metric <- "Mean_Balanced_Accuracy"


#################
#   Train knn   #
#################

file_name <- "./models//kknn.rds"
if (RETRAIN) {
  time_start <- unclass(Sys.time())
  fit_kknn <- train(Activity ~ ., data = df_pca_or, method = "kknn", metric = metric,
                   trControl = control)
  time_end <- unclass(Sys.time())
  kknn_time <- time_end - time_start
  # If "models" folder is not exist, create it
  if (!dir.exists("./models")) {dir.create("./models")}
  # save fits
  saveRDS(fit_kknn, file_name)
} else {
  # if file is not found, download it.
  if (!file.exists(file_name)) {
    download.file("https://github.com/ParalogyX/HAPT/releases/download/trained/kknn.rds", file_name)
  }
  # read from file
  fit_kknn <- readRDS(file_name)
}


#################
#   Train pda   #
#################

file_name <- "./models//pda.rds"
if (RETRAIN) {
  time_start <- unclass(Sys.time())
  fit_pda <- train(Activity ~ ., data = df_pca_or, method = "pda", metric = metric,
                   trControl = control)
  time_end <- unclass(Sys.time())
  pda_time <- time_end - time_start
  # If "models" folder is not exist, create it
  if (!dir.exists("./models")) {dir.create("./models")}
  # save fits
  saveRDS(fit_pda, file_name)
} else {
  # if file is not found, download it.
  if (!file.exists(file_name)) {
    download.file("https://github.com/ParalogyX/HAPT/releases/download/trained/pda.rds", file_name)
  }
  # read from file
  fit_pda <- readRDS(file_name)
}

##################
# Train multinom #
##################


file_name <- "./models//multinom.rds"
if (RETRAIN) {
  time_start <- unclass(Sys.time())
  fit_multinom <- train(Activity ~ ., data = df_pca_or, method = "multinom", metric = metric,
                   trControl = control, MaxNWts = 15000)
  time_end <- unclass(Sys.time())
  multinom_time <- time_end - time_start
  # If "models" folder is not exist, create it
  if (!dir.exists("./models")) {dir.create("./models")}
  # save fits
  saveRDS(fit_multinom, file_name)
} else {
  # if file is not found, download it.
  if (!file.exists(file_name)) {
    download.file("https://github.com/ParalogyX/HAPT/releases/download/trained/multinom.rds", file_name)
  }
  # read from file
  fit_multinom <- readRDS(file_name)
}


##################
#   Train gbm    #
##################

file_name <- "./models//gbm.rds"
if (RETRAIN) {
  time_start <- unclass(Sys.time())
  fit_gbm <- train(Activity ~ ., data = df_pca_or, method = "gbm", metric = metric,
                        trControl = control, 
                        bag.fraction = 1, nTrain = 3000)
  time_end <- unclass(Sys.time())
  gbm_time <- time_end - time_start
  # If "models" folder is not exist, create it
  if (!dir.exists("./models")) {dir.create("./models")}
  # save fits
  saveRDS(fit_gbm, file_name)
} else {
  # if file is not found, download it.
  if (!file.exists(file_name)) {
    download.file("https://github.com/ParalogyX/HAPT/releases/download/trained/gbm.rds", file_name)
  }
  # read from file
  fit_gbm <- readRDS(file_name)
}

##################
# Train xgbTree  #
##################

file_name <- "./models//xgbTree.rds"
if (RETRAIN) {
  time_start <- unclass(Sys.time())
  fit_xgbTree <- train(Activity ~ ., data = df_pca_or, method = "xgbTree", metric = metric,
                   trControl = control)
  time_end <- unclass(Sys.time())
  xgbTree_time <- time_end - time_start
  # If "models" folder is not exist, create it
  if (!dir.exists("./models")) {dir.create("./models")}
  # save fits
  saveRDS(fit_xgbTree, file_name)
} else {
  # if file is not found, download it.
  if (!file.exists(file_name)) {
    download.file("https://github.com/ParalogyX/HAPT/releases/download/trained/xgbTree.rds", file_name)
  }
  # read from file
  fit_xgbTree <- readRDS(file_name)
}

##################
#  Train parRF   #
##################

file_name <- "./models//parRF.rds"
if (RETRAIN) {
  time_start <- unclass(Sys.time())
  fit_parRF <- train(Activity ~ ., data = df_pca_or, method = "parRF", metric = metric,
                       trControl = control)
  time_end <- unclass(Sys.time())
  parRF_time <- time_end - time_start
  # If "models" folder is not exist, create it
  if (!dir.exists("./models")) {dir.create("./models")}
  # save fits
  saveRDS(fit_parRF, file_name)
} else {
  # if file is not found, download it.
  if (!file.exists(file_name)) {
    download.file("https://github.com/ParalogyX/HAPT/releases/download/trained/parRF.rds", file_name)
  }
  # read from file
  fit_parRF <- readRDS(file_name)
}

##################
#  Train nnet   #
##################

file_name <- "./models//nnet.rds"
if (RETRAIN) {
  time_start <- unclass(Sys.time())
  fit_nnet <- train(Activity ~ ., data = df_pca_or, method = "nnet", metric = metric,
                     trControl = control, MaxNWts = 15000)
  time_end <- unclass(Sys.time())
  nnet_time <- time_end - time_start
  # If "models" folder is not exist, create it
  if (!dir.exists("./models")) {dir.create("./models")}
  # save fits
  saveRDS(fit_nnet, file_name)
} else {
  # if file is not found, download it.
  if (!file.exists(file_name)) {
    download.file("https://github.com/ParalogyX/HAPT/releases/download/trained/nnet.rds", file_name)
  }
  # read from file
  fit_nnet <- readRDS(file_name)
}




######################
#  Results analysis  #
######################
models <- c("kknn", "pda", "multinom", "gbm", "xgbTree", "parRF", "nnet")
fits <- list(fit_kknn, fit_pda, fit_multinom, 
             fit_gbm,
             fit_xgbTree, fit_parRF, fit_nnet)
names(fits) <- models

result_df <- data.frame(t(sapply(models, function(n){
  pos_max <- which.max(fits[[n]]$results$Mean_Balanced_Accuracy)
  c(fits[[n]]$method, 
    fits[[n]]$results$Mean_Balanced_Accuracy[pos_max],
    fits[[n]]$results$Kappa[pos_max],
    fits[[n]]$results$Accuracy[pos_max],
    fits[[n]]$times$everything["elapsed"])
})))

colnames(result_df) <- c("Name", "Mean_Balanced_Accuracy", "Kappa", "Accuracy", "Time")
result_df[2:5] <- lapply(result_df[2:5],as.numeric)
result_df[, 2:5] <- round(result_df[, 2:5], digits = 4)

result_df

# plot kappa
results_kappa <- data.frame(t(sapply(models, function(n){
  pos_max <- which.max(fits[[n]]$results$Kappa)
  c(fits[[n]]$method, fits[[n]]$results$Kappa[pos_max], fits[[n]]$times$everything["elapsed"])
})))

colnames(results_kappa) <- c("Name", "Kappa", "Time")


results_kappa %>% mutate(Time = as.numeric(Time) / 60, Kappa = as.numeric(Kappa)) %>% ggplot(aes(x = Time, y = Kappa, color = Name))+
  geom_point(size = 1) + geom_text(aes(label = Name), check_overlap = TRUE) + ggtitle("Full dataset training kappa")


#plot AUC
results_AUC <- data.frame(t(sapply(models, function(n){
  pos_max <- which.max(fits[[n]]$results$AUC)
  c(fits[[n]]$method, fits[[n]]$results$AUC[pos_max], fits[[n]]$times$everything["elapsed"])
})))

colnames(results_AUC) <- c("Name", "AUC", "Time")


results_AUC %>% mutate(Time = as.numeric(Time) / 60, AUC = as.numeric(AUC)) %>% ggplot(aes(x = Time, y = AUC, color = Name))+
  geom_point(size = 1) + geom_text(aes(label = Name), check_overlap = TRUE) + ggtitle("Full dataset training AUC")




# plot accuracy
results_Acc <- data.frame(t(sapply(models, function(n){
  pos_max <- which.max(fits[[n]]$results$Accuracy)
  c(fits[[n]]$method, fits[[n]]$results$Accuracy[pos_max], fits[[n]]$times$everything["elapsed"])
})))

colnames(results_Acc) <- c("Name", "Accuracy", "Time")


results_Acc %>% mutate(Time = as.numeric(Time) / 60, Accuracy = as.numeric(Accuracy)) %>% ggplot(aes(x = Time, y = Accuracy, color = Name))+
  geom_point(size = 1) + geom_text(aes(label = Name), check_overlap = TRUE) + ggtitle("Full dataset training Accuracy")

# plot bal accuracy
results_bal_Acc <- data.frame(t(sapply(models, function(n){
  pos_max <- which.max(fits[[n]]$results$Mean_Balanced_Accuracy)
  c(fits[[n]]$method, fits[[n]]$results$Mean_Balanced_Accuracy[pos_max], fits[[n]]$times$everything["elapsed"])
})))

colnames(results_bal_Acc) <- c("Name", "Mean_Balanced_Accuracy", "Time")


results_bal_Acc %>% mutate(Time = as.numeric(Time) / 60, Mean_Balanced_Accuracy = as.numeric(Mean_Balanced_Accuracy)) %>% 
  ggplot(aes(x = Time, y = Mean_Balanced_Accuracy, color = Name))+
  geom_point(size = 1) + geom_text(aes(label = Name), check_overlap = TRUE) + ggtitle("Full dataset training Mean_Balanced_Accuracy")




# plot train confusion matrices
lapply(models, function(model){
  plot_confusion(fits[[model]]$pred$obs, fits[[model]]$pred$pred, name = paste(model, "train"))
})

modelLookup("multinom")

# plot decay vs metrics
fit_multinom$results %>% ggplot(aes(x = decay)) +
  geom_line(aes(y = Mean_Balanced_Accuracy), color = "blue") +
  ggtitle("Multinom decay vs balanced accuracy") +
  theme_bw() +
  theme(plot.title = element_text(size = 15))

# Cross-validation: bootstrap
control <- trainControl(method="boot", classProbs= TRUE, summaryFunction = multiClassSummary, 
                        savePredictions = "final",
                        search="grid",
                        verbose = PRINT_DEBUG)

# extended multinom grid
multinom_grid <- expand.grid(decay = c(0.1, 0.5, 1, 3, 5, 8, 10, 13, 15))

file_name <- "./models//multinom_expand_grid.rds"

if (RETRAIN) {
  time_start <- unclass(Sys.time())
  fit_multinom_grid <- train(Activity ~ ., data = df_pca_or, method = "multinom", metric = metric,
                             trControl = control, MaxNWts = 15000, tuneGrid = multinom_grid)
  time_end <- unclass(Sys.time())
  
  multinom_time_grid <- time_end - time_start
  saveRDS(fit_multinom_grid, file_name)
} else {
  # if file is not found, download it.
  if (!file.exists(file_name)) {
    download.file("https://github.com/ParalogyX/HAPT/releases/download/trained/multinom_expand_grid.rds", file_name)
  }
  # read from file
  fit_multinom_grid <- readRDS(file_name)
}

print(fit_multinom$results$Mean_Balanced_Accuracy)
print(fit_multinom_grid$results$Mean_Balanced_Accuracy)
# plot decay vs metrics
fit_multinom_grid$results %>% ggplot(aes(x = decay)) +
  geom_line(aes(y = Mean_Balanced_Accuracy), color = "blue") +
  ggtitle("Multinom decay vs balanced accuracy") +
  theme_bw() +
  theme(plot.title = element_text(size = 15))


plot_confusion(fit_multinom_grid$pred$obs, fit_multinom_grid$pred$pred, name = "Multinom train")




# build final model

decay = fit_multinom_grid$bestTune$decay



multinomGrid_fin <-  expand.grid(decay = decay)
control <- trainControl(method="boot", classProbs= TRUE, summaryFunction = multiClassSummary, 
                        savePredictions = "final",
                        search="grid",
                        verbose = PRINT_DEBUG)

file_name <- "./models//multinom_final_pca_100.rds"

if (RETRAIN) {
  time_start <- unclass(Sys.time())
  fit_multinom_fin_100 <- train(Activity ~ ., data = df_pca_or, method = "multinom", metric = metric,
                             trControl = control, MaxNWts = 15000, maxit = 500, tuneGrid = multinomGrid_fin)
  time_end <- unclass(Sys.time())
  
  multinom_time_fin_100 <- time_end - time_start
  saveRDS(fit_multinom_fin_100, file_name)
} else {
  # if file is not found, download it.
  if (!file.exists(file_name)) {
    download.file("https://github.com/ParalogyX/HAPT/releases/download/trained/multinom_final_pca_100.rds", file_name)
  }
  # read from file
  fit_multinom_fin_100 <- readRDS(file_name)
}

# add results to the table
result_multinom <- data.frame("multinom_optimal_PCA100",
                     fit_multinom_fin_100$results$Mean_Balanced_Accuracy,
                     fit_multinom_fin_100$results$Kappa,
                     fit_multinom_fin_100$results$Accuracy,
                     fit_multinom_fin_100$times$everything["elapsed"])

colnames(result_multinom) <- c("Name", "Mean_Balanced_Accuracy", "Kappa", "Accuracy", "Time")

result_multinom[2:5] <- lapply(result_multinom[2:5],as.numeric)
result_multinom[, 2:5] <- round(result_multinom[, 2:5], digits = 4)
result_df <- rbind(result_df, result_multinom)

result_df

plot_confusion(fit_multinom_fin_100$pred$obs, fit_multinom_fin_100$pred$pred, name = "Multinom (100 PC) 500 iter train")


# try with extended pca (200 PC's)
# keep 200 features from df_pca
df_pca_or <- as.data.frame(predict(pca, newdata = df[1:561]))
df_pca_or <- df_pca_or[1:200] %>% cbind(df[562])

file_name <- "./models//multinom_final_pca_200.rds"

if (RETRAIN) {
  time_start <- unclass(Sys.time())
  fit_multinom_fin_200 <- train(Activity ~ ., data = df_pca_or, method = "multinom", metric = metric,
                            trControl = control, MaxNWts = 15000, maxit = 500, tuneGrid = multinomGrid_fin)
  time_end <- unclass(Sys.time())
  
  multinom_time_fin_200 <- time_end - time_start
  saveRDS(fit_multinom_fin_200, file_name)
} else {
  # if file is not found, download it.
  if (!file.exists(file_name)) {
    download.file("https://github.com/ParalogyX/HAPT/releases/download/trained/multinom_final_pca_200.rds", file_name)
  }
  # read from file
  fit_multinom_fin_200 <- readRDS(file_name)
}

# add results to the table
result_multinom <- data.frame("multinom_optimal_PCA200",
                              fit_multinom_fin_200$results$Mean_Balanced_Accuracy,
                              fit_multinom_fin_200$results$Kappa,
                              fit_multinom_fin_200$results$Accuracy,
                              fit_multinom_fin_200$times$everything["elapsed"])

colnames(result_multinom) <- c("Name", "Mean_Balanced_Accuracy", "Kappa", "Accuracy", "Time")

result_multinom[2:5] <- lapply(result_multinom[2:5],as.numeric)

result_multinom[, 2:5] <- round(result_multinom[, 2:5], digits = 4)

result_df <- rbind(result_df, result_multinom)

result_df


plot_confusion(fit_multinom_fin_200$pred$obs, fit_multinom_fin_200$pred$pred, name = "Multinom (200 PC) 500 iter train")

# try with original dataset
file_name <- "./models//multinom_final_orig.rds"

if (RETRAIN) {
  time_start <- unclass(Sys.time())
  fit_multinom_orig <- train(Activity ~ ., data = df, method = "multinom", metric = metric,
                                trControl = control, MaxNWts = 15000, maxit = 500, tuneGrid = multinomGrid_fin)
  time_end <- unclass(Sys.time())
  
  multinom_time_orig <- time_end - time_start
  saveRDS(fit_multinom_orig, file_name)
} else {
  # if file is not found, download it.
  if (!file.exists(file_name)) {
    download.file("https://github.com/ParalogyX/HAPT/releases/download/trained/multinom_final_orig.rds", file_name)
  }
  # read from file
  fit_multinom_orig <- readRDS(file_name)
}


# add results to the table
result_multinom <- data.frame("multinom_optimal_orig",
                              fit_multinom_orig$results$Mean_Balanced_Accuracy,
                              fit_multinom_orig$results$Kappa,
                              fit_multinom_orig$results$Accuracy,
                              fit_multinom_orig$times$everything["elapsed"])

colnames(result_multinom) <- c("Name", "Mean_Balanced_Accuracy", "Kappa", "Accuracy", "Time")

result_multinom[2:5] <- lapply(result_multinom[2:5],as.numeric)
result_multinom[, 2:5] <- round(result_multinom[, 2:5], digits = 4)
result_df <- rbind(result_df, result_multinom)

result_df

plot_confusion(fit_multinom_orig$pred$obs, fit_multinom_orig$pred$pred, name = "Multinom (orig dataset) 500 iter train")

# Final model function
multinom_final_predict <- function(features_space){
  val_pca <- predict(pca, newdata = features_space)
  predict(fit_multinom_fin_200, val_pca[,1:200])
}


###########################################################
#                     Final validation                    #
###########################################################

truth <- df_validation$Activity
pred <- multinom_final_predict(df_validation[1:561]) 

xtab <- table(pred, truth)

cm <- confusionMatrix(xtab)

xtab <- prop.table(table(pred, truth))
cm_prop <- confusionMatrix(xtab)

result_final_val <- data.frame("multinom_final_validation",
                              mean(cm$byClass[,"Balanced Accuracy"]),
                              cm$overall["Kappa"],
                              cm$overall["Accuracy"],
                              "Not applicable")

colnames(result_final_val) <- c("Name", "Mean_Balanced_Accuracy", "Kappa", "Accuracy", "Time")

result_final_val[2:4] <- lapply(result_final_val[2:4],as.numeric)
result_final_val[, 2:4] <- round(result_final_val[, 2:4], digits = 4)
result_df <- rbind(result_df, result_final_val)

result_df


plot_confusion(df_validation$Activity, multinom_final_predict(df_validation[1:561]), name = "Validation confusion matrix")



total_time_end <- unclass(Sys.time())
total_time <- total_time_end - total_time_start