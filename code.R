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
if(!require(conflicted)) install.packages("conflicted")

# Load libraries
library(rstudioapi)
library(tidyverse)
library(gridExtra)
library(grid)
library(conflicted)


###########################################################
#                      PROGRAM CONTROL                    #
###########################################################

# Set working directory to source file location
setwd(dirname(getActiveDocumentContext()$path))


# function select is masked in other package, but we need only this one
conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

# Program controls
RETRAIN <- T       # TRUE: models will be retrained; FALSE: trained models will be loaded from files
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

# Function to extract legend
g_legend <- function(a.gplot){ 
  tmp <- ggplot_gtable(ggplot_build(a.gplot)) 
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box") 
  legend <- tmp$grobs[[leg]] 
  legend
} 

# Function for confusion plot
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
    geom_tile() + geom_text(aes(label=Freq)) +
    scale_fill_gradient(low="white", high="#009194") +
    labs(x = "Reference",y = "Prediction") +
    scale_x_discrete(labels=levels(plt$Prediction)) +
    scale_y_discrete(labels=rev(levels(plt$Prediction))) +
    ggtitle(ifelse(prop, paste(name, "%"), name)) +
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

invisible(legend <- g_legend(dummy_plot))
trends[[nrow(activity_labels)+1]] <- legend

grid.arrange(grobs = trends, ncol=3, top = textGrob("Distribution of low variance features", x = 0.023, hjust = 0))

# remove unnecessary variables:
rm(dummy_plot, legend, low_5_df, trends, low_5, features_stat)

# find most correlated with outcome features



# visualization with PCA

pca <- prcomp(df[-ncol(df)], scale. = TRUE)

var_explained <- data.frame(PC= paste0("PC",1:(ncol(df)-1)),
                            var_explained=(pca$sdev)^2/sum((pca$sdev)^2))

var_explained[1:20,] %>%
  ggplot(aes(x=factor(PC, levels = PC),y=var_explained * 100)) +
  geom_col(col=rgb(0.1,0.4,0.5,0.7), fill=rgb(0.1,0.4,0.5,0.7)) +
  xlab("Principal component") + ylab("% of variance explained") +
  ggtitle("Variance explained by Principal Components") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.0))


var_explained[1:100,] %>%
  ggplot(aes(x=as.numeric(factor(PC, levels = PC)), y=cumsum(var_explained * 100), group = 1)) +
  geom_point(alpha = 0.5, size = 1) +
  geom_line() +
  xlab("Principal component") + ylab("% of variance explained") +
  #scale_x_discrete(labels = c(1, 50, 100)) + 
  ggtitle("Cumulative variance explained by Principal Components") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.0))


# plot with two first PC's
df_pca <- as.data.frame(pca$x) %>% cbind(df[ncol(df)])

# visualize  classification of two PC

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
  theme(legend.position = "none")

p2 <- df_pca %>% filter(Activity %in% static) %>%
  ggplot(aes(x = PC3, y = PC4, color = Activity)) +
  geom_point() +
  theme(legend.position = "none")

p3 <- df_pca %>% filter(Activity %in% static) %>%
  ggplot(aes(x = PC5, y = PC6, color = Activity)) +
  geom_point() +
  theme(legend.position = "none")

p4 <- df_pca %>% filter(Activity %in% static) %>%
  ggplot(aes(x = PC7, y = PC8, color = Activity)) +
  geom_point() +
  theme(legend.position = "none")

# dummy plot for legend
invisible(dummy_plot <- df_pca %>% filter(Activity %in% static) %>%
            ggplot(aes(x = PC1, y = PC2, color = Activity)) +
            geom_point() +
            theme(legend.position = "bottom"))
invisible(legend <- g_legend(dummy_plot))


grid.arrange(arrangeGrob(p1, p2, p3, p4, ncol=2), legend, nrow = 2, heights = c(10, 1),  top = textGrob("Static activities on eight PC's", x = 0.023, hjust = 0))


# Moving activities
p1 <- df_pca %>% filter(Activity %in% moving) %>%
  ggplot(aes(x = PC1, y = PC2, color = Activity)) +
  geom_point() +
  theme(legend.position = "none")

p2 <- df_pca %>% filter(Activity %in% moving) %>%
  ggplot(aes(x = PC3, y = PC4, color = Activity)) +
  geom_point() +
  theme(legend.position = "none")

p3 <- df_pca %>% filter(Activity %in% moving) %>%
  ggplot(aes(x = PC5, y = PC6, color = Activity)) +
  geom_point() +
  theme(legend.position = "none")

p4 <- df_pca %>% filter(Activity %in% moving) %>%
  ggplot(aes(x = PC7, y = PC8, color = Activity)) +
  geom_point() +
  theme(legend.position = "none")

# dummy plot for legend
invisible(dummy_plot <- df_pca %>% filter(Activity %in% moving) %>%
            ggplot(aes(x = PC1, y = PC2, color = Activity)) +
            geom_point() +
            theme(legend.position = "bottom"))
invisible(legend <- g_legend(dummy_plot))


grid.arrange(arrangeGrob(p1, p2, p3, p4, ncol=2), legend, nrow = 2, heights = c(10, 1),  top = textGrob("Moving activities on eight PC's", x = 0.023, hjust = 0))

# Postural transitions
p1 <- df_pca %>% filter(Activity %in% postural_trans) %>%
  ggplot(aes(x = PC1, y = PC2, color = Activity)) +
  geom_point() +
  theme(legend.position = "none")

p2 <- df_pca %>% filter(Activity %in% postural_trans) %>%
  ggplot(aes(x = PC3, y = PC4, color = Activity)) +
  geom_point() +
  theme(legend.position = "none")

p3 <- df_pca %>% filter(Activity %in% postural_trans) %>%
  ggplot(aes(x = PC5, y = PC6, color = Activity)) +
  geom_point() +
  theme(legend.position = "none")

p4 <- df_pca %>% filter(Activity %in% postural_trans) %>%
  ggplot(aes(x = PC7, y = PC8, color = Activity)) +
  geom_point() +
  theme(legend.position = "none")

# dummy plot for legend
invisible(dummy_plot <- df_pca %>% filter(Activity %in% postural_trans) %>%
            ggplot(aes(x = PC1, y = PC2, color = Activity)) +
            geom_point() +
            theme(legend.position = "bottom"))
invisible(legend <- g_legend(dummy_plot))


grid.arrange(arrangeGrob(p1, p2, p3, p4, ncol=2), legend, nrow = 2, heights = c(10, 1),  top = textGrob("Postural transitions on eight PC's", x = 0.023, hjust = 0))

# It doesn't look very distinguishable

# Check higher dimensions
p1 <- df_pca %>% filter(Activity %in% postural_trans) %>%
  ggplot(aes(x = PC9, y = PC10, color = Activity)) +
  geom_point() +
  theme(legend.position = "none")

p2 <- df_pca %>% filter(Activity %in% postural_trans) %>%
  ggplot(aes(x = PC11, y = PC12, color = Activity)) +
  geom_point() +
  theme(legend.position = "none")

p3 <- df_pca %>% filter(Activity %in% postural_trans) %>%
  ggplot(aes(x = PC13, y = PC14, color = Activity)) +
  geom_point() +
  theme(legend.position = "none")

p4 <- df_pca %>% filter(Activity %in% postural_trans) %>%
  ggplot(aes(x = PC15, y = PC16, color = Activity)) +
  geom_point() +
  theme(legend.position = "none")

# dummy plot for legend
invisible(dummy_plot <- df_pca %>% filter(Activity %in% postural_trans) %>%
            ggplot(aes(x = PC1, y = PC2, color = Activity)) +
            geom_point() +
            theme(legend.position = "bottom"))
invisible(legend <- g_legend(dummy_plot))


grid.arrange(arrangeGrob(p1, p2, p3, p4, ncol=2), legend, nrow = 2, heights = c(10, 1),  top = textGrob("Postural transitions on PC's 9-16", x = 0.023, hjust = 0))

rm(p1, p2, p3, p4, dummy_plot, legend, df_pca, pca, var_explained)
stop("Do not train yet")

###########################################################
#                      Model building                     #
###########################################################


# Metric will be Cohen’s Kappa score for multiclass classification
# It is a good single value metric for unbalanced multiclass problems.

if(!require(caret)) install.packages("caret")
if(!require(kknn)) install.packages("kknn")
if(!require(HDclassif)) install.packages("HDclassif")
if(!require(earth)) install.packages("earth")
if(!require(rrcov)) install.packages("rrcov")
if(!require(rrcovHD)) install.packages("rrcovHD")
if(!require(xgboost)) install.packages("xgboost")


library(caret)
library(kknn)
library(HDclassif)
library(earth)
library(rrcov)
library(rrcovHD)
library(xgboost)

models <- c("kknn", "pda", "slda", "hdda", "pam", "multinom", "C5.0Tree", "CSimca", "rf", "pls", "earth", "xgbTree")
models <- c("kknn", "pda", "multinom", "xgbTree")

control <- trainControl(method="cv", number=10, classProbs= TRUE, summaryFunction = multiClassSummary, savePredictions = "final",
                        verbose = PRINT_DEBUG)

metric <- "Kappa"

file_name <- "./models//multiple_fits.rds"

if (RETRAIN) {

  time_start <- unclass(Sys.time())
  
  fits <- lapply(models, function(model){ 
    print(model)
    if (model != "multinom") {
      train(Activity ~ ., data = df, method = model, metric = metric, trControl = control)
    } 
    else {
      train(Activity ~ ., data = df, method = model, metric = metric, trControl = control, MaxNWts = 15000)
    }
  }) 
  
  time_end <- unclass(Sys.time())
  
  all_total_time <- time_end - time_start
  
  names(fits) <- models

  # If "models" folder is not exist, create it
  if (!dir.exists("./models")) {dir.create("./models")}
  # save fits
  saveRDS(fits, file_name)
} else {
    # if file is not found, stop and message. 
    if (!file.exists(file_name)) {stop("File not found. Rerun code with RETRAIN = TRUE")} 
    # read from file
    else {fits <- readRDS(file_name)}
}
  

#ensemble_total_time/3600  # ~4 hours
# print results

results <- data.frame(t(sapply(models, function(n){
  pos_max <- which.max(fits[[n]]$results$Kappa)
  c(fits[[n]]$method, fits[[n]]$results$Kappa[pos_max], fits[[n]]$times$everything["elapsed"])
})))

colnames(results) <- c("Name", "Kappa", "Time")


results %>% mutate(Time = as.numeric(Time) / 60, Kappa = as.numeric(Kappa)) %>% ggplot(aes(x = Time, y = Kappa, color = Name))+
  geom_point(size = 1) + geom_text(aes(label = Name), check_overlap = TRUE) + ggtitle("Full dataset training")

# pda, multinom and rf are the best, but rf is very long


lapply(models, function(model){
  plot_confusion(fits[[model]]$pred$obs, fits[[model]]$pred$pred, name = paste(model, "full"))
})

#plot(fits$rf$finalModel)

# try NN

# if(!require(neuralnet)) install.packages("neuralnet")
# library(neuralnet)
# 
# nn <- neuralnet(Activity ~ ., data = df, hidden = c(250, 100, 25), act.fct = "logistic", linear.output = FALSE)
# 
# nn_pred <- compute(nn, df_validation[1:ncol(df_validation)-1])$net.result
# 
# labels <- levels(factor(activity_labels$Activity))
# 
# pred_label <- data.frame(max.col(nn_pred)) %>%
#   mutate(prediction = labels[max.col.nn_pred.]) %>%
#   dplyr::select(2) %>%
#   unlist()
# 
# plot_confusion(pred_label, df_validation$Activity, name = "NN full")

# All have the same problems: Sitting/standing distinguishing and lie_to_sit/lie_to_stand prediction

# keep only standing and sitting and try to classify them

#df_sit_stand <- df %>% filter(Activity %in% c("STANDING", "SITTING")) %>%
#  mutate(standing = as.factor(ifelse(Activity == "STANDING", 1, 0))) %>% dplyr::select(-Activity)

# df_sit_stand <- df %>% filter(Activity %in% c("STANDING", "SITTING")) %>% mutate(Activity = droplevels(Activity))
# 
# 
# # plot outcomes distribution
# df_sit_stand %>% group_by(Activity) %>% mutate(n = n()) %>%
#   ggplot(aes(reorder(Activity, -n))) +
#   geom_bar(col=rgb(0.1,0.4,0.5,0.7), fill=rgb(0.1,0.4,0.5,0.7)) + 
#   xlab("Activity") + ylab("Count") +
#   scale_y_continuous(breaks = seq(0,1500,100)) +
#   ggtitle("Distribution of activities") +
#   theme_bw() +
#   theme(axis.text.x=element_text(angle = -60, hjust = 0))
# 
# # how many samples of each class in df
# sort(table(df_sit_stand$Activity),decreasing=TRUE)
# 
# # dataset is almost balanced, try feature reduction by correlation
# 
# # calculate correlation matrix
# correlationMatrix <- cor(df_sit_stand[-562])
# # summarize the correlation matrix
# #print(correlationMatrix)
# 
# if(!require(corrplot)) install.packages("corrplot")
# library(corrplot)
# corrplot(correlationMatrix, method = "color", tl.pos='n')
# # find attributes that are highly corrected (ideally >0.75)
# highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
# # print indexes of highly correlated attributes
# #print(highlyCorrelated)
# #print(highlyCorrelated2)
# 
# # remove highly correlated features
# df_reduced_75 <- df_sit_stand %>% dplyr::select(-all_of(highlyCorrelated))
# 
# corrplot(cor(df_reduced_75 %>% dplyr::select(-Activity)), method = "color", tl.pos='n')
# 
# 
# 
# # for binary classification we expect less features to be needed. Let's try PCA
# pca <- prcomp(df_sit_stand[-562], scale = TRUE)
# 
# # plot variance explained by each of principal components
# ggplot(aes(1:length(pca$sdev), (pca$sdev^2 / sum(pca$sdev^2))*100), data = NULL) + geom_col() +
#   scale_y_continuous(name = "% variance explained", limits = c(0,15)) + xlab("PCs") +
#   xlim(0, 30) + 
#   ggtitle("Variance explained by Principal Components")+
#   theme(plot.title = element_text(hjust = 0.5))
# 
# # plot cumulative variance explained by principal components
# ggplot(aes(1:length(pca$sdev), cumsum(pca$sdev^2 / sum(pca$sdev^2))*100), data = NULL) + 
#   geom_point(alpha = 0.5, size = 1) +
#   scale_y_continuous(name = "% variance explained", limits = c(0,100)) + xlab("PCs") +
#   xlim(0, length(pca$sdev)) + geom_line() +
#   ggtitle("Cumulative variance explained by Principal Components")+
#   theme(plot.title = element_text(hjust = 0.5))
# 
# plot(pca)
# 
# df_sit_stand_pca <- as.data.frame(pca$x[,1:200]) %>% cbind(df_sit_stand[562])
#   
# # visualize binary classification of two PC
# 
# df_sit_stand_pca %>% ggplot(aes(x = PC1, y = PC4, color = Activity)) +
#   geom_point()
# 
# 
#   
# # repeat the same with binary
# 
# 
# 
# models <- c("kknn", "pda", "slda", "hdda", "knn", "pam", "C5.0Tree", "rf", "pls", "earth", "xgbTree")
# 
# #models <- c("knn", "pam", "C5.0Tree", "CSimca", "rf", "pls")
# 
# control <- trainControl(method="cv", number=10, classProbs= TRUE, summaryFunction = twoClassSummary, savePredictions = "final",
#                         verbose = PRINT_DEBUG)
# 
# metric <- "ROC"
# 
# file_name <- "./models//binary_fits.rds"
# 
# 
# if (1) {
#   
#   time_start <- unclass(Sys.time())
#   
#   fits <- lapply(models, function(model){ 
#     print(model)
#     train(Activity ~ ., data = df_sit_stand, method = model, metric = metric, trControl = control)
#   }) 
#   
#   time_end <- unclass(Sys.time())
#   
#   bin_ensemble_total_time <- time_end - time_start
#   
#   names(fits) <- models
#   
#   # If "models" folder is not exist, create it
#   if (!dir.exists("./models")) {dir.create("./models")}
#   # save fits
#   saveRDS(fits, file_name)
# } else {
#   # if file is not found, stop and message. 
#   if (!file.exists(file_name)) {stop("File not found. Rerun code with RETRAIN = TRUE")} 
#   # read from file
#   else {fits <- readRDS(file_name)}
# }
# 
# 
# #ensemble_total_time/3600  # ~4 hours
# # print results
# 
# results <- data.frame(t(sapply(1:length(models), function(n){
#   pos_max <- which.max(fits[[n]]$results$ROC)
#   c(fits[[n]]$method, fits[[n]]$results$ROC[pos_max], fits[[n]]$times$everything["elapsed"])
# })))
# 
# colnames(results) <- c("Name", "ROC", "Time")
# 
# 
# results %>% mutate(Time = as.numeric(Time) / 60, ROC = as.numeric(ROC)) %>% ggplot(aes(x = Time, y = ROC, color = Name))+
#   geom_point(size = 1) + geom_text(aes(label = Name), check_overlap = TRUE) + ggtitle("Binary df train")
# 
# plot_confusion(fits$pda$pred$obs, fits$pda$pred$pred, name = "PDA")
# plot_confusion(fits$rf$pred$obs, fits$rf$pred$pred, name = "RF")
# plot_confusion(fits$kknn$pred$obs, fits$kknn$pred$pred, name = "KKNN")
# plot_confusion(fits$xgbTree$pred$obs, fits$xgbTree$pred$pred, name = "xgbTree")


# repeat with reduced dataset
# 
# file_name <- "./models//binary_fits_corr_reduced.rds"
# 
# 
# if (1) {
#   
#   time_start <- unclass(Sys.time())
#   
#   fits <- lapply(models, function(model){ 
#     print(model)
#     train(Activity ~ ., data = df_reduced_75, method = model, metric = metric, trControl = control)
#   }) 
#   
#   time_end <- unclass(Sys.time())
#   
#   bin_ensemble_total_time <- time_end - time_start
#   
#   names(fits) <- models
#   
#   # If "models" folder is not exist, create it
#   if (!dir.exists("./models")) {dir.create("./models")}
#   # save fits
#   saveRDS(fits, file_name)
# } else {
#   # if file is not found, stop and message. 
#   if (!file.exists(file_name)) {stop("File not found. Rerun code with RETRAIN = TRUE")} 
#   # read from file
#   else {fits <- readRDS(file_name)}
# }
# 
# 
# #ensemble_total_time/3600  # ~4 hours
# # print results
# 
# results <- data.frame(t(sapply(1:length(models), function(n){
#   pos_max <- which.max(fits[[n]]$results$ROC)
#   c(fits[[n]]$method, fits[[n]]$results$ROC[pos_max], fits[[n]]$times$everything["elapsed"])
# })))
# 
# colnames(results) <- c("Name", "ROC", "Time")
# 
# 
# results %>% mutate(Time = as.numeric(Time) / 60, ROC = as.numeric(ROC)) %>% ggplot(aes(x = Time, y = ROC, color = Name))+
#   geom_point(size = 1) + geom_text(aes(label = Name), check_overlap = TRUE) + ggtitle("Binary df train")
# 
# plot_confusion(fits$pda$pred$obs, fits$pda$pred$pred, name = "PDA cor reduced")
# plot_confusion(fits$rf$pred$obs, fits$rf$pred$pred, name = "RF cor reduced")
# plot_confusion(fits$kknn$pred$obs, fits$kknn$pred$pred, name = "KKNN cor reduced")
# plot_confusion(fits$xgbTree$pred$obs, fits$xgbTree$pred$pred, name = "xgbTree cor reduced")
# 
# 
# # and with pca dataset
# file_name <- "./models//binary_fits_pca.rds"
# 
# 
# if (1) {
#   
#   time_start <- unclass(Sys.time())
#   
#   fits <- lapply(models, function(model){ 
#     print(model)
#     train(Activity ~ ., data = df_sit_stand_pca, method = model, metric = metric, trControl = control)
#   }) 
#   
#   time_end <- unclass(Sys.time())
#   
#   bin_ensemble_total_time <- time_end - time_start
#   
#   names(fits) <- models
#   
#   # If "models" folder is not exist, create it
#   if (!dir.exists("./models")) {dir.create("./models")}
#   # save fits
#   saveRDS(fits, file_name)
# } else {
#   # if file is not found, stop and message. 
#   if (!file.exists(file_name)) {stop("File not found. Rerun code with RETRAIN = TRUE")} 
#   # read from file
#   else {fits <- readRDS(file_name)}
# }
# 
# 
# #ensemble_total_time/3600  # ~4 hours
# # print results
# 
# results <- data.frame(t(sapply(1:length(models), function(n){
#   pos_max <- which.max(fits[[n]]$results$ROC)
#   c(fits[[n]]$method, fits[[n]]$results$ROC[pos_max], fits[[n]]$times$everything["elapsed"])
# })))
# 
# colnames(results) <- c("Name", "ROC", "Time")
# 
# 
# results %>% mutate(Time = as.numeric(Time) / 60, ROC = as.numeric(ROC)) %>% ggplot(aes(x = Time, y = ROC, color = Name))+
#   geom_point(size = 1) + geom_text(aes(label = Name), check_overlap = TRUE) + ggtitle("Binary df train")
# 
# plot_confusion(fits$pda$pred$obs, fits$pda$pred$pred, name = "PDA pca")
# plot_confusion(fits$rf$pred$obs, fits$rf$pred$pred, name = "RF pca")
# plot_confusion(fits$kknn$pred$obs, fits$kknn$pred$pred, name = "KKNN pca")
# plot_confusion(fits$xgbTree$pred$obs, fits$xgbTree$pred$pred, name = "xgbTree pca")
# 
# 
# 
# 

