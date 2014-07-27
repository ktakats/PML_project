setwd("/home/kati/munka/courses//mlearning_r")
library(caret)
library(gbm)
library(AppliedPredictiveModeling)
set.seed(5)
dataTesting<-read.csv('pml-testing.csv')
dataTraining<-read.csv('pml-training.csv')

inTrain=createDataPartition(dataTraining$classe, p=3/4)[[1]]
training=dataTraining[inTrain,]
testing=dataTraining[-inTrain,]


dim(training)
na_test = sapply(training, function(x) {sum(is.na(x))})
table(na_test)
bad_columns = names(na_test[na_test==max(na_test)])
training = training[, !names(training) %in% bad_columns]
testing = testing[, !names(testing) %in% bad_columns]
num_test = sapply(training, function(x) {sum("#DIV/0!" %in% levels(x))})
table(num_test)
bad_columns = names(num_test[num_test==max(num_test)])
training = training[, !names(training) %in% bad_columns]
testing = testing[, !names(testing) %in% bad_columns]
dim(training)
training<-subset(training[,8:60])
testing<-subset(testing[,8:60])

preProc<-preProcess(training[,1:52],method="pca",thresh=0.90)
trainPC<-cbind(classe=training[,53],predict(preProc,training[,1:52]))
testPC<-cbind(classe=testing[,53],predict(preProc,testing[,1:52]))

fitControl <- trainControl(method = "repeatedcv",number = 5,repeats = 2)

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9),
                        n.trees = (1:30)*50,
                        shrinkage = 0.1)
gbmFit <- train(classe ~ ., data = trainPC,
                 method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE,
                 tuneGrid = gbmGrid)
