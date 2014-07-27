Project
========================================================

Loading the data.

```r
library(caret)
library(ggplot2)
library(AppliedPredictiveModeling)
set.seed(5)
dataTesting<-read.csv('pml-testing.csv')
dataTraining<-read.csv('pml-training.csv')
```
Since testing data is small, I split the training data 3:1 for training and testing respectively. Then I will split the training data 70:30 to create the real training set and a set for cross-validation.


```r
inTrain=createDataPartition(dataTraining$classe, p=3/4)[[1]]
training=dataTraining[inTrain,]
testing=dataTraining[-inTrain,]
crossTrain=createDataPartition(training$classe, p=7/10)[[1]]
smallTrain=training[crossTrain,]
smallCross=training[-crossTrain,]
```
From the summary of the data and from the project information it is clear that there are 5 classes of exercises (A-E) and 6 participants. belt, forearm, arm, and dumbell 


## Feature selection
Find columns with "accel" in their names.

```r
accList<-colnames(training)[grep("\\<accel|classe", colnames(training),perl=TRUE)]
trainSelect<-training[accList[grep("^(accel|classe)",accList,perl=TRUE)]]
testingSelect<-testing[accList[grep("^(accel|classe)",accList,perl=TRUE)]]
smallTrainSelect<-smallTrain[accList[grep("^(accel|classe)",accList,perl=TRUE)]]
smallCrossSelect<-smallCross[accList[grep("^(accel|classe)",accList,perl=TRUE)]]
```

Some plotting, lets see which feature correlates with the classes.

```r
belt<-colnames(trainSelect)[grep("belt",names(trainSelect))]
arm<-colnames(trainSelect)[grep("_arm",names(trainSelect))]
forearm<-colnames(trainSelect)[grep("forearm",names(trainSelect))]
dumbbell<-colnames(trainSelect)[grep("dumbbell",names(trainSelect))]
```

```r
featurePlot(x = trainSelect[, belt],
             y = trainSelect$classe,
             plot = "box",
             ## Pass in options to bwplot() 
             scales = list(y = list(relation="free"),
                           x = list(rot = 90)),
             layout = c(4,1 ),
             auto.key = list(columns = 2))
```

![plot of chunk plotting](figure/plotting1.png) 

```r
featurePlot(x = trainSelect[, arm],
             y = trainSelect$classe,
             plot = "box",
             ## Pass in options to bwplot() 
             scales = list(y = list(relation="free"),
                           x = list(rot = 90)),
             layout = c(4,1 ),
             auto.key = list(columns = 2))
```

![plot of chunk plotting](figure/plotting2.png) 

```r
featurePlot(x = trainSelect[, forearm],
             y = trainSelect$classe,
             plot = "box",
             ## Pass in options to bwplot() 
             scales = list(y = list(relation="free"),
                           x = list(rot = 90)),
             layout = c(4,1 ),
             auto.key = list(columns = 2))
```

![plot of chunk plotting](figure/plotting3.png) 

```r
featurePlot(x = trainSelect[, dumbbell],
             y = trainSelect$classe,
             plot = "box",
             ## Pass in options to bwplot() 
             scales = list(y = list(relation="free"),
                           x = list(rot = 90)),
             layout = c(4,1 ),
             auto.key = list(columns = 2))
```

![plot of chunk plotting](figure/plotting4.png) 

## Training
