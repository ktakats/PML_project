#Practical machine learning course project


## Cleaning the data

Loading necessary libraries and the data.

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
## Find out what's changed in ggplot2 with
## news(Version == "1.0.0", package = "ggplot2")
```

```r
library(AppliedPredictiveModeling)
set.seed(5)
dataTesting<-read.csv('pml-testing.csv')
dataTraining<-read.csv('pml-training.csv')
```


I split the training data into two parts for training and testing with the ratio of 3:1.

```r
inTrain=createDataPartition(dataTraining$classe, p=3/4)[[1]]
training=dataTraining[inTrain,]
testing=dataTraining[-inTrain,]
```


Let's see the dimensions of the training set:

```r
dim(training)
```

```
## [1] 14718   160
```

Then I check, how many NAs there are in the data:

```r
na_test = sapply(dataTesting, function(x) {sum(is.na(x))})
table(na_test)
```

```
## na_test
##   0  20 
##  60 100
```

Looks like there are many columns that contains mostly NAs especially in the final testing set. I'm getting rid of these columns from both the training and the test set.


```r
bad_columns = names(na_test[na_test==max(na_test)])
training = training[, !names(training) %in% bad_columns]
testing = testing[, !names(testing) %in% bad_columns]
dataTesting = dataTesting[, !names(dataTesting) %in% bad_columns]
```


Let's check, what columns are left in the training set:

```r
colnames(training)
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
##  [7] "num_window"           "roll_belt"            "pitch_belt"          
## [10] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
## [13] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [16] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [19] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [22] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [25] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [28] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [31] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [34] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [37] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [40] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [43] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [46] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [49] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [52] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [55] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [58] "magnet_forearm_y"     "magnet_forearm_z"     "classe"
```

For the training I won't use the columns X, user names, and the timestamps, so I remove these columns from both the training and the test set:

```r
training<-subset(training[,8:60])
testing<-subset(testing[,8:60])
dataTesting<-subset(dataTesting[,8:60])
dim(training)
```

```
## [1] 14718    53
```
So in the end there are 52 variables left plus the classifications.

## Preprocessing and modell building

I run PCA to with the treshold of 0.90 to produce the variables to the fitting??? and apply it to the training and test sets.

```r
preProc<-preProcess(training[,1:52],method="pca",thresh=0.90)
trainPC<-cbind(classe=training[,53],predict(preProc,training[,1:52]))
testPC<-cbind(classe=testing[,53],predict(preProc,testing[,1:52]))
dataTestingPC<-cbind(classe=dataTesting[,53],predict(preProc,dataTesting[,1:52]))
```

Building the model using random forest method:


```r
fitControl <- trainControl(method = "repeatedcv",number = 10,repeats = 10)
fitRF<-train(trainPC[,2:ncol(trainPC)],trainPC[,1],method="rf",trControl=fitControl)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
fitRF
```

```
## Random Forest 
## 
## 14718 samples
##    18 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 10 times) 
## 
## Summary of sample sizes: 13248, 13247, 13247, 13245, 13246, 13247, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.005        0.006   
##   10    1         1      0.005        0.007   
##   20    1         0.9    0.006        0.007   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

Applying the model and compare the results using the test set:

```r
confusionMatrix(testPC$classe,predict(fitRF,testPC))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1380    5    7    3    0
##          B    8  924   15    1    1
##          C    2   15  834    2    2
##          D    0    3   29  772    0
##          E    0    1   10    8  882
## 
## Overall Statistics
##                                         
##                Accuracy : 0.977         
##                  95% CI : (0.973, 0.981)
##     No Information Rate : 0.283         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.971         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.993    0.975    0.932    0.982    0.997
## Specificity             0.996    0.994    0.995    0.992    0.995
## Pos Pred Value          0.989    0.974    0.975    0.960    0.979
## Neg Pred Value          0.997    0.994    0.985    0.997    0.999
## Prevalence              0.283    0.193    0.183    0.160    0.180
## Detection Rate          0.281    0.188    0.170    0.157    0.180
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.994    0.984    0.963    0.987    0.996
```

## Out of sample error


```r
accuracy <- postResample(testPC$classe, predict(fitRF,testPC))[[1]]
error<-1-accuracy
```
The out of sample error is 2.2838.

