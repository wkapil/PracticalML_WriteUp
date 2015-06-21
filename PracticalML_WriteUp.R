library(caret)
set.seed(12345)

rawData <- read.csv("pml-training.csv")

## Removing unwanted data
# 1. Remove unwanted columns (e.g. columns containing names, or irrelevant informaton)
trainingData <- subset(rawData, select=-c(1:7))

# 2. Removing nearZeroValues 
nzv <- nearZeroVar(trainingData)
trainingData <- trainingData[,-nzv]

# 3. Removing NA's.
completeColumns <- (colSums(is.na(trainingData)) == 0)
trainingData <- subset(trainingData, select=completeColumns)
classeColumn <- dim(trainingData)[2] # Classe column is the last column
trainingData[, -classeColumn] <- apply(trainingData[, -classeColumn], 2, function(x) as.numeric(as.character(x)))

inTrain <- createDataPartition(y = trainingData$classe, p=0.7, list=FALSE)
training <- trainingData[inTrain,]
testing <- trainingData[-inTrain,]

trainingControl <- trainControl(method = "cv", number = 4, allowParallel = TRUE)
modelFit <- train(training$classe ~. , data = training, method = "rf", trControl = trainingControl, preProcess = c("center","scale"))

modelFit$finalModel

predictions <- predict(modelFit, testing)
confusionMatrix(predictions, testing$classe)

sum(predictions == testing$classe) / nrow(testing)

predictionData <- read.csv("pml-testing.csv")
predictResults <- predict(modelFit, newdata=predictionData)
predictResults

# Write files for submission
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predictResults)

