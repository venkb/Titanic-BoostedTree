#Predicting Titanic survival rate using boosted tree
library(gbm)
library(mice)
library(caTools)

#Load the training data
#setwd('Please enter the folder which contains train.csv')
titanic = read.csv('train.csv')
names(titanic)

#Select the variables of interest
titanic = titanic[,c(2,3,5,6,7,8,10)]
names(titanic)

#Determing the modeling type of selected variables
str(titanic)

#Look for missing values and impute missing values
md.pattern(titanic)

#Impute missing Age feature using mice package
titanic_impute = mice(data = titanic,
                      m = 1, 
                      maxit = 3)
titanic_imputed = complete(titanic_impute,1)

#Scale the continuous features
titanic_imputed[,c(4,5,6,7)] = scale(titanic_imputed[,c(4,5,6,7)])

#Split the data into training and test set
set.seed(5)
split = sample.split(Y = titanic_imputed$Survived,
                     SplitRatio = 0.8)
training_set = subset(titanic_imputed,
                      split == TRUE)
test_set = subset(titanic_imputed,
                  split == FALSE)

#Build the boosted tree classifier using the training set
titanic_boostedtree = gbm(formula = Survived ~ .,
                     data = training_set,
                     distribution = "bernoulli",
                     n.trees = 100)

        
#Predict the survival in test set using the boosted tree classifier
y_prob = predict(titanic_boostedtree,
                 newdata = test_set[,-1],
                 n.trees = 100,
                 type = 'response')

y_pred = ifelse(y_prob >= 0.37, 1, 0)

#Build the confusion matrix
cm = table(test_set[,1], y_pred)
cm

#Calculate the misclassification rate
misclassification_rate = (cm[2,1] + cm[1,2])/ (cm[1,1] + cm[1,2] + cm[2,1] + cm[2,2])
misclassification_rate

