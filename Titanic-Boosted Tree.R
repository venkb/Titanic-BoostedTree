#Predicting Titanic survival rate using Boosted Trees
library(gbm)
library(caret)
library(mice)

##Set the working directory to the folder where train and test csv are available
dataset = read.csv('train.csv')

#Select features of interest
features = c(2,3,5,6,7,8,10.12)
dataset = dataset[,features]

#Move Survived to the end of the data frame
dataset = dataset[,c(2,3,4,5,6,7,1)]

#Data analysis
str(dataset)

#Check for missing values and impute missing values
md.pattern(dataset)
impute_dataset = mice(dataset,
                      m=1,
                      maxit = 3)
imputed_dataset = complete(impute_dataset,1)

#Build Boosted Tree classifier
#Perform grid search to arrive at optimal hyper parameter values
parameters = train(form = Survived ~ .,
                   data = imputed_dataset,
                   method = 'gbm')

boostedtree_classifier = gbm(formula = Survived ~ .,
                             data = imputed_dataset,
                             distribution = "bernoulli",
                             n.trees = 100,
                             interaction.depth = 3,
                             shrinkage = 0.1,
                             n.minobsinnode = 10)

#Model validation through k-fold cross validation
set.seed(5)
folds = createFolds(y = imputed_dataset, 
                    folds = 10)
boostedtree_classifier_misclassification_rate = lapply(folds, function(x){
        training_fold = imputed_dataset[-x,]
        test_fold = imputed_dataset[x,]
        folds_boostedtree_classifier = gbm(formula = Survived ~ .,
                                           data = training_fold,
                                           distribution = "bernoulli",
                                           n.trees = 100,
                                           interaction.depth = 3,
                                           shrinkage = 0.1,
                                           n.minobsinnode = 10)
        folds_y_prob = predict(folds_boostedtree_classifier,
                               newdata = test_fold,
                               n.trees = 100,
                               type = 'response')
        folds_y_pred = ifelse(folds_y_prob>0.40, 1, 0)
        folds_cm = table(test_fold[,7], folds_y_pred)
        folds_misclassification_rate = (folds_cm[2,1] + folds_cm[1,2])/(folds_cm[1,1]+folds_cm[1,2]+folds_cm[2,1]+folds_cm[2,2])
        return(folds_misclassification_rate)
})

mean_misclassification_rate = mean(as.numeric(boostedtree_classifier_misclassification_rate))
mean_misclassification_rate

#Predict the survival rate in unseen test data
unseen_test_set_full = read.csv('test.csv')

#Run the unseen test set through all the preparatory steps that training set went through
unseen_test_set_features = c(2,4,5,6,7,9)
unseen_test_set = unseen_test_set_full[,unseen_test_set_features]

#Check for missing values and impute missing values
md.pattern(unseen_test_set)
impute_unseen_test_set = mice(unseen_test_set,
                      m=1,
                      maxit = 3)
imputed_unseen_test_set = complete(impute_unseen_test_set,1)

#Make predictions using boosted tree classifier
unseen_test_set_y_prob = predict(boostedtree_classifier,
                       newdata = imputed_unseen_test_set,
                       n.trees = 100,
                       type = 'response')
unseen_test_set_y_pred = ifelse(unseen_test_set_y_prob>0.40, 1, 0)
boostedtree_output = data.frame(unseen_test_set_full$PassengerId,unseen_test_set_y_pred)
colnames(boostedtree_output) = c('PassengerId', "Survived")
write.csv(x = boostedtree_output,
          file = 'boostedtree_output.csv',
          row.names = FALSE)