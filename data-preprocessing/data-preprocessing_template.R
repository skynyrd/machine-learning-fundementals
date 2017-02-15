# Import data
dataset = read.csv('Data.csv')
dataset = dataset[, 2:3]

# Splitting the data into training set and test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling (All features should be in same scale.)
# factoring in R is not numeric, so column 1 and 4 is not applicable to be scaled.
# as that columns are not important for now, we ignore them.
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])