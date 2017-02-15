# Data Preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values    # All rows(observations) & all columns(variables) except the last one.
y = dataset.iloc[:, 3].values  # All rows(observations) & only 4th column (variable).


# Splitting the data into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling (All features should be in same scale.)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # once fit, then transform.
X_test = sc_X.transform(X_test) # transform according to the fit of X_train"""
