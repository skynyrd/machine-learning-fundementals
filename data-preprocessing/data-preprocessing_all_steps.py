# Data Preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values    # All rows(observations) & all columns(variables) except the last one.
y = dataset.iloc[:, 3].values  # All rows(observations) & only 4th column (variable).

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values= 'NaN', strategy='mean', axis= 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Equalize each category (country)
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

# Encode categorical value "Purchased"
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the data into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling (All features should be in same scale.)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # once fit, then transform.
X_test = sc_X.transform(X_test) # transform according to the fit of X_train
