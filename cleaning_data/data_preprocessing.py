import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split 

#import data
dataset = pd.read_csv('./data/Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#take care of missing data
imputer = Imputer(missing_values = 'NaN', strategy="mean", axis=0)
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#encoding categorical data
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# split into testing and training sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

#feature scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sx_x.transform(x_test)





