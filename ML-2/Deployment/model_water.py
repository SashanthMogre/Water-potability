# -*- coding: utf-8 -*-

# Importing necessary libraries
import pandas as pd
from xgboost import XGBClassifier
import pickle
import os
os.chdir('C:\\Users\\jana\\Documents\\Machine Learning 2\\Project\\Deployment')
os.listdir()
os.getcwd()


# Importing the dataset
df = pd.read_csv('water_potability.csv')

df['Sulfate'] = df['Sulfate'].fillna(df.groupby('Potability')['Sulfate'].transform('median'))
df['ph'] = df['ph'].fillna(df.groupby('Potability')['ph'].transform('median'))
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df.groupby('Potability')['Trihalomethanes'].transform('median'))

# Dictionary containing the mapping
variety_mappings = {0: 'Not-Drinkable', 1: 'Drinkable'}

# Encoding the target variables to integers
# df = df.replace([0,1],['Not-Drinkable','Drinkable'])

X = df.iloc[:, 0:-1] # Extracting the independent variables
y = df.iloc[:, -1] # Extracting the target/dependent variable

# Initializing the xgb model
xgb_model = XGBClassifier(learning_rate=0.2, max_depth=5, n_estimators=30)

xgb_model.fit(X, y)

# save the model to disk
filename = 'logreg1.pkl'
pickle.dump(xgb_model, open(filename, 'wb')) 

# try using the pickled model
# load the model to disk
filename = 'logreg1.pkl'
xgb_yourcopy = pickle.load(open(filename, 'rb'))
import numpy as np
features = np.array([10,9,8,7,6,5,4,3,2]).reshape(1,-1)
xgb_yourcopy.predict(features)

