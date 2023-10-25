# -*- coding: utf-8 -*-

#original file is located at
  #  https://colab.research.google.com/drive/1gwAuv6LHLed0lreVySv5cH187n9p9nkU?usp=sharing


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load data
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('data.csv')
df.head(7)

#Count the number of rows and columns in the data set
df.shape

#Count the number of empty (NaN, NAN, na) values in each column
df.isna().sum()

#Drop the column with all missing values
df = df.dropna(axis=1)

#Get the new count of the number of rows and columns
df.shape

#Get a count of the number of Malignant(M) or Benign (B) cells
df['diagnosis'].value_counts()

#Visualize the count
sns.countplot(df['diagnosis'], label = 'count')

#Look at the data types to see which columns need to be encoded
df.dtypes

#Encode the categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1] = labelencoder_Y.fit_transform(df.iloc[:,1].values)

#Create a pair plot
sns.pairplot(df.iloc[:,1:5], hue='diagnosis')

#Print the first 5 rows of the new data
df.head(5)

#Get the correlation of the columns
df.iloc[:,1:12].corr()

#Visualize the correlation
plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:12].corr(), annot = True, fmt='.0%')

#Split the data set into independent (X) and dependent (Y) data sets
X = df.iloc[:,2:31].values
Y = df.iloc[:,1].values

#Split data set into 75% training and 25% testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Scale the date (Feature Scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Create a function for the models
def models(X_train, Y_train):

  #Logistic Regression
  from sklearn.linear_model import LogisticRegression
  log = LogisticRegression(random_state=0)
  log.fit(X_train, Y_train)

  #Decision Tree
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
  tree.fit(X_train, Y_train)

  #Random Forest Classifier
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=0)
  forest.fit(X_train, Y_train)

  #Print the models accuracy on the training data
  print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
  print('[1]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
  print('[2]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))

  return log, tree, forest

#Getting all the models
model = models(X_train, Y_train)

#test model accuracy on test data on confusion matrix
from sklearn.metrics import confusion_matrix

for i in range( len(model) ):
  print('Model ', i)
  cm = confusion_matrix(Y_test, model[i].predict(X_test))

  TP = cm[0][0]
  TN = cm[1][1]
  FN = cm[1][0]
  FP = cm[0][1]

  print(cm)
  print('Testing Accuracy = ', (TP + TN)/ (TP + TN + FN + FP))
  print()
