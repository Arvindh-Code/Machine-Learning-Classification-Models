#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd

from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
import warnings
# as i am getting warning so filtering it for better visual
warnings.filterwarnings('ignore')

#reading nba2021 file using pandas
sample_data = pd.read_csv("nba2021.csv")
#print(len(sample_data))

#https://sparkbyexamples.com/pandas/pandas-dataframe-query-examples/
#As MP and GS are low than 2 then player might not played the game, so querying as below
sample_data = sample_data.query("`MP`>=2 and `GS`>=2")
       
#sample_nba_data = sample_data.filter(['FT', 'FG%', '3P', '3PA', '3P%', '2P%','2P', 'eFG%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK'])
#sample_nba_data = sample_data.filter(["G","MP","FG","FGA","FG%","3P","3PA","3P%","2P","2PA","2P%","FT","FTA","FT%","TRB","AST","STL","BLK","TOV","PF","PTS"])
#sample_nba_data = sample_data.filter(["FG%", "3P%", "2P%", "eFG%", "FT%","TRB", "AST", "STL", "BLK"])
#sample_nba_data = sample_data.filter(["FG","FGA","FG%","3P%","2P%","eFG%","FT","FTA","FT%","ORB","DRB","TRB","AST","STL","BLK","TOV","PF"])
#sample_nba_data = sample_data.filter(['FT', 'FG%', '3P', '3PA', '3P%', '2P%','2P', 'eFG%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK'])

#Tried mutliple steps, end up with below filter.
#as 3pa+3p = 3p% so eliminating 3p, 3pa. Same for 2 points too and even for PTS
sample_nba_data = sample_data.filter(["FT", "FG%", "3P%", "2P%", "eFG%", "ORB", "DRB", "TRB", "AST", "STL", "BLK"])


#for training_size,test_size initialization
#as per question test - 0.25 and train - 0.75 has been assigned
train_feature, test_feature, train_class, test_class = train_test_split(
    sample_nba_data, sample_data.Pos, stratify=sample_data.Pos,train_size=0.75, test_size=0.25) #question 1

'''
#Knn 
print("using model KNN:")
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(train_feature, train_class)
print("Test set score: {:.3f}".format(knn.score(test_feature, test_class)))
scores = cross_val_score(knn, sample_nba_data, sample_data.Pos, cv=10)
#print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print()
'''

#https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
#as we have sample size > sample_nba_data
#Linear SVC
print("Using Linear SVC\n")
linearsvm = LinearSVC(dual=False).fit(train_feature, train_class)
print("Question 1 and Question 2:")
#printing accuracy of linearSVM
print("Test set accuracy: {:.3f}".format(linearsvm.score(test_feature, test_class))) #question 2
print("\nQuestion 3:")
prediction = linearsvm.predict(test_feature)
print("confusion matrix :")
print(pd.crosstab(test_class, prediction, margins=True))    #printing confusion matrix for linearSVM   
print()
scores = cross_val_score(linearsvm, sample_nba_data, sample_data.Pos, cv=10)
print("\nQuestion 4 and Question 5:")
print("Cross-validation scores: {}".format(scores))  #printing the 10 folded cross validation value
print("Average cross-validation score: {:.2f}".format(scores.mean()))  #printing the average of each fold

'''
#Naive Bayes - GaussianNB
print("Using Naive Bayes - GaussianNB")
naive_bayes=GaussianNB().fit(train_feature,train_class)
print("Test set score: {:.3f}".format(naive_bayes.score(test_feature, test_class)))
scores = cross_val_score(naive_bayes, sample_nba_data, sample_data.Pos, cv=10)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
prediction = naive_bayes.predict(test_feature)
print(pd.crosstab(test_class, prediction, margins=True))       
print()
'''
'''
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
#Naive Bayes - MultinomialNB
print("Using Naive Bayes - MultinomialNB")
classifier = MultinomialNB()
p=classifier.fit(train_feature, train_class)
#print(classifier.predict(train_feature))
pred = classifier.predict(train_feature)
scores = cross_val_score(p, sample_nba_data, sample_data.Pos, cv=10)
print("Cross-validation scores: {}".format(scores))
print()
print("Average cross-validation score: {:.2f}".format(scores.mean()))
prediction = p.predict(test_feature)
print(pd.crosstab(test_class, prediction, margins=True))       
print()
'''
'''
#Decision Tree Classifier         
print("Using Decision Tree Classifier")
tree = DecisionTreeClassifier(max_depth=30)
tree.fit(train_feature, train_class)
print("Test set score: {:.3f}".format(tree.score(test_feature, test_class)))
scores = cross_val_score(tree, sample_nba_data, sample_data.Pos, cv=10)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
prediction = tree.predict(test_feature)
print(pd.crosstab(test_class, prediction, margins=True))       
print()
'''



# In[ ]:




