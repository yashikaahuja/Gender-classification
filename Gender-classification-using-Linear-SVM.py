# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 10:11:37 2018

@author: LENOVO
"""
#from sklearn.feature_extraction.text import countVectorizer
from sklearn import neighbors,datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm,metrics
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import accuracy_score as ac
import numpy as np
from sklearn.model_selection import train_test_split

x=np.load("D:/P3/New folder/mini-project/x.npy")
y=np.load("D:/P3/New folder/mini-project/y.npy")

#Cross validation
xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size=0.10, random_state = 0)
'''print(xtrain)
print(xtest)
print(ytrain)
print(ytest)'''
'''__init__(n_components=None, copy=True, iterated_power=3, whiten=False, random_state=None)'''
'''pca = RandomizedPCA(n_components=90)
pca.fit(xtrain)   
train_res = pca.fit_transform(xtrain)
#xtest = (np.float32(xtest[:])/255.)
test_res = pca.transform(xtest)

classifier = svm.SVC(gamma = 0.01,C=3, kernel='rbf')
classifier.fit(train_res,ytrain)

expected=ytest
predicted=classifier.predict(test_res)

ac = accuracy_score(expected,predicted)
print ac'''
#SVM
model=svm.SVC(kernel='poly')
model.fit(xtrain, ytrain)
pred=model.predict(xtest);
print (" accuracy is",ac(pred,ytest)*100)


