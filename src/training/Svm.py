""" File: Svm.py
    Author: Jan Polisensky
    ----
    Modules for SVM classifier training
"""


import json
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import Database
from sklearn import metrics
import pickle
import array






'''		
Parameters
----------
model: SVM model to be trained
x_dataset: Values for training
y_dataset: Labels for training 
		  
''' 
def train(model, x_dataset, y_dataset):

    # Train SVM module
    model.fit(x_dataset, y_dataset)

    print("predicting smv...")
    y_pred = clf.predict(x_dataset)

    # testing model
    print("Accuracy:",metrics.accuracy_score(y_dataset, y_pred))








'''		
Parameters
----------
dataset : list
		  
''' 
def validate_model(model, testset, graph=False):
    pass










if __name__ == '__main__':

    svm_kernel = 'rbf'
    test_learn_ratio = 0.2


    clf = svm.SVC(kernel=svm_kernel, verbose=True) # Linear Kernel

    d = Database.Database('domains')
    bad_data = list()
    good_data = list()


    good_collection = d.return_collection("bad_dataset")
    bad_collection = d.return_collection("good_dataset")


    for name in good_collection.find():
        good_data.append(name)


    for name in bad_collection.find():
        bad_data.append(name)




    label = None

    # shuffle datasets
    shufled_data = shuffle(good_data + bad_data)

    x_testset = list()
    y_testset = list()

    x_dataset = list()
    y_dataset = list()
    counter =0

    dataset_len = len(shufled_data)



    # Split data to train data and test data
    for data_row in shufled_data:
        if counter < dataset_len*test_learn_ratio:
            x_testset.append(data_row['data'])
            y_testset.append(data_row['label'])
        else:
            x_dataset.append(data_row['data'])
            y_dataset.append(data_row['label'])

        counter+=1

    # Train model 
    train(clf, x_dataset, y_dataset)

    # Validate model
    validate_model(clf, x_testset, y_testset, True)








