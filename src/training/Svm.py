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
model: SVM model to be validated
x_testset: Values for testing
y_testset: Labels for testing
graph: bool value, if should be printed ROC curve
		  
''' 
def validate_model(model, x_testset, y_testset, graph=False):

    # Compute F1 metric for system evaluation
    metric = metrics.F1Score(num_classes=1, threshold=0.5)

    #
    y_true = list()
    y_pred = list()

    for i in range(len(x_testset)):
        label = y_testset[i]
        data = x_testset[i]

        prediction = float(model.predict(data))

        y_true.append([label])
        y_pred.append([prediction])
        counter+=1

    y_true_converted = np.array(y_true, np.float32)
    y_pred_converted = np.array(y_pred, np.float32)


    metric.update_state(y_true_converted, y_pred_converted)
    result = metric.result()


    print("F1 score for model is:", result)


    # Build ROC curve chart
    if graph is not False:
        from sklearn import metrics as sk_metrics

        fpr, tpr, _ = sk_metrics.roc_curve(y_true_converted,  y_pred_converted)

        plt.rcParams['font.size'] = 10
        #create ROC curve
        plt.plot(fpr,tpr, label = "ROC křivka SVM modelu")
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('ROC křivka SVM modelu')
        plt.show()





if __name__ == '__main__':

    ### Learning constants ###
    svm_kernel = 'rbf'
    test_learn_ratio = 0.2


    # SVM definition #
    clf = svm.SVC(kernel=svm_kernel, verbose=True) # Linear Kernel

    # Prepare learning dataset #
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
    counter = 0

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








