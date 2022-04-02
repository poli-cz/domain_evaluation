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




#clf = svm.SVC(kernel='rbf', verbose=True) # Linear Kernel

d = Database.Database('domains')
bad_data = list()
good_data = list()


good_collection = d.return_collection("bad_dataset")
bad_collection = d.return_collection("good_dataset")


for name in good_collection.find():
    good_data.append(name)


for name in bad_collection.find():
    bad_data.append(name)


#print(len(bad_data))
#print(len(good_data))



 

label = None
dats = shuffle(good_data + bad_data)

x_testset = list()
y_testset = list()

x_dataset = list()
y_dataset = list()
counter =0

for piece in dats:
    if counter < 5000:
        x_testset.append(piece['data'])
        y_testset.append(piece['label'])
    else:
        x_dataset.append(piece['data'])
        y_dataset.append(piece['label'])

    counter+=1



print("loading svm....")

clf = pickle.load(open('./svm_model.smv', 'rb'))

#clf.fit(x_dataset, y_dataset)
#print("predicting smv...")
#y_pred = clf.predict(x_dataset)
#print("Accuracy:",metrics.accuracy_score(y_dataset, y_pred))

#filename = ''
#pickle.dump(clf, open(filename, 'wb'))

for line in dats:
    print("Input data:", line['data'], "for doamain:", line['domain'])
    input()
    in_data = np.array([line['data']], dtype=np.float32)
    prediction = clf.predict(in_data)
    print("Predicted result:", prediction)
    input()