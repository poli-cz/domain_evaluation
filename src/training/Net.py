""" File: Net.py
    Author: Jan Polisensky
    ----
    Modules for neural network training
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import json
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_addons import metrics
import matplotlib.pyplot as plt

# Import custom modules 
import Database




'''		
Class: Net
pytorch definition of neural network structure

'''       
class Net(nn.Module):

    # Network structure definition
    def __init__(self):         
        super().__init__()
        self.fc1 = nn.Linear(13, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 1)


    # Data flow definition
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)) # For binary classification is sigmoid best



'''		
Parameters
----------
dataset : shufled dataset from database
		  
'''          
def train(dataset):
    counter=0
    sum=0
    batch_sum=0

    checkpoint_position = 500

    for line in dataset:
        label = float(line["label"])

        dat = torch.FloatTensor(line["data"])
        output=net(dat)

        if counter % checkpoint_position == 0:

            print("Loss:", round(sum/checkpoint_position, 3),"Progress:",  round((counter/len(dataset))*100, 3), "%")
            sum=0

        counter+=1

        loss = loss_fn(output, torch.FloatTensor([label]))
        sum+=float(loss)
        batch_sum+=float(loss)

        # backwards
        loss.backward()
        optimizer.step()
            

    # Print batch results
    print("--------------------------------------------------")
    print("Batch loss:", float(batch_sum)/float(len(dataset)))
    print("--------------------------------------------------")





'''		
Parameters
----------
model: Neural network model to be validated
testset: Data for model validation
graph: bool value, if should be printed ROC curve
		  
'''   
def validate_model(model, testset, graph=False):
    # Compute F1 metric for system evaluation
    metric = metrics.F1Score(num_classes=1, threshold=0.5)

    y_true = list()
    y_pred = list()

    counter =0

    for line in testset:
        label = float(line["label"])
        data = torch.FloatTensor(line["data"])

        prediction = float(model(data))

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
        plt.plot(fpr,tpr, label = "ROC křivka datového modelu")
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('ROC křivka datového modelu')
        plt.show()






if __name__ == "__main__":

    ### Learning constants ###
    learning_rate = 0.000001
    test_learn_ratio = 0.2
    epoch_count = 25



    # Initialize network
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()
    optimizer.zero_grad()


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


    # mixing dataset
    label = None
    shufled_data = shuffle(good_data + bad_data)

    testset = list()
    dataset = list()
    counter = 0

    dataset_len = len(shufled_data)


    # split to testset and dataset
    for data_row in shufled_data:
        if counter < dataset_len*test_learn_ratio:
            testset.append(data_row)
        else:
            dataset.append(data_row)

        counter+=1

    
    # Load saved model for validation if needed
    # data_model = torch.load('../../models/v1.3_0.12err.pt')


    # traing network
    for i in range(epoch_count):
        train(dataset)
        print("Batch number:", i)


    # evaluate model
    validate_model(net, testset)


    # safe model
    torch.save(net, "./model_bigram.pt")




