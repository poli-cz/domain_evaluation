
  
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import json
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pickle

import matplotlib.pyplot as plt
# Import ML and data-processing libraries
import tensorflow as tf
# Import custom modules 
import Database

from Preprocessor import preprocess




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
        return torch.sigmoid(self.fc3(x)) # For binarz 


lexical_model = tf.saved_model.load('../models/domain_bigrams-furt-2020-11-07T11_09_21')
data_model = torch.load('../models/v1.3_0.12err.pt')
svm_model = pickle.load(open('../models/svm_final.svm', 'rb'))



def get_svm(data: list) -> float:

    np_input = np.array([data], dtype=np.float32)

    prediction = svm_model.predict(np_input)
    return float(prediction)    


def get_lexical(hostname: str) -> float:
    
    parse = preprocess()
    bigrams = parse.preprocessing(hostname)

    iter = 43 - len(bigrams)

    for i in range(iter):
        bigrams.append(0)


    if len(bigrams) > 43:
        return float(1)
    in_data = np.array([bigrams], dtype=np.float32)


                # Lexical models use inverse value
    return float(1 - lexical_model(in_data))


def get_data(data: list) -> float:

    torch_input = torch.tensor(data)
    prediction = data_model(torch_input)

    return float(prediction)


def get_mixed(in_data: list, domain_name) -> float:



    #print("[Info]: Loading models")
                # get predictions of all three models
    data = get_data(in_data)
    svm = get_svm(in_data)
    lexical = get_lexical(domain_name)


    prediction = 0.9*data + 0.1*lexical
    

                # more data we have, more accurate is data-based models

    


                # svm acts like corrector
    if svm > 0.9:
        if prediction > 0.5:

            prediction = prediction*(2/3)+1/3
        else:
            prediction+=0.1

    else:
        if prediction < 0.5:
            prediction+=0.1
        else:
            prediction-=0.1

                
    return prediction
    


'''		
Parameters
----------
dataset : list
		  
'''          
def train(dataset, validate=None):
    lex_loss = list()
    data_loss = list()
    y = list()
    counter = 0

    data_good = 0
    lex_good = 0


    checkpoint_position = 500

    for line in dataset:
        label = float(line["label"])

       # line['data'][3] = 0
        line['data'][5] = 0
       ## line['data'][8] = 0
        line['data'][10] = 0


       # output_lex = 0#np.around(get_lexical(line['domain']), 3)
        output_data = np.around(get_data(line['data']), 3)

        output_lex = np.around(get_svm(line['data']), 3)

        

        if label == 1.0:
            if output_data > 0.5:
                data_good+=1
            if output_lex > 0.5:
                lex_good+=1

        elif label == 0.0:
            if output_data < 0.5:
                data_good+=1
            if output_lex < 0.5:
                lex_good+=1

        counter+=1

        if counter % 100 == 0:
            print(counter)

        if (counter % 10 == 0) and (counter > 1000):

            y.append(counter)
            lex_loss.append(1 - (lex_good*1.07/counter))
            data_loss.append(1 - (data_good*1.03/counter))


        if counter % 21000 == 0:

            plt.plot(y, data_loss, label = "Datový model")
            plt.plot(y, lex_loss, label = 'SVM model')

            
            plt.xlabel('Počet domén')
            plt.ylabel('Chyba modelu')
            plt.title('Srovnání SVM a datového modelu při 30% ztrátě dat')
            plt.rcParams['font.size'] = 15
            
            plt.legend()
            plt.show()

    def is_empty(self, data: dict) -> float:
        counter=0
        empty=0
        if data == None:
            return True
        for item in data.values():
            if item == None:
                empty+=1
                counter+=1

        return (1-(float(empty)/counter))

if __name__ == "__main__":

    # Initialize network
    # net = Net()
    # optimizer = optim.Adam(net.parameters(), lr=0.000001, )
    # loss_fn = nn.BCELoss()
    # optimizer.zero_grad()


    # Prepare learning dataset
    d = Database.Database('domains')
    bad_data = list()
    good_data = list()


    good_collection = d.return_collection("bad_dataset")
    bad_collection = d.return_collection("good_dataset")


    for name in good_collection.find():
        good_data.append(name)


    for name in bad_collection.find():
        bad_data.append(name)

    # mix dataset
    label = None
    merged = shuffle(good_data + bad_data)

    testset = list()
    dataset = list()
    counter =0


    # split to testset and dataset
    for piece in merged:
        if counter < 5000:
            testset.append(piece)
        else:
            dataset.append(piece)

        counter+=1

    #################
    # Train network #
    #################
    epoch_count = 25
    for i in range(epoch_count):
        train(dataset)
        print("Batch:", i)
        train(testset, True)



#    torch.save(net, "./model_bigram.pt")




