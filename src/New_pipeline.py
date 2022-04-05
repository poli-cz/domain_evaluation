# Import basic modules and libraries
import json
import time
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# Import ML and data-processing libraries
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
import numpy as np
from array import array
import pickle
import torch.optim as optim


# Load custom modules
import Database
from Data_loader import Base_parser
import SSL_loader
import Lex
from Lex import Net
from Preprocessor import preprocess 
from Core import clasifier


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








if __name__ == "__main__":






        # Initialize network
        net = Net()
        optimizer = optim.Adam(net.parameters(), lr=0.000001, )
        loss_fn = nn.BCELoss()
        optimizer.zero_grad()


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
        counter = 0


        # split to testset and dataset
        for piece in merged:
                if counter < 5000:
                        testset.append(piece)
                else:
                        dataset.append(piece)

                counter+=1

        cls = clasifier()
        count = 0
        counter = 0
        for item in dataset:
                print("[Info]: Enter domain name")
                domain_name = str(input())
                #cls.preload_data(item['data'], item['domain'])
                mixed_data = cls.get_mixed(domain_name)
                print("--------------------------------------")
                print(cls.get_data(domain_name), cls.get_svm(domain_name), cls.get_lexical(domain_name))
                print(mixed_data)
                print("--------------------------------------")


                input()





	#





