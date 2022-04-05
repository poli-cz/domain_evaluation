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
import argparse

# Load custom modules
import Database
from Data_loader import Base_parser
import SSL_loader
import Lex
from Lex import Net
from Preprocessor import preprocess 
from Core import clasifier



### Parse arguments ###
parser = argparse.ArgumentParser(description='domain name analysis tool')
parser.add_argument('domain_name', type=str, help='Required domain name')
args = parser.parse_args()
domain_name = args.domain_name


### Initialize classifier and get prediction for each model ###
cls = clasifier()
lexical = cls.get_lexical(domain_name)
data_based = cls.get_data(domain_name)
svm = cls.get_svm(domain_name)

combined, accuracy = cls.get_mixed(domain_name) # combining all three models, described in documentation


### Round values ###

combined = np.around(combined, 3)
accuracy = np.around(accuracy, 3)

svm = np.around(svm, 3)
lexical = np.around(lexical, 3)
data_based = np.around(data_based, 3)

### Output values ###

rating ={
    "domain_name" : domain_name,
    "lexical" : lexical,
    "data-based" : data_based,
    "svm" : svm,
    "combined": combined,
    "accuracy": accuracy
}

rating_json = json.dumps(rating, indent = 4)

with open(domain_name + '.json', "w") as outfile:
    outfile.write(rating_json)

exit(0)







