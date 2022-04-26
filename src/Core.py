""" File: Core.py
    Author: Jan Polisensky
    ----
    Abstraction over data resolver, machine learning models etc
    Provides interface over many modules
"""


# Import generic modules
import json
import time
import re
import os
from typing import List
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# Import ML and data-processing libraries
import tensorflow as tf
import torch
import numpy as np
from array import array
import pickle
from dotenv import load_dotenv
from os import getenv
import logging
import threading


# Import custom modules
import Database
from Data_loader import Base_parser
import SSL_loader
import Parser
from Parser import Net, Lexical_analysis
from Preprocessor import preprocess 



### constants for adjusting models ratio ###

auto_weight = True
## If set as FALSE, set desired weights ##
lexical_weight = 0.1
data_weight = 0.9
svm_weight = 0.1



## set grey zone for SVM prediction correction ##
grey_zone_width = 0.1


## Constants required by models
MAX_BIGRAM_LEN = 43



class clasifier:   
        """
        Core class, controling models, data resolving etc..
        ...
        Attributes
        ----------
                .env file is required for setup resolver timeout, db-connection string, etc, 
                for details see readme

        """     
        def __init__(self) -> None:
                load_dotenv()
                self.models = getenv("MODELS_FOLDER")
                self.resolver_timeout = int(getenv("RESOLVER_TIMEOUT"))
                self.hostname = None
                self.data = None
                self.loaded_data = False
                self.accuracy = 0

                # model for paralel data resolving
                self.paralel = getenv("PARALEL")


                # Load classification models
                self.lexical_model_path = str(getenv("LEXICAL_MODEL"))
                self.data_model_path = str(getenv("DATA_MODEL"))
                self.svm_model_path = str(getenv("SVM_MODEL"))
        
        # Function to load domain data
        def load_data(self, hostname: str) -> None:
                if not self.reset_data(hostname):
                        return self.data

                domain = Base_parser(hostname, self.resolver_timeout)


                ### Paralel or sequestial data-load ###
                if self.paralel == 'True':
                        print("using paralel")
                        threading.Thread(target=domain.load_dns_data).start()
                        threading.Thread(target=domain.load_geo_info).start()
                        threading.Thread(target=domain.load_whois_data).start()
                        threading.Thread(target=domain.load_ssl_data).start()

                else:
                #sequential data load
                        domain.load_dns_data()
                        domain.load_geo_info()
                        domain.load_whois_data()
                        domain.load_ssl_data()

                # Get loaded data #
                dns_data = domain.get_dns()
                geo_data = domain.get_geo_data()
                whois_data = domain.get_whois_data()
                ssl_data = domain.get_ssl_data()

                self.accuracy = np.around((self.is_empty(dns_data) + self.is_empty(geo_data) + self.is_empty(whois_data) + self.is_empty(ssl_data))/4, 3)

                print("[Info]: All data collected, data loss: ", np.around((1-self.accuracy)*100, 2), " %")

                in_data = {"name": hostname, "dns_data": dns_data, "geo_data": geo_data, "whois_data": whois_data, "ssl_data": ssl_data}
                
                lex = Lexical_analysis()
                self.data = lex.process_data(in_data)

                self.loaded_data = True
        
         # Prediction with lexical model
        def get_lexical(self, hostname: str) -> float:
                self.lexical_model = tf.saved_model.load(self.models + '/' + self.lexical_model_path)
                parse = preprocess()
                bigrams = parse.preprocessing(hostname)

                iter = MAX_BIGRAM_LEN - len(bigrams)
                for i in range(iter):
                        bigrams.append(0)
                if len(bigrams) > MAX_BIGRAM_LEN:
                        print("[Error]: Domain name to long, cant fit lexical model")
                        exit(1)  

                in_data = np.array([bigrams], dtype=np.float32)

                # Lexical models use inverse value
                return float(self.lexical_model(in_data))

        # Prediction with support vector machines model
        def get_svm(self, hostname: str) -> float:
                self.load_data(hostname)
                svm_model = pickle.load(open(self.models + '/' + self.svm_model_path, 'rb'))

                np_input = np.array([self.data], dtype=np.float32)
                prediction = svm_model.predict(np_input)

                return float(1 - prediction)

        # Prediction with data-based model
        def get_data(self, hostname: str) -> float:
                self.load_data(hostname)
                data_model = torch.load(self.models + '/' + self.data_model_path)

                torch_input = torch.tensor(self.data)
                prediction = data_model(torch_input)

                return float(1 - prediction)

        # prediction with mixed model
        def get_mixed(self, hostname: str):

                # Load data for model evaluation
                self.load_data(hostname)


                # get predictions of all three models, value needs to be inverted for calculation
                data = 1 - self.get_data(hostname)
                svm = 1 - self.get_svm(hostname)
                lexical = 1 - self.get_lexical(hostname)


                ## weight mode ##
                # auto -> based od accuracy
                # manual -> based on user input
                if auto_weight:
                        prediction = float(data*self.accuracy + lexical*(1-self.accuracy))
                else:
                        prediction = float((data_weight*data + lexical_weight*lexical) / (data_weight+lexical_weight))


                # svm acts like corrector
                if svm > (1 - grey_zone_width):
                        if prediction > 0.5:
                                self.accuracy = (self.accuracy+1)/2
                                prediction = prediction*(2/3)+1/3
                        else:
                                prediction+=svm_weight
                                self.accuracy = (self.accuracy+0.5)/2

                elif svm < (0.5 - grey_zone_width):
                        if prediction < 0.5:
                                self.accuracy = (self.accuracy+1)/2
                        else:
                                prediction-=svm_weight
                                self.accuracy = (self.accuracy+0.5)/2

                # Correction of bad results of prediction #
                if prediction > 1:
                        prediction = 1.00
                elif prediction < 0:
                        prediction = 0.00



                ### Inverting value to fit specifications, 1 -> bad domain, 0 -> good domain
                prediction = 1 - prediction        
                
                return prediction, self.accuracy

        # Loads domain data, usefull if you already have database with fetched data
        # Param data: JSON object representing domain data, for exact form see README
        def preload_data(self, data: list, hostname: str) -> None:
                self.hostname = hostname
                self.data = data
                self.loaded_data = True

        # Only fetched domain data and returns them
        def get_raw(self, hostname: str):
                domain = Base_parser(hostname, self.resolver_timeout)

                domain.load_dns_data()
                domain.load_geo_info()
                domain.load_whois_data()
                domain.load_ssl_data()

                dns_data = domain.get_dns()
                geo_data = domain.get_geo_data()
                whois_data = domain.get_whois_data()
                ssl_data = domain.get_ssl_data()

                raw_data = {"name": hostname, "dns_data": dns_data, "geo_data": geo_data, "whois_data": whois_data, "ssl_data": ssl_data}

                return raw_data

        # Reset data loader
        def reset_data(self, hostname: str) -> bool:
                
                if self.hostname != hostname:
                        self.data = None
                        self.accuracy = None
                        self.loaded_data = False
                        self.hostname = hostname
                        return True
                else:
                        return False

        # Define percentage of loss in data-category(whois/ssl/dns)
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