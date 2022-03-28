import json
import time
import re
import concurrent.futures
import Database
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from pymongo import MongoClient
import pymongo

maxlen = list()
a = list()
with open('./data/bigram_vocabulary_all.json') as json_file:
	bigrams_vocab2 = json.load(json_file)
def get_database():
	client = MongoClient("mongodb://localhost/domains")
	return client['domains']


db = get_database()
bad_bigram_dataset = db['good_bigram_dataset']




import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

import numpy as np
from array import array
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
from Data_loader import Base_parser
import SSL_loader
import Lex
from Lex import Net
import tensorflow as tf
from Preprocessor import preprocess 


model = tf.saved_model.load('../models/domain_bigrams-furt-2020-11-07T11_09_21')




d = Database.Database('domains')
bad_data = list()
good_data = list()


good_collection = d.return_collection("bad_dataset")
bad_collection = d.return_collection("good_dataset")


for name in good_collection.find():
	good_data.append(name)


for name in bad_collection.find():
	bad_data.append(name)


print(len(bad_data))
print(len(good_data))



label = None
dats = shuffle(good_data + bad_data)

counter=0
count=0

for domain in dats:
	parse = preprocess()
	bigrams = parse.preprocessing(domain['domain'])


	iter = 43 - len(bigrams)
	for i in range(iter):
		bigrams.append(0)

	if len(bigrams) > 43:
		continue

	hostname = str(domain['domain'])
	domain_ = Base_parser(hostname)

	domain_.load_dns_data()
	domain_.load_geo_info()
	domain_.load_whois_data()
	domain_.load_ssl_data()
	

	dns_data = domain_.get_dns()
	geo_data = domain_.get_geo_data()
	whois_data = domain_.get_whois_data()
	ssl_data = domain_.get_ssl_data()
	accuracy = np.around((Lex.is_empty(dns_data) + Lex.is_empty(geo_data) + Lex.is_empty(whois_data) + Lex.is_empty(ssl_data))/4, 1)
	data_loss = 1 - accuracy
	domain_data = {"name": hostname, "dns_data": dns_data, "geo_data": geo_data, "whois_data": whois_data, "ssl_data": ssl_data}
	in_data = np.array([bigrams], dtype=np.float32)
	DATA_prediction = np.around(float(Lex.process_data(domain_data)), 5)
	GK_prediction = np.around((1 - float(model(in_data))), 5)
	prediction=(DATA_prediction*1.3+GK_prediction*0.2)/2
	prediction=np.around(float(prediction)*float(accuracy) + float(GK_prediction)*float(data_loss),4)


	print(GK_prediction, DATA_prediction)

	if prediction >0.5:
		print("Good site, rating:", np.around(prediction*100, 2), "%", hostname, domain['label'])
	else:
		if prediction < 0.1:
			print("Oh GOD, go away, rating:", np.around(prediction*100,2), "%", hostname, domain['label'])
		else:

			print("Bad site, rating:", np.around(prediction*100,2), "%", hostname, domain['label'])






	

	

	

	

	#input()





