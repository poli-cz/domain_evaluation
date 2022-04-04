# Import basic modules and libraries
import json
import time
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# Import ML and data-processing libraries
import tensorflow as tf
import torch
import numpy as np
from array import array
import pickle



# Load custom modules
import Database
from Data_loader import Base_parser
import SSL_loader
import Lex
from Lex import Net
from Preprocessor import preprocess 


model = tf.saved_model.load('../models/domain_bigrams-furt-2020-11-07T11_09_21')
clf = pickle.load(open('../models/svm_model.smv', 'rb'))
net = torch.load('../models/net_0.149_err.pt')

print("[Log]: models loaded")






while True:
	parse = preprocess()
	print("Enter domain name:")
	in_name = input()
	bigrams = parse.preprocessing(in_name)


	iter = 43 - len(bigrams)
	for i in range(iter):
		bigrams.append(0)

	if len(bigrams) > 43:
		continue

	hostname = str(in_name)
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
	GK_prediction = np.around((1 - float(model(in_data))), 5)


	
	
	data_based_input = Lex.process_data(domain_data)
	print(data_based_input)

	SVM_prediction = clf.predict(np.array([data_based_input], dtype=np.float32))
	DATA_prediction = net(torch.tensor(data_based_input))
	



	prediction=(DATA_prediction*1.3+GK_prediction*0.2)/2
	prediction=np.around(float(prediction)*float(accuracy) + float(GK_prediction)*float(data_loss),4)


	print(GK_prediction, float(DATA_prediction), SVM_prediction)
	input()
	if prediction >0.5:
		print("Good site, rating:", np.around(prediction*100, 2), "%", hostname)
	else:
		if prediction < 0.1:
			print("Oh GOD, go away, rating:", np.around(prediction*100,2), "%", hostname)
		else:

			print("Bad site, rating:", np.around(prediction*100,2), "%", hostname)




	input()









	#input()





