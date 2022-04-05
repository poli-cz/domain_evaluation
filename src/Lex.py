import requests
import socket
import json
import csv
import urllib
import re
import io
import os
import time

import Database
import concurrent.futures
from datetime import timedelta

from pymongo import MongoClient
import pymongo

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import json
from sklearn.utils import shuffle






def is_empty(data):
	counter=0
	empty=0
	if data == None:
		return True
	for item in data.values():
		if item == None:
			empty+=1
		counter+=1


	return (1-(float(empty)/counter))

	
def get_database():
	client = MongoClient("mongodb://localhost/domains")
	return client['domains']
db = get_database()


class Net(nn.Module):
	def __init__(self):         # only structure of network
		super().__init__()
		self.fc1 = nn.Linear(13, 6000)
		self.fc2 = nn.Linear(6000, 50)
		self.fc3 = nn.Linear(50, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return torch.sigmoid(self.fc3(x))





class Lexical_analysis:


	def __init__(self, collection):
		self.collection = collection
		self.names = []
	'''		
	Parameters
	----------
	bigram_list : list
			Encoding a list of bigrams to the list of integers
			If bigram is not in the dictionary, it replaces with 1 (out of vocabulary token).

	Returns
	-------
	bigram_int : list
			list of integers.
	'''        
	def load_names(self):
		d = Database.Database('domains')
		collection = d.return_collection(self.collection)
		for domain in collection.find():
			self.names.append(domain['name'])

	'''		
	Parameters
	----------
	bigram_list : list
			Encoding a list of bigrams to the list of integers
			If bigram is not in the dictionary, it replaces with 1 (out of vocabulary token).

	Returns
	-------
	bigram_int : list
			list of integers.
	'''        
	def get_rtt(self, url):

		initial_time = time.time() #Store the time when request is sent
		request = requests.get("http://" + url)
		ending_time = time.time() #Time when acknowledged the request
		elapsed_time = ending_time - initial_time
		return elapsed_time
	'''		
	Parameters
	----------
	bigram_list : list
			Encoding a list of bigrams to the list of integers
			If bigram is not in the dictionary, it replaces with 1 (out of vocabulary token).

	Returns
	-------
	bigram_int : list
			list of integers.
	'''        
	def get_names(self):
		return self.names

	def get_ratings(self):
		return self.ratings
	'''		
	Parameters
	----------
	bigram_list : list
			Encoding a list of bigrams to the list of integers
			If bigram is not in the dictionary, it replaces with 1 (out of vocabulary token).

	Returns
	-------
	bigram_int : list
			list of integers.
	'''        
	def load_rating(self, file_path):
		try:
			with open('data/tlds.json', 'r') as openfile:
				self.ratings = json.load(openfile)
				#print("loaded")
		except:

			with open(file_path, newline='') as csvfile:
				spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
				tlds = []
				i=0
				for row in spamreader:
					if i == 0:
						i+=1
						continue

					try:
						domain = {"name": row[0], "badnes": row[1], "popularity": row[2]}
						tlds.append(domain)
					except:
						continue
					
					i+=1
			self.ratings = tlds
			#print("slowly loaded")


def regex_cnt(string, pattern):
	return len(re.findall(pattern+"$", string))

resolved = []

com = None

def match(name):
	found=False
	i=0
	for rating in ratings:

		if i == 0:
			rating = com 

		x = regex_cnt(name, "\."+rating['name'])
		if x:
			resolved.append({"name":name, "rating":rating})
			found=True
			break
	if not found:
		print("domain name: ",  name, " not resolved...")
	i+=1


lex = Lexical_analysis("badDomains")
#lex.load_names()
lex.load_rating('../Data/suffixes-v0-20190309.csv')



ratings = lex.get_ratings()







# process data for machine learning
# scale data from -5 to +5
def process_data(name, logging=None):

	domain_name = name['name']

	data = list()


	# 1. 
	# Domain name rating and lexical analysis
	rating_counter=0
	tld_rating=0
	for rating in ratings:
		x = regex_cnt(domain_name, "\."+rating['name'])
		if x:
			tld_rating += float(rating['badnes'])*(-5)
			rating_counter+=1

	rating_mean = float(tld_rating/rating_counter)
	data.append(rating_mean)



	## Domain level
	level = domain_name.count('.')
	if level == 1:
		data.append(1.0)
	elif level == 2:
		data.append(0.75)
	else:
		data.append(-0.25)


	lot_digits_flag = False
	## If there is digit in domain, it is not good
	if any(c.isdigit() for c in domain_name):
		data.append(-10)
		lot_digits_flag=True
	else:
		data.append(0.0)

	if logging:
		print(data)
		input()
	# 2. 
	# SSL rating and ssl expiration
	try:
		if name['ssl_data']['is_ssl']:
			cert_duradion = name['ssl_data']['ssl_data']['end_date'] - name['ssl_data']['ssl_data']['start_date']

			if cert_duradion > timedelta(days=300):
				data.append(5.0)
			elif cert_duradion > timedelta(days=80):
				data.append(0.6*5)
			elif cert_duradion > timedelta(days=30):
				data.append(0.3*5)
			else:
				data.append(0.1*5)

		# data not available
		else:
			data.append(-5.0)

		if name['ssl_data']['ssl_data']['issuer']:
			issuer = name['ssl_data']['ssl_data']['issuer']

			if issuer == 'Google Trust Services LLC':
				data.append(5.0)
			elif issuer == 'Amazon':
				data.append(5.0)
			elif issuer == "Let's Encrypt":
				data.append(2.0)
			elif issuer == "Cloudflare, Inc.":
				data.append(2.0)
			elif issuer == 'DigiCert Inc':
				data.append(1.0)
			else:
				data.append(0.0)
				

	except:
		if logging:
			print("[Log]: omiting ssl data")
		data.append(-5.0)
		data.append(0.0)

	if logging:
		print(data)
		input()
	# 3.
	# Geographical data
	try:
		if name['geo_data']:
			#print("Country:", name['geo_data']['country'])
			#print("Coordinates:", name['geo_data']['loc'])
			coordinates = name['geo_data']['loc'].split(",")
			data.append((float(coordinates[0])/90)*5)
			data.append((float(coordinates[1])/180)*5)
		else:
			data.append(0)
			data.append(0)

	except:
		if logging:
			print("[Log]: error in Geo data section")
		data.append(0)
		data.append(0)


	# 4. 
	# DNS data rating

	try:
		dns_data = name['dns_data']
		if dns_data['TXT'] is not None:
			data.append(1.0)
		else:
			data.append(0.0)


		if dns_data['MX'] is not None:
			data.append(2.0)
		else:
			data.append(0.0)

		if dns_data['SOA'] is not None:
			data.append(1.0)
		else:
			data.append(-1.0)

		if dns_data['NS'] is not None:
			data.append(0.5)
		else:
			data.append(-0.5)
			
	except:
		if logging:
			print("[Log]: error in DNS data section")
		data.append(-5.0)
		data.append(-5.0)
		data.append(-5.0)
		data.append(-5.0)


	# 5. 
	# Whois data
	try:
		registrar_data = name['whois_data'] 
		registration_duradion = name['whois_data']['expiration_date'] - name['whois_data']['creation_date']

		if registration_duradion > timedelta(days=5000):
			data.append(5.0)
		elif registration_duradion > timedelta(days=3000):
			data.append(0.6*5)
		elif registration_duradion > timedelta(days=1000):
			data.append(0.3*5)
		else:
			data.append(0.1*5)

		if registrar_data['dnssec'] is not None:
			data.append(5.0)
		else:
			data.append(0)

	except:
		data.append(-5.0)
		data.append(0.0)


	while len(data) < 13:
		data.append(0.0)

	if len(data) != 13:
		print(data)
		print("len error")
		return

	return data



