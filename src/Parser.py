""" File: Parser.py
    Author: Jan Polisensky
    ----
    Collection of modules and functions for evaluation domain data
"""


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
from sklearn.utils import shuffle


class Net(nn.Module):
	"""
        Class defining neural network, needs to be present for loading model

        """ 
	def __init__(self) -> None:         
		super().__init__()
		self.fc1 = nn.Linear(13, 6000)
		self.fc2 = nn.Linear(6000, 50)
		self.fc3 = nn.Linear(50, 1)

	def forward(self, x) -> torch.sigmoid:
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return torch.sigmoid(self.fc3(x))


class Lexical_analysis:
	"""
        Class providing functions to perform data evaluating over domain values
	as well as loading pre-made tld ratings and so on...

        """ 

	def __init__(self):
		self.names = []

	# Load domain names from given collection
	def load_names(self, col):
		d = Database.Database('domains')
		collection = d.return_collection(col)
		for domain in collection.find():
			self.names.append(domain['name'])

	# Computes RTT to given host
	def get_rtt(self, url):

		initial_time = time.time() #Store the time when request is sent
		request = requests.get("http://" + url)
		ending_time = time.time() #Time when acknowledged the request
		elapsed_time = ending_time - initial_time
		return elapsed_time
     
	def get_names(self):
		return self.names

	def get_ratings(self):
		return self.ratings

	def regex_cnt(self, string, pattern):
		return len(re.findall(pattern+"$", string))

	# Function that load tlds rating from saved file, used for evaluation
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

	# main function providing translation of collected data to rating
	def process_data(self, name, logging=None) -> dict:
		self.load_rating('../Data/suffixes-v0-20190309.csv')
		ratings = self.get_ratings()
		domain_name = name['name']

		data = list()


		# 1. 
		# Domain name rating and lexical analysis
		rating_counter=0
		tld_rating=0
		for rating in ratings:
			x = self.regex_cnt(domain_name, "\."+rating['name'])
			if x:
				tld_rating += float(rating['badnes'])*(-5)
				rating_counter+=1

		if rating_counter == 0:
			data.append(-3)
		else:
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










