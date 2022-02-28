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



class Lexical_analysis:


	def __init__(self, collection):
		self.collection = collection
		self.names = []

	def load_names(self):
		d = Database.Database('domains')
		collection = d.return_collection(self.collection)
		for domain in collection.find():
			self.names.append(domain['name'])

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

	def load_rating(self, file_path):
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
			print(len(tlds), "tlds loaded")
			self.ratings = tlds

	def load_candidates(self, file_path):
		with open(file_path, newline='') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			candidates = []
			i=0
			for row in spamreader:
				if i == 0:
					i+=1
					continue

				try:
					raiting = {"candidate": row[0], "neg": row[3], "pos": row[4]}
					candidates.append(raiting)
				except:
					continue
				
				i+=1
			print(len(candidates), 'words loaded')
			self.candidates = candidates

	def get_candidates(self):
		return self.candidates

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
			print(rating['name'])
			break
	if not found:
		print("domain name: ",  name, " not resolved...")
	i+=1


lex = Lexical_analysis("badDomains")
lex.load_names()
lex.load_rating('../Data/suffixes-v0-20190309.csv')
lex.load_candidates('../Data/candidates.domain.csv')

names = lex.get_names()
ratings = lex.get_ratings()
candidates = lex.get_candidates()


d = Database.Database('domains')

good_collection = d.return_collection("goodDomains")
bad_collection = d.return_collection("badDomains")


for name in bad_collection.find():
	domain_name = name['name']
	badnes = 0
	print(domain_name)

	# Domain ratings 
	for rating in ratings:
		x = regex_cnt(domain_name, "\."+rating['name'])
		if x:
			print("match in domain name:", rating['badnes'], rating['name'])
			

	# Wordlist rating
	for candidate in candidates:
		if candidate['candidate'] in domain_name:
			print("matched word:",candidate['candidate'], "==> good:" ,candidate['pos'], "bad:", candidate['neg'])
			
	
	# SSL rating and ssl expiration
	if name['ssl_data']['is_ssl']:
		print(name['ssl_data']['ssl_data']['issuer'])
		print(name['ssl_data']['ssl_data']['end_date'] - name['ssl_data']['ssl_data']['start_date'])
	else:
		print("ssl data for this domain NOT available")

	# Rtt
	#
	print("RRT: ", str(lex.get_rtt(name['name'])) + ' s')
	#


	# Whois registrator rating
	#
	# TODO
	#


	# Geo data 
	if name['geo_data']:
		print("Country:", name['geo_data']['country'])
		print("Coordinates:", name['geo_data']['loc'])
	

	# DNS data
	# if name['dns']
	# is txt used for verification? 
	input()





# for name in new_collection.find():
# 	domain_name = name['name']
# 	badnes = 0
# 	for candidate in candidates:
# 		if candidate['candidate'] in domain_name:
# 			badnes = badnes + float(candidate['pos']) - float(candidate['neg'])

# 	print(badnes, "for:", domain_name)
# 	input()


# 		x = regex_cnt(name, "\."+rating['name'])
# 		if x:
# 			my_dict = {"name":name, "badnes": rating['badnes'], "popularity": rating['popularity']}
# 			new_collection.insert_one(my_dict)
# 			print(len(names))
# 			names.remove(name)



