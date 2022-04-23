import json
import re




class preprocess:

		def __init__(self):
			with open('./data/bigram_vocabulary_all.json') as json_file:
				self.bigrams_vocab2 = json.load(json_file)

		def urlDecode(self, url1):
				try:
						r = url1.encode('utf-8')
						res = r.decode('idna')
				except:
						print ("Can't process domain: ", url1)
						res = ''
				return res

		### split domain to bigrams
		def findBigrams(self, input_string):
				'''
				Parameters
				----------
				input_string : string
				Split domain string to bigrams.

				Returns
				-------
				bigram_list : list
						list of bigrams.

				'''
				bigram_list = []
				for i in range(0, (len(input_string)-1), 1):
						bigram_list.append(input_string[i] + input_string[i+1])
				return bigram_list

		### encode bigrams to integers 
		def bigrams2int(self, bigram_list):
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
				bigram_int = []
				for item in bigram_list:
						if item in self.bigrams_vocab2.keys():
								bigram_int.append(self.bigrams_vocab2[item])
						else:
								bigram_int.append(int(1))               
				return bigram_int
		
		### Domain preprocessing:
		### we get data as a domain string, we need to process it to vector format:
		def preprocessing(self, domain_str):
				

				'''
				Parameters
				----------
				domain_str : string
						input is a domain string in format <tld>.<domain_name> :
								'com.greycortex'
						We need to transform it to the list of bigrams:
								co, om, m., .g, gr, re, ey, tc, co, or, rt, te, ex
						For embedding layer, we replace each bigram with integer (according to the dictionary).
								850, 469,  91 264, 384, 186, 575, 351, 850, 82, 461, 753, 435  

				Returns
				-------
				bigram_int : list
						A list of integers where each integer corresponds a bigram.
				'''
				### decoding (if necessary)
				#domain_str = urlDecode(domain_str)
				### lower case
				domain_low = domain_str.lower()
				### Rotate domain to form <tld>.<domain_name>
				domain_rotated = self.rotate_domain(domain_low)
				### replace characters: numbers with 0
				domain0 = re.sub('\d', '0', domain_rotated)
				### replace non-ascii with ?
				domain_ascii = re.sub(r'[^\.\-0-9a-z]','?', domain0)
				### create bigrams
				bigrams = self.findBigrams(domain_ascii) # list of bigram
				### encode bigram to integer
				int_list = self.bigrams2int(bigrams) # list of integers
										
				return int_list
		
		def rotate_domain(self, domain_name):
			splited = domain_name.split('.')

			rotated = ''
			max = len(splited)
			for i in range(max):
				if i == (max-1):
					rotated += splited[(max-1)-i] 
				else:
					rotated += splited[(max-1)-i] + '.'

			return rotated
