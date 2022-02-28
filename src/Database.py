

# Custom module for databse usage
# requires to configure connection in connection string


from pymongo import MongoClient
import pymongo


class Database:

	def __init__(self, database_name):
		client = MongoClient("mongodb://localhost/domains")
		self.db = client[database_name]


	def return_collection(self, collection):
		return self.db[collection]

	def return_db(self):
		return self.db

	def insert_domain(self, collection_name, data):
		domain_collection = self.db[collection_name]
		print("Inserting: ", data, " into: ", collection_name)

		try:
			name = data['name']

		except:
			print("No domain name, are you inserting domain data??")
			print("This is what I got to insert: ", data)
			return False

		domain_collection.replace_one({'name': data['name']}, data, upsert=True)

	def get_stats(self, collection_list = ["goodDomains", "badDomains"]):
		if len(collection_list[0]) == 1:
			print("[Warning]: assuming argument as one collection")
			collection_list = [collection_list]
			
		for collection in collection_list:


			db_collection = self.db[collection]

			no_dns = 0
			no_whois = 0
			no_geo = 0
			total = 0

			ssl = 0
			no_ssl = 0
			not_discovered_ssl = 0

			for domain in db_collection.find():
				################################
				try:
					geo_data = domain['geo_data']
					if geo_data['loc'] is not None:
					# print(geo_data['loc'])
						
						txt = geo_data['loc']

						x = txt.split(',')

						x[0] = txt.split(' ')
						x[1] = txt.split(' ')

					if geo_data is None:
						no_geo=no_geo+1
				except:
					no_geo=no_geo+1
				################################
				try:
					whois_data = domain['whois_data']
					if whois_data is None:
						no_whois=no_whois+1        
				except:
					no_whois=no_whois+1
				################################
				try:
					dns_data = domain['dns_data']
					if dns_data is None:
						no_dns=no_dns+1
				except:
					no_dns=no_dns+1

				try:
					ssl_data = domain['ssl_data']
					if ssl_data['is_ssl'] is True:
						ssl = ssl+1
					
					elif ssl_data['is_ssl'] is False:
						no_ssl = no_ssl+1
				except:
					not_discovered_ssl=not_discovered_ssl+1
				################################
				total = total+1
			
			print("----- Basic statistics for collection:", collection, "-----")
			print("-------------------------------------------------------")
			print("Total domains: ", total)
			print(round(no_geo/total *100, 10), "% without geographical data (absolute)")
			print(round(no_dns/total*100, 3), "% without dns data (absolute)")
			print(round(no_whois/total*100, 3), "% without whois data (absolute)")
			print("SSL data statistics:")
			print("------------------")
			print("ssl data harvested for:", total-not_discovered_ssl, "domains")
			print(round(not_discovered_ssl/total*100, 3), "% not discovered ssl data")
			print(round(ssl/(total-not_discovered_ssl)*100, 4), "domains with ssl ON (relative)")
			print(round(no_ssl/(total-not_discovered_ssl)*100, 4), "domains with ssl OFF (relative)")
			print("-------------------------------------------------------")
	
#######################################################



if __name__ == "__main__":
	d = Database('domains')
	d.get_stats()
#d.get_stats("badDomains")