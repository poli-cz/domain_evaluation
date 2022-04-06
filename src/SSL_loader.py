

import whois
import socket
import concurrent.futures
import dns.resolver
import requests
import json
import urllib
import re
import io
import os
import time
import OpenSSL
import datetime
# Database
import Database
from datetime import timedelta


  
class SSL_loader:

	def __init__(self, domain_name, resolver_timeout: int):
		self.domain = domain_name
		self.timeout = resolver_timeout

	def GetRootCert(self, _cert):
		rootCerts = "./data/root_certs"
		issuer = _cert.get_issuer()


		found = False

		for subdir, dirs, files in os.walk(rootCerts, topdown="true"):
			for f in files:
				path = subdir + os.sep + f
				c = open(path).read()
				crt = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, c)
				subject = crt.get_subject()
				if subject == issuer:
					found = True
					break

		if found:
			return crt
		else:
			return None

 
	def GetCertChain(self, host):
		"""
		First it test the connection with host on port 443.
		If it timeouts go te next else do handshake and get the cert chain
		"""
		global version 

		cant_load = True
		cont = OpenSSL.SSL.Context(OpenSSL.SSL.SSLv23_METHOD)
		cont.set_timeout(self.timeout)
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.settimeout(self.timeout)
		try:
			sock.connect((host, 443))
			get = str.encode("GET / HTTP/1.1\nUser-Agent:Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36\n\n")
			sock.send(get)
			sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			sock = OpenSSL.SSL.Connection(context=cont, socket=sock)
			sock.settimeout(self.timeout)
			sock.connect((host, 443))
			sock.setblocking(1)
			sock.set_connect_state()
			sock.set_tlsext_host_name(str.encode(host))
			sock.do_handshake()
		except socket.gaierror as e:
			if cant_load:
				#print("[Info]: SSL loader cant load all data")
				cant_load = False
			return None
		except socket.timeout as e:
			if cant_load:
				#print("[Info]: SSL loader cant load all data")
				cant_load = False
			return None
		except OpenSSL.SSL.Error as e:
			if cant_load:
				#print("[Info]: SSL loader cant load all data")
				cant_load = False
			return None
		except ConnectionRefusedError as e:
			if cant_load:
				#print("[Info]: SSL loader cant load all data")
				cant_load = False
			return None
		except OSError as e:
			if cant_load:
				#print("[Info]: SSL loader cant load all data")
				cant_load = False
			return None



		chain = sock.get_peer_cert_chain()
		version = sock.get_protocol_version_name()
		try:
			sock.shutdown()
			sock.close()
		except:
			print("[Fatal]: error closing socket")

		return [chain, version]



	def get_cert(self):
		host = self.domain
		certsAndHosts = {}
		rootCert = None

		returned = self.GetCertChain(host)
		chain = returned[0] if returned else None
		version = returned[1] if returned else None


		if not chain and not host.startswith("www."):
			host = "www." + host
			returned = self.GetCertChain(host)
			chain = returned[0] if returned else None
			version = returned[1] if returned else None

		if chain:
			for i in range(len(chain)):
				rootCert = self.GetRootCert(chain[i])
				if rootCert:
					break

		if not rootCert and not host.startswith("www."):
			host = "www." + host
			returned = self.GetCertChain(host)   
			chain = returned[0] if returned else None
			version = returned[1] if returned else None

			if chain:
				for i in range(len(chain)):
					rootCert = self.GetRootCert(chain[i])
					if rootCert:
						break
		if rootCert and chain and version:

			certsAndHosts[host] = [chain[0],rootCert,version]
			return certsAndHosts
		else:
			return None



def insert_ssl_data(ssl_data, domain_name):
	print("For: ", domain_name)

	d = Database.Database('domains')

	domain_collection = d.return_collection("goodDomains")

	orig = domain_collection.find_one({'name': domain_name})

	if domain_name != orig['name']:
		print("Domain name mismatch, exiting")
		exit(1)

	print(orig['dns_data'])
	data = {
		'name': domain_name,
		'dns_data': orig['dns_data'],
		'geo_data': orig['geo_data'],
		'whois_data': orig['whois_data'],
		'ssl_data': ssl_data
	}
	#print(data)
	domain_collection.replace_one({'name': domain_name},data, upsert=True)



# 
def discover_ssl(name, timeout: int):
	print("[Info]: SSL discovery started")
	s = SSL_loader(name, timeout)

	ssl_data = {
		"is_ssl": False, 
		"ssl_data": 
			{
			"issuer": None,
			"end_date": None,
			"start_date": None
			}
		}


		
	data = s.get_cert()
	_domCert = None

	try:
		_domCert = data[name][0]

	except:
		try:
			url = "www." + name
			_domCert = data[url][0]
		except:
			return ssl_data
		
	ssl_data['is_ssl'] = True

	cert_data = str(_domCert.get_issuer())
	atributes = cert_data.split('/')
		
	i = 0
	for atribute in atributes:
		if i == 0:
			i=i+1
			continue
		if i == 2:
			ssl_data['ssl_data']['issuer'] = atribute
		i=i+1
	domNotAfter = (_domCert.get_notAfter()).decode("utf-8")[:-1]
	domNotAfter = datetime.datetime.strptime(domNotAfter, "%Y%m%d%H%M%S")

	domNotBefore = (_domCert.get_notBefore()).decode("utf-8")[:-1]
	domNotBefore = datetime.datetime.strptime(domNotBefore, "%Y%m%d%H%M%S")

	ssl_data['ssl_data']['start_date'] = domNotBefore
	ssl_data['ssl_data']['end_date'] = domNotAfter
	return ssl_data


		
if __name__ == '__main__':

	d = Database.Database('domains')
	collection = d.return_collection("goodDomains")
	domain_names = []

	for domain in collection.find():
		try:
			ssl = domain['ssl_data']
		except:
			domain_names.append(domain['name'])
			print("apending..")


	with concurrent.futures.ThreadPoolExecutor(max_workers=50) as pool:
		list(pool.map(discover_ssl, domain_names))

	
	
