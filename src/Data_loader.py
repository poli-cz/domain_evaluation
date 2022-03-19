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
import csv

# Database
import Database
import SSL_loader

from pymongo import MongoClient
import pymongo



forbiddenIps = {"0.0.0.0", "127.0.0.1", "255.255.255.255"} # nonsense IPs, feel free to add more
nonvalidTypes = {"csv"}  
validTxtTypes = {"plain", "octet-stream", "html"} 
validArchTypes = {"x-gzip"}  
ipRegEx = r"^((?:(?:(?:(?:(?:(?:(?:[0-9a-fA-F]{1,4})):){6})(?:(?:(?:(?:(?:[0-9a-fA-F]{1,4})):(?:(?:[0-9a-fA-F]{1,4})))|(?:(?:(?:(?:(?:25[0-5]|(?:[1-9]|1[0-9]|2[0-4])?[0-9]))\.){3}(?:(?:25[0-5]|(?:[1-9]|1[0-9]|2[0-4])?[0-9])))))))|(?:(?:::(?:(?:(?:[0-9a-fA-F]{1,4})):){5})(?:(?:(?:(?:(?:[0-9a-fA-F]{1,4})):(?:(?:[0-9a-fA-F]{1,4})))|(?:(?:(?:(?:(?:25[0-5]|(?:[1-9]|1[0-9]|2[0-4])?[0-9]))\.){3}(?:(?:25[0-5]|(?:[1-9]|1[0-9]|2[0-4])?[0-9])))))))|(?:(?:(?:(?:(?:[0-9a-fA-F]{1,4})))?::(?:(?:(?:[0-9a-fA-F]{1,4})):){4})(?:(?:(?:(?:(?:[0-9a-fA-F]{1,4})):(?:(?:[0-9a-fA-F]{1,4})))|(?:(?:(?:(?:(?:25[0-5]|(?:[1-9]|1[0-9]|2[0-4])?[0-9]))\.){3}(?:(?:25[0-5]|(?:[1-9]|1[0-9]|2[0-4])?[0-9])))))))|(?:(?:(?:(?:(?:(?:[0-9a-fA-F]{1,4})):){0,1}(?:(?:[0-9a-fA-F]{1,4})))?::(?:(?:(?:[0-9a-fA-F]{1,4})):){3})(?:(?:(?:(?:(?:[0-9a-fA-F]{1,4})):(?:(?:[0-9a-fA-F]{1,4})))|(?:(?:(?:(?:(?:25[0-5]|(?:[1-9]|1[0-9]|2[0-4])?[0-9]))\.){3}(?:(?:25[0-5]|(?:[1-9]|1[0-9]|2[0-4])?[0-9])))))))|(?:(?:(?:(?:(?:(?:[0-9a-fA-F]{1,4})):){0,2}(?:(?:[0-9a-fA-F]{1,4})))?::(?:(?:(?:[0-9a-fA-F]{1,4})):){2})(?:(?:(?:(?:(?:[0-9a-fA-F]{1,4})):(?:(?:[0-9a-fA-F]{1,4})))|(?:(?:(?:(?:(?:25[0-5]|(?:[1-9]|1[0-9]|2[0-4])?[0-9]))\.){3}(?:(?:25[0-5]|(?:[1-9]|1[0-9]|2[0-4])?[0-9])))))))|(?:(?:(?:(?:(?:(?:[0-9a-fA-F]{1,4})):){0,3}(?:(?:[0-9a-fA-F]{1,4})))?::(?:(?:[0-9a-fA-F]{1,4})):)(?:(?:(?:(?:(?:[0-9a-fA-F]{1,4})):(?:(?:[0-9a-fA-F]{1,4})))|(?:(?:(?:(?:(?:25[0-5]|(?:[1-9]|1[0-9]|2[0-4])?[0-9]))\.){3}(?:(?:25[0-5]|(?:[1-9]|1[0-9]|2[0-4])?[0-9])))))))|(?:(?:(?:(?:(?:(?:[0-9a-fA-F]{1,4})):){0,4}(?:(?:[0-9a-fA-F]{1,4})))?::)(?:(?:(?:(?:(?:[0-9a-fA-F]{1,4})):(?:(?:[0-9a-fA-F]{1,4})))|(?:(?:(?:(?:(?:25[0-5]|(?:[1-9]|1[0-9]|2[0-4])?[0-9]))\.){3}(?:(?:25[0-5]|(?:[1-9]|1[0-9]|2[0-4])?[0-9])))))))|(?:(?:(?:(?:(?:(?:[0-9a-fA-F]{1,4})):){0,5}(?:(?:[0-9a-fA-F]{1,4})))?::)(?:(?:[0-9a-fA-F]{1,4})))|(?:(?:(?:(?:(?:(?:[0-9a-fA-F]{1,4})):){0,6}(?:(?:[0-9a-fA-F]{1,4})))?::)))))|((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
ValidHostnameRegex = r"(?:[a-z0-9](?:[a-z0-9-_]{0,61}[a-z0-9])?\.)+[a-z0-9][a-z0-9-_]{0,61}[a-z0-9]"


# if not used, limited number of requests
#ip_auth_token="f6157341b9e078"  # medikem token
ip_auth_token="6b3b15bcf578ec"  # seznam token
#ip_auth_token="7b7427498417ed"  # medikem token

# pip install git+https://github.com/rthalley/dnspython


class Data_loader:
    def get_hostnames(self, file_path, position, max=1000):
        with open(file_path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            i = 0
            top_1k = []
            for row in spamreader:
                if i == max:
                    break
                try:
                    top_1k.append(row[position])
                    i=i+1
                except:
                    continue

            return top_1k
    
    def get_links(self, file_path):
        links = []
        with open(file_path) as csvf:
            reader = csv.reader(csvf)
            for row in reader:
                links.append(row[1])
            links = links[12:]
            links = [x for x in links if x!='']
            return links

    def clean_links(self, links):
        out_links = []
        for link in links:
            domain = re.search(ValidHostnameRegex, link)
            if domain:

                out_links.append(domain.group(0))

        return out_links
    
    def get_hostnames_from_links(self, input):
        ips = []
        hostnames = []
        i = 0
        for source in input:
            print("LOADED", i)
            i=i+1
            if i > 60:
                return hostnames
            if source.startswith("http"):
                try:
                    retrieved = urllib.request.urlretrieve(source, filename=None)
                except urllib.error.HTTPError as e:
                    print(str(e) + " " + source)
                    continue
                except urllib.error.URLError as e:
                    print(str(e) + " " + source)
                    continue
                # retrieved file
                file_tmp = retrieved[0]

                # file type of retrieved file
                file_info = retrieved[1]

                ctype = file_info.get_content_subtype()
                print(ctype)
                if ctype in nonvalidTypes:
                    continue

                print("Reading " + source + " " + ctype)

                if ctype in validTxtTypes:
                    with io.open(file_tmp, "r", encoding="utf-8") as f:
                        for line in f:
                            # All kinds of comments are being used in the sources, they could contain non-malicious domains
                            if len(line) != 0 and  \
                                    not line.startswith("#") and \
                                    not line.startswith(";") and \
                                    not line.startswith("//"):
                                x = re.search(ipRegEx, line)
                                if x:
                                    ip = x.group()
                                    if ip not in forbiddenIps:
                                        #print(ip)
                                        pass

                                        ##ips.append(ip)
                                    # if there is a nonsense ip the script still needs to ask if 
                                    # there is a domain because some of the sources look like this: 0.0.0.0 adservice.google.com.vn
                                    else:
                                        #print(ip)
                                        pass

                                else:
                                    domain = re.search(ValidHostnameRegex, line)
                                    if domain:

                                        hostnames.append(domain.group(0))
                    os.remove(file_tmp)
        return hostnames

class Base_parser:
    def __init__(self, hostname):
        self.hostname = hostname
        self.dns = None
        self.ip = None
        self.geo_data = None
        self.whois_data = None
        self.ssl_data = None

        self.dns_resolver = dns.resolver.Resolver()
        self.dns_resolver.nameservers = ["8.8.8.8", "8.8.4.4"]
        self.dns_resolver.timeout = 90
        self.dns_resolver.lifetime = 90


    def get_dns(self):
        return self.dns

    def get_ip(self):
        return self.ip

    def get_geo_data(self):
        return self.geo_data

    def get_ssl_data(self):
        return self.ssl_data

    def get_whois_data(self):
        return self.whois_data

    def load_whois_data(self):
        whois_record = {}
        try:
            types = ['registrar', 'creation_date', 'expiration_date', 'dnssec', 'emails']
            w = whois.whois(self.hostname)
            i = 0
            for type in types:
                try:
                    whois_record[types[i]] = w[types[i]]
                except:
                    whois_record[types[i]] = None

                i=i+1
            self.whois_data = whois_record
            return True

        except Exception as e:
            print("Failed to get some whois data, continue without them")
            return False

    def load_dns_data(self):
        print("Loading DNS data")
        types = ['A', 'AAAA', 'CNAME', 'SOA', 'NS', 'MX', 'TXT']
        #types = ['TXT']
        dns_records = {}
        i = 0
        for type in types:
            result = None;
            try:
                result = self.dns_resolver.resolve(self.hostname, type)
            except Exception as e:
                #print(type + " is not available for this hostname")
                dns_records[types[i]] = None
                i=i+1
                continue

            #print(type + " " + self.hostname + " --> " + str(result[0]))
            #input()
            if type == 'A':
                self.ip = result[0]
            dns_records[types[i]] = str(result[0])
            i=i+1

        self.dns = dns_records

    def load_geo_info(self, ip=None):
        print("Loading Geo info data")
        if ip is None:
            if self.ip is None:
                print("Ip of hostname not discovered, doing it manualy...")
                try:
                    self.ip = self.ip_from_host()[self.hostname][0]
                except:
                    print("Hostname cannot be resolved")
                return False
        else:
            self.ip = ip
        
        geo_data = {}
        keys = ['country', 'region' ,'city' ,'loc' ,'org']
        url =  "https://ipinfo.io/" + str(self.ip) + "/?token=" + ip_auth_token
        raw_json = None
        try:
            raw_json = requests.get(url).json()
        except:
            self.geo_data = None
            return
        for i in range(len(keys)):
            try:
                geo_data[keys[i]] = raw_json[keys[i]]
            except:
                geo_data[keys[i]] = None

        self.geo_data = geo_data

    def load_ssl_data(self):
        self.ssl_data = SSL_loader.discover_ssl(self.hostname)

    def ip_from_host(self):
        hostname = self.hostname

        ips = []
        domainsIps = {}

        try:
            answer = self.dns_resolver.resolve(hostname)

            for item in answer:
                ips.append(item.to_text())

            domainsIps[hostname] = ips
            return domainsIps

        except Exception as e:
            print(answer)
            print(ips)
  
            print(str(e))
            domainsIps[hostname] = []
            return domainsIps

# fetch all data
def get_data(hostname):
    domain = Base_parser(hostname)
    domain.load_dns_data()
    domain.load_geo_info()
    domain.load_whois_data()
    

    dns_data = domain.get_dns()
    geo_data = domain.get_geo_data()
    whois_data = domain.get_whois_data()
 
   # return {"name": hostname, "dns_data": dns_data}
    domain_data = {"name": hostname, "dns_data": dns_data, "geo_data": geo_data, "whois_data": whois_data}
    return domain_data

def get_database():
    client = MongoClient("mongodb://localhost/domains")
    return client['domains']

# insert good domains
def insert(hostname):
    db = get_database()
    print(hostname)
    good_domain_collection = db['goodDomains']
    data = get_data(hostname)
    print("G")
    print(str(good_domain_collection.replace_one({'name': data['name']},data, upsert=True)))

# insert bad domains
def insert_bad(hostname):
    db = get_database()
    bad_domain_collection = db['badDomains']
    data = get_data(hostname)
    print("B")
    print(str(bad_domain_collection.replace_one({'name': data['name']},data, upsert=True)))
    
def geo_corrector(collection):
    db = get_database()
    bad_domain_collection = db[collection]
    for domain in bad_domain_collection.find():
        print(domain)
        try:
            geo_data = domain['geo_data']
        except:
            print(domain)

            if domain['dns_data']['A'] != None:
                print(domain['name'], "resolvable!")
                p = Base_parser(domain['name'])
                p.load_geo_info(domain['dns_data']['A'])
                geo_data = p.get_geo_data()
                domain['geo_data'] = geo_data
                print(domain['name'])
                print(bad_domain_collection.replace_one({'name': domain['name']}, domain, upsert=True))
                print("corrected")
fetched = True

l = Data_loader()
db = get_database()

d = Database.Database('domains')
allDomains = d.return_db()





if not fetched:
    ### Fetch data ###
    raw_blacklisted = l.get_links('../Data/blacklists-2021.01.csv')
    good_hostnames = l.get_hostnames('../Data/top-1m.csv', 1, 100000)
    bad_hostnames_1 = l.get_hostnames_from_links(raw_blacklisted)
    raw_spam = l.get_hostnames('../Data/spyware.csv', 0, 70000)

    # clean links from source
    cleaned_spam = l.clean_links(raw_spam)

    bad_hostnames = cleaned_spam + bad_hostnames_1
    ### inserting data in db ###
    good_domains = {
        "name": "good_domains",
        "domain_count": len(good_hostnames),
        "names": good_hostnames
    }

    bad_domains = {
        "name": "bad_domains",
        "domain_count": len(bad_hostnames),
        "names": bad_hostnames
    }
    #############################
    all_doms = d.return_collection('allDomains')

    result = all_doms.insert_many([good_domains, bad_domains])


if __name__ == '__main__':

    # Create a new collection
    good_domains = d.return_collection('allDomains')

    all_good_domains = good_domains.find_one({"name": "good_domains"})
    all_bad_domains = good_domains.find_one({"name": "bad_domains"})

    bad_domains = all_good_domains['names']
    good_domains = all_good_domains['names']



    bad_collection = d.return_collection("goodDomains")
    bad_in_db = []
    good_collection = d.return_collection("goodDomains")

    for domain in bad_collection.find():
        bad_in_db.append(domain['name'])


    print("filtering for duplicit records")
    bad_domains = list(dict.fromkeys(bad_domains))


    print("filtering already fetched data from database")
    final = []
    i=0
    for name in bad_domains:
        if name not in bad_in_db:
            final.append(name)
            i=i+1
            print(i)
            if i > 7000:
                break


    print("Ok, found", i, " Not fetched ips")

    print(len(good_domains))
    #input()


    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as pool:
        list(pool.map(insert, final))
    #  list(pool.map(insert_bad, final))


    #d.get_stats("goodDomains")
    #d.get_stats("badDomains")
    #get_basic_stats("goodDomains")
    #get_basic_stats("badDomains")

    #geo_corrector("goodDomains")

            


