import whois
import socket
import concurrent.futures
import dns.resolver
import requests
import json
import csv
import urllib
import re
import io
import os
import time
import pythonwhois


# Database
import Database

from pymongo import MongoClient
import pymongo




def db_cleaner():
        d = Database.Database('domains')
        collection = d.return_collection("badDomains")
        i=0
        for domain in collection.find():
                dns_data = None
                try:
                        dns_data = domain['dns_data']
                except:
                        print("No dns data, delete?")

                if not domain['dns_data']['A'] and not domain['dns_data']['MX']:
                        collection.delete_one({'name': domain['name']})
                        #print("cleaned", i)
                        #input()
                        i=i+1

                       
        print(i)



if __name__ == "__main__":
        db_cleaner()
#d = Database.Database('domains')
#d.get_stats("goodDomains")
#d.get_stats("badDomains")


