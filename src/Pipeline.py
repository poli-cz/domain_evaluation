
#import tensorflow
#from tensorflow import keras

from Data_loader import Base_parser
import SSL_loader
import Lex
import Database
from Lex import Net
import tensorflow as tf


print("[Log]: Starting modules...")

print("[Prompt]: Enter hostname:")
hostname = input()
hostname = str(hostname)
domain = Base_parser(hostname)

print("[Info]: Loading values...")
domain.load_dns_data()
domain.load_geo_info()
domain.load_whois_data()
domain.load_ssl_data()
    

dns_data = domain.get_dns()
geo_data = domain.get_geo_data()
whois_data = domain.get_whois_data()
ssl_data = domain.get_ssl_data()



if Lex.is_empty(dns_data) and Lex.is_empty(geo_data) and Lex.is_empty(whois_data):
        print("[Error]: Failed to harvers domain data")
        print("[Info]: Switching to lexical analysis AI")
        model = tf.saved_model.load('./domain_bigrams-furt-2020-11-07T11_09_21')
        input()



domain_data = {"name": hostname, "dns_data": dns_data, "geo_data": geo_data, "whois_data": whois_data, "ssl_data": ssl_data}

print("[Info]: Values loaded, starting AI")

prediction = Lex.process_data(domain_data)
print("[Info]: ", prediction)
prediction = float(prediction)

print("----------------------------------------------------------------")
if prediction < 0.5:
        accuracy = 1-prediction
        print("[Info]: Domain:", hostname, "is with:", accuracy*100, "% BAD")

else:
        accuracy = prediction
        print("[Info]: Domain:", hostname, "is with:", accuracy*100, "%", "GOOD")

print("----------------------------------------------------------------")

