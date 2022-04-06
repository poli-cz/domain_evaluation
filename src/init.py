""" File: init.py
    Author: Jan Polisensky
    ----
    Example usage of classifier core module
"""


# Import basic modules and libraries
import json
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import argparse


# Load custom modules
import Database
from Data_loader import Base_parser
import SSL_loader
import Lex
from Lex import Net
from Preprocessor import preprocess 
from Core import clasifier


class resolver:
    """
    Class for controling core module 
    ...
    Attributes
    ----------
    domain_name : str
        string containing domain name  

    """

    def __init__(self, domain_name) -> None:
        self.cls = clasifier()
        self.domain_name = domain_name

    # Get combined prediction, details can be found in REDME or documentation
    # Returns -> dictionary containg prediction of domain badnes
    def get_combined(self) -> dict:
        lexical = self.cls.get_lexical(self.domain_name)
        data_based = self.cls.get_data(self.domain_name)
        svm = self.cls.get_svm(self.domain_name)

        combined, accuracy = self.cls.get_mixed(self.domain_name) 


        combined = np.around(combined, 3)
        accuracy = np.around(accuracy, 3)

        svm = np.around(svm, 3)
        lexical = np.around(lexical, 3)
        data_based = np.around(data_based, 3)

        rating ={
            "domain_name" : self.domain_name,
            "lexical" : lexical,
            "data-based" : data_based,
            "svm" : svm,
            "combined": combined,
            "accuracy": accuracy
        }
        return rating

    # Get prediction based only on lexical features(does not need to fetch any data)
    # Returns -> dictionary containg prediction of domain badnes
    def get_lexical(self) -> dict:
        lexical = self.cls.get_lexical(self.domain_name)
        lexical = np.around(lexical, 3)
        rating = {
            "domain_name" : self.domain_name,
            "lexical" : lexical
        }

        return rating

    # Get prediction from SVM model
    # Returns -> dictionary containg prediction of domain badnes
    def get_svm(self) -> dict:
        svm = self.cls.get_svm(self.domain_name)
        svm = np.around(svm, 3)
        rating = {
            "domain_name" : self.domain_name,
            "support-vector-machines" : svm
        }

        return rating

    # Get prediction based on main Data model
    # Returns -> dictionary containg prediction of domain badnes
    def get_data(self) -> dict:
        data_based = self.cls.get_data(self.domain_name)
        data_based = np.around(data_based, 3)
        rating = {
            "domain_name" : self.domain_name,
            "data-based_prediction" : data_based
        }

        return rating

    # Param data: JSON object to be printed to output file
    def output_json(self, data) -> None:
        rating_json = json.dumps(data, indent = 4)

        with open(self.domain_name + '.json', "w") as outfile:
            outfile.write(rating_json)

    # Param data: JSON object to be printed to STDOUT
    def output_stdout(self, data) -> None:
        print(json.dumps(data, indent = 4))
        



if __name__ == "__main__":
    ### Supported arguments ###
    parser = argparse.ArgumentParser(description='domain name analysis tool')
    parser.add_argument('domain_name', type=str, help='Required domain name')
    parser.add_argument('--lexical', action='store_true', help='Use only lexical model for classification')
    parser.add_argument('--data_based', action='store_true', help='Use only data-based model for classification')
    parser.add_argument('--svm', action='store_true', help='Use only svm model for classification')
    parser.add_argument('--silent', action='store_true', help='No output')
    parser.add_argument('--stdout', action='store_true', help='Output to stdout instead of file')


    ### Parse arguments ###
    args = parser.parse_args()
    domain_name = args.domain_name
    m_lexical = args.lexical
    m_data_based = args.data_based
    m_svm = args.svm
    m_silent = args.silent
    m_stdout = args.stdout

    if [m_lexical, m_data_based, m_svm].count(True) > 1:
        print("[Error]: Can use only one classification mode")
        exit(1)

    if m_silent:

        f = open(os.devnull, 'w')
        sys.stdout = f
    
    ### Starting main resolver ###
    res = resolver(domain_name)

    if m_lexical:
        r_data = res.get_lexical()

    elif m_data_based:
        r_data = res.get_data()

    elif m_svm:
        r_data = res.get_svm()
    
    else:
        r_data = res.get_combined()

    if not m_stdout:
        res.output_json(r_data)
    else:
        res.output_stdout(r_data)




