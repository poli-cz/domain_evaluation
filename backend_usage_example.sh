#!/bin/bash


# 1. get info about domain fit.vut.cz printed to STDOUT
cd src
python3 init.py fit.vut.cz --stdout



# 2. get only lexical info about domain google.com printed to STDOUT
python3 init.py google.com --lexical --stdout


