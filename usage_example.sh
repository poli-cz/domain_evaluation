#!/bin/bash



# get info about fit.vut.cz printed to STDOUT
cd src
python3 init.py fit.vut.cz --stdout



# get only info about lexical analysis printed to STDOUT
python3 init.py google.com --lexical --stdout
