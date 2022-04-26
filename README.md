# Domain risk evaluation
System for evaluation risk of internet domains

### Usage
- Example usage can be found in: usage_example.sh
- Module init.py provides simple abstraction over clasifier modules
- Supported parameters for init.py are:
- --silent dont print status outputs
- --stdout print result to STDOUT otherwise results is printed to file: <domain>.<tld>.json
- --lexical get only lexical analysis
- --data_based get only data analysis
- --svm get only svm analysis

### Directories
- **./src** Main implementation
- **./mongo** Database backups and test data
- **./src/training**  Scripts for training models
- **./web_app/** Source codes for web implementation 


### Instalation
- pip3 install -r requirements.txt


### Web implementation

