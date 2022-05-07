# Domain risk evaluation
AI based system for evaluating risk of domain names. This repository includes implementation, testing datasets and web implementation of system. System uses combination of **neural networks** and **support vector machines** to determine danger asociated with domain name. System can be used to perform detection of **DGA**, **Phishing sites**, **Malware sites** and more. 
Test app is available at **[Urlcheck](https://urlcheck.eu/)**
 

 ### Instalation
Backend of application is made in python3, web app in [nodejs]( https://nodejs.org/en/).
Required **python 3.8.10** or higher.
Required **node 14.18** or higher.
Requirements can be installed with commands bellow.
```console
poli@poli:~$ pip3 install -r requirements.txt
poli@poli:~$ bash init_web_app.sh
```

### Usage
Backend usage
- Before first run **.env file needs to be configures** in /src/ (from env.example)
- Example usage can be found in: **backend_usage_example.sh**
- Module **init.py** provides simple abstraction over clasifier modules
- Supported parameters for **init.py** are:
- --silent dont print status outputs
- --stdout print result to STDOUT otherwise results is printed to file: domain.tld.json
- --lexical get only lexical analysis
- --data_based get only data analysis
- --svm get only svm analysis

Frontend usage:
 - After instalation simply run:
    ```console
    poli@poli:~$ cd ./web_app && node app.js
    ```
 - Test app will be available at port 4444 (or you can define custom port in app.js)


### Directories
- **./src** Main backend
- **./mongo** Database backups and test data
- **./models** Trained models for classification
- **./src/training**  Scripts for training models
- **./web_app/** Source codes for web implementation .
- **./web_app/sites** Cached sites that have been resolved.

### Database
For models training or validation [MongoDB](https://www.mongodb.com/) is required. Database importing can be done by running following command:
```console
poli@poli:~$ sudo systemctl status mongod #Ensure that mongoDB is running
poli@poli:~$ mongorestore -d domains mongo
```

