# Disaster Response

## Table of contents
1. [Motivation](#motivation)
2. [Dependencies](#dependencies)
3. [Files description](#files)
4. [Installing](#install)
5. [Executing](#execute)
6. [Licensing](#license)


<a name="motivation"></a>
## 1 - Motivation

Offer help as quickly as possible in an emergency to minimize damage and lost 
lives. For that, it is necessary to keep an eye on where the resources are 
needed and / or where they were offered.<br>
With this purpose in mind, the Udacity Data Science course has given us this 
task:<br>
Build a pipeline that can start with a message and obtain multi-class 
classification to help with crisis coordination.


<a name="dependencies"></a>
## 2 - Dependencies

Those scripts were tested in python 3.5 with following libraries:<br>

	Scikit Learn 0.24.1
	NLTK 3.6.1
	Numpy 1.19
	Pandas 1.1.5
	SQLAlchemy 1.4.7
	Flask 1.1.2
	Plotly 4.14.3
	Pickle 2.0
	joblib 1.0.0


<a name="files"></a>
## 3 - Files description

    * app
        * run.py - Start a webserver to interact with the classification results
        * template
            * master.html - Display the first page with graphs
            * go.html - Handle the input message and classification result
    * data
        * disaster_categories.csv - categories of messages
        * disaster_messages.csv - messages to classify
        * Messages_cleaned.db - SQL database after the ETL pipeline
        * process_data.py - python script to perform the ETL pipeline
    * models
        * model_data_augment.pkl - ML model to recover genre attributes from messages
        * model.pkl - ML model to classifiy the messages
        * pre_procesing_attributes.py - script to train a ML to recover genre attributes form messages
        * train_classifier.py - script to train a ML to classify disaster messages


<a name="install"></a>
## 4 - Installing

Clone the repository
   git clone https://github.com/edbroni/disaster_pipeline.git


<a name="execute"></a>
## 5 - Executing

Run the ETL pipeline inside data folder (extract and cleanning the messages)

    `$ python3 process_data.py disaster_messages.csv disaster_categories.csv Messages_cleaned.db`

Run the machine learning algorithm inside models folder (to train the model with the messages database)

    `$ python3 train_classify.py Messages_cleaned.db model.pkl`

Start a webserver to display the result of classification in action, inside app folder.

    `$ python3 run.py`

Type this address in a browser http://0.0.0.0:3001


<a name="license"></a>
## 6 - Licensing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<br>
This project is licensed under MIT License - see [License](LICENSE) for details