# import libraries
import sys
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
import pickle

from sklearn.metrics import classification_report, f1_score, precision_score
from sklearn.metrics import recall_score, accuracy_score


def load_data(database_filepath):
    '''Load the database and return a pandas dataframe.
    Input: database_filepath - string
    Output: X - numpy array os sentences,
            Y - targets - numpy array with categories,
            categories_names - categories to classify - list'''

    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(table_name='Messages_no_null_categories', con=engine)

    X = df.message.values
    categories_names = df.columns[3:].tolist()
    Y = df[categories_names].values

    return X, Y, categories_names

def tokenization(text):
    '''Tokenize the input text transforming in a normalized text
    Input: text - list of strings
    Output: clean_tokens - list of strings'''

    remove_other = RegexpTokenizer(r'\w+')
    tokens = remove_other.tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens

class remove_3_last_col(BaseEstimator, TransformerMixin):
    '''Remove the last 3 columns of a numpy array'''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, matrix):
        return matrix[:,0]

class remove_first_col(BaseEstimator, TransformerMixin):
    '''Remove the first column of the numpy array'''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, matrix):
        return np.array(matrix[:,1:],dtype='int')

class add_attributes(BaseEstimator, TransformerMixin):
    '''Data augmentation to recover 3 attributes of direct,news and social'''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, messages):
        model_att = joblib.load("../models/model_data_augment.pkl")
        attributes = model_att.predict(messages)
        return np.hstack((messages.reshape((len(messages),1)),attributes))

def build_model():
    '''Build a model with Pipeline with a CountVectorizer, TfIDF transformer \n
    and a SGD classifier.
    Input:
    Output: cv - a GridSearch machine learning model.
    '''

    pipeline = Pipeline([
    ('data_augmentation',add_attributes()),
    ('features', FeatureUnion([
        ('text_transform', Pipeline([
            ('remove_3', remove_3_last_col()),
            ('vect', CountVectorizer(tokenizer=tokenization)),
            ('tfidf',TfidfTransformer(smooth_idf=True)),
        ])),
        ('new_attributes', remove_first_col()),
    ])),

    ('clf', MultiOutputClassifier(SGDClassifier(max_iter=1000,random_state=123,
                                                warm_start=True,n_jobs=3,alpha=0.0001,
                                                power_t=0.25,class_weight='balanced',
                                                shuffle=True,
                                                fit_intercept=False,
                                                loss='modified_huber'))),
    ])

    parameters = {
        'clf__estimator__class_weight':[None,'balanced'],
        'clf__estimator__alpha': [0.001,0.0001],
        'clf__estimator__penalty': ['l2','elasticnet'],
    }

    #Preparing a grid search to output
    cv = GridSearchCV(pipeline,param_grid=parameters,scoring='accuracy',
                      refit=True)

    return cv


def evaluate_model(model, data_test, target_test, category_names):
    ''' Perform the report classification trought categories.
    Input: categories_names - list with name of categories
        data_test - dataframe with data to predict
        target_test - numpy array with labeled category
    Output: Print the results'''

    evaluation=[]
    prediction = model.predict(data_test)
    for col,category_name in enumerate(category_names):
        evaluation.append([category_name,
                          accuracy_score(prediction[:,col],target_test[:,col]),
                          precision_score(prediction[:,col],target_test[:,col]),
                          recall_score(prediction[:,col],target_test[:,col]),
                          f1_score(prediction[:,col],target_test[:,col])])

    print(pd.DataFrame(evaluation,columns=['Category','Accuracy','Precision',
                                           'Recall','F1_Score']))

def save_model(model, model_filepath):
    '''Save the machine learning model.
    Input: model - the model file
           model_filepath - the file path to save the model
    Output: '''
    
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=0.2,
                                                            shuffle=True,
                                                           random_state=123)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()