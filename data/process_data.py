import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''Load both message and categories databases
    Input: messages_filepath - messages file location - string
           categories_filepath - categories file location - string
    Output: messages - messages - dataframe
            categories - categories - dataframe.
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath, sep=',')

    #Recording the categories names
    categories_columns=[]
    [categories_columns.append(item.split('-')[0]) for item in categories.categories.str.split(pat=';')[0]]

    #Spliting the data from categories names
    categories_data = categories.categories.str.split(pat=';')
    data=[]
    for row_data in categories_data:
        temp_row = []
        for categorie in row_data:
            temp_row.append(categorie.split('-')[1])
        data.append(temp_row)

    # merge datasets
    df = messages.set_index('id').join(categories.set_index('id'))

    return df


def clean_data(df):
    '''Clean the dataframe according the pre defined steps.
    Input: df - data pandas dataframe.
    Output: df - data - cleaned pandas dataframe.
    '''

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = []
    [category_colnames.append(item.split('-')[0]) for item in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(['categories'],axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories)

    # there are fields in related with number 2, let's change to 1
    df.loc[df.related==2, ['related']] = 1

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    '''Save a dataframe to a SQL database
    Input: df - dataframe
           database_filename - name of databse file - string
    Output:
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Messages', engine, index=False)
    print("Saved database Messages")

    #Identify category without occurence
    remove=[]
    for category_name in df.select_dtypes(include=['int']).columns:
        if df[category_name].sum() == 0:
            remove.append(category_name)
            print('Empty category removed: ',category_name)

    #Removing the category(ies) found
    df_new = df.drop(remove,axis=1)

    #Saving the database in other other datasheet
    df_new.to_sql('Messages_no_null_categories', engine, index=False)
    print("Saved database Messages_no_null_categories")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()