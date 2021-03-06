# import libraries
import sys
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
# pd.set_option('max_colwidth',200)
pd.set_option('expand_frame_repr', False)

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
        messages_filepath - (string) a specific file path string for loading messages data
        categories_filepath - (string) a specific file path string for loading categories data
    
    OUTPUT:
        df - (pandas dataframe) a dataframe merged with messages and categories

    Description:
    This function loads datasets (i.e., messages and categories) from local disk

    '''
    
    # load messages, categories dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, how='left', on=['id'])

    return df

def convert_categorial_variables(value):

    # convert category values to just numbers 0 or 1
    value = value.split('-')[1]

    # convert column from string to numeric
    value = int(value)
    
    # convert any values different than 0 and 1 to 0 (False)
    if (value not in [0, 1]):
        value = 0
        
    return value

def clean_data(df):
    ''' 
    INPUT:
    df - (pandas dataframe) a raw dataframe
    
    OUTPUT:
    df - (pandas dataframe) a cleansed dataframe

    Description:
    This function cleans dataframe with various approaches

    '''

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x.split('-')[0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # iterate through the category columns in df to keep only the last character of each string (the 1 or 0)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda value: convert_categorial_variables(value))
        
        # # convert column from string to numeric
        # categories[column] = categories[column].astype("int")

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates(keep='last')

    return df


def save_data(df, database_filename):
    '''
    INPUT:
    df - (pandas dataframe) a cleansed dataframe
    database_filename - (string) a specific string for naming database and table
    
    OUTPUT:
    None

    Description:
    This function saves dataset into SQLite database

    '''

    # get database/table name
    name = os.path.basename(database_filename).split('.')[0]

    # create sqlite connection
    engine = create_engine('sqlite:///{}'.format(database_filename))

    # write data into database
    df.to_sql(name, engine, index=False,if_exists='replace') 


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