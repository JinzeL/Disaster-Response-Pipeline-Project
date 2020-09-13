# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ 
    Load two raw data files.
  
    Parameters: 
    messages_filepath (str): path of messages data file
    categories_filepath (str): path of categories data file
    
    Returns: 
    DataFrame: a DataFrame combined two data files
  
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories, on = 'id')
    return df

def clean_data(df):
    """ 
    Clean the dataframe.
  
    Parameters: 
    df (DataFrame): the combined DataFrame of categories and messages
    
    Returns: 
    DataFrame: a DataFrame with messages and cleaned categories
  
    """    
    categories = df.categories.str.split(';',expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df.drop('categories',axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    # drop duplicates
    df = df.loc[~df.duplicated(),:]
    return df

def save_data(df, database_filename):
    """ 
    Save cleaned data into a sqlite database ('data_table' table).
  
    Parameters: 
    df (DataFrame): the cleaned DataFrame of categories and messages
    database_filename (str): path of the sqlite database saved
    
    Returns: 
    None
  
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('data_table', engine, index=False, if_exists='replace')


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