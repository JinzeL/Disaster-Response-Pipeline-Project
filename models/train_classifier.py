import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import sklearn.externals 
import joblib 
from nltk.corpus import stopwords
import string

def load_data(database_filepath):
    """ 
    Load data from database.
  
    Parameters: 
    database_filepath (str): path of database
    
    Returns: 
    X (DataFrame): a DataFrame contains input data
    Y (DataFrame): a DataFrame contains target data
    cols (Array): an array of categories names
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('data_table', con = engine)
    X = df['message']
    Y = df.iloc[:,4:]
    Y.related.replace(2,1,inplace=True)
    cols = list(Y.columns)
    return X, Y, cols

def tokenize(text):
    """ 
    Tokenize the input text
  
    Parameters: 
    text (str): input text
    
    Returns: 
    clean_tokens (list): list of tokenized text
    """
    tokens = word_tokenize(text)
    # remove all tokens that are not alphabetic
    words = [word for word in tokens if word.isalpha()]
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    #stop_words = set(stopwords.words('english'))
    #words = [w for w in words if not w in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """ 
    Build a model through a pipeline
  
    Parameters: 
    None
    
    Returns: 
    cv (model): a trained model
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                 ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__max_df': (0.5, 1.0),
    'vect__max_features': (None, 5000),
    'tfidf__use_idf': (True, False),
    'clf__estimator__n_estimators': [10, 20] }
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, n_jobs=-1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """ 
    Print the classification report for each category
  
    Parameters: 
    model (model): a trained model
    X_test (panda series): input message data
    Y_test (DataFrame): the real categories data
    category_names (list): a list of categories' name
    
    Returns: 
    None
    """
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print('Classification Report for ' + category_names[i])
        print(classification_report(np.array(Y_test)[:,i],y_pred[:,i]))

def save_model(model, model_filepath):
    """ 
    Save the model as a pickle in a file
  
    Parameters: 
    model (model): a trained model
    model_filepath (str): pickle file path

    Returns: 
    None
    """
    joblib.dump(model, model_filepath) 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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