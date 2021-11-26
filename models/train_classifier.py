# import libraries
import sys
import os
import re
import numpy as np
import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
# pd.set_option('max_colwidth',200)
pd.set_option('expand_frame_repr', False)

from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
nltk.download(['punkt', 'wordnet', 'stopwords'])

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score

import pickle

def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - (string) a specific file path string for target database

    OUTPUT:
    X - (pandas Series) features dataframe
    y - (pandas dataframe) target dataframe
    category_names - (numpy array) categorical variables' labels

    Description:
    A function used to load the clean data from the SQLite database

    '''

    # get database/table name
    name = os.path.basename(database_filepath).split('.')[0]

    # load data from database
    engine = create_engine('sqlite:///{}.db'.format(name))

    df = pd.read_sql_table(name, engine)  

    X = df['message']
    y = df.iloc[:, 4:]

    return X, y, y.columns


def tokenize(text):
    '''    
    INPUT:
    text - raw text data
    
    OUTPUT: 
    clean_tokens - (list) a list of clean words in their roots form

    Description:
    The process of raw texts tokenization, which includes:
        1. replace any urls with the string 'urlplaceholder'
        2. remove punctuation
        3. tokenize texts
        4. remove stop words
        5. lemmatize and normalize (lowercase, strip space) texts

    '''
    
    # 1. remove url and replace url string as 'urlplaceholder'
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # 2. remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)

    # 3. tokenize words
    tokens = word_tokenize(text)

    # 4. remove stop words
    tokens = [tok for tok in tokens if tok not in stopwords.words("english")]

    # create lemmatization (method)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        
        # 5. lemmatizing, converting lowercase and removing space in tokens
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class DisasterWordExtractor(BaseEstimator, TransformerMixin):

    """
        A custom transformer which will identify buzzwords signaling disaster
    """

    def disaster_words(self, text):
        '''
        INPUT: 
        text - raw text data

        OUTPUT: 
        bool - (bool) True or False

        Description:
        This function is used to do extra disaster words detection

        '''

        # list of words that are commonly used during a disaster event
        disaster_words = ['tsunami','volcano','tornado','avalanche','earthquake','blizzard','drought','bushfire',
                          'tremor','duststorm','magma','twister','windstorm','heat','cyclone','fire','flood',
                          'hailstorm','lava','lightning','high-pressure','hail','hurricane','seismic','erosion',
                          'whirlpool','Richter scale','whirlwind','cloud','thunderstorm','barometer',
                          'gale','blackout','gust','force','low-pressure','volt','snowstorm','rainstorm','storm',
                          'nimbus','violentstorm','sandstorm','casualty','fatal','fatality','cumulonimbus','death',
                          'lost','destruction','money','tension','cataclysm','damage','uproot','underground',
                          'destroy','arsonist','arson','rescue','permafrost','disaster','fault','scientist','shelter']

        # lemmatize the buzzwords
        lemmatized_words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in disaster_words]
        # Get the stem words of each word in lemmatized_words
        stem_disaster_words = [PorterStemmer().stem(w) for w in lemmatized_words]

        # tokenize the input text
        clean_tokens = tokenize(text)
        for token in clean_tokens:
            if token in stem_disaster_words:
                return True
        return False

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X_disaster_words = pd.Series(X).apply(self.disaster_words)
        return pd.DataFrame(X_disaster_words)


def grid_search_best_model(pipeline):
    '''
    INPUT:
    pipeline - an object generated from pipeline

    OUTPUT:
    cv - a best model with optimsed parameters generated by Grid Search method

    Description:
    A function used to apply grid search method to find the best model.

    '''

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=500, num=5)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(start=5, stop=50, num=10)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'clf__estimator__n_estimators': n_estimators,
                   'clf__estimator__max_features': max_features,
                   'clf__estimator__max_depth': max_depth,
                   'clf__estimator__min_samples_split': min_samples_split,
                   'clf__estimator__min_samples_leaf': min_samples_leaf,
                   'clf__estimator__bootstrap': bootstrap}

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=random_grid, cv=3, verbose=3)

    return cv

def build_model():
    '''
    INPUT:
    None

    OUTPUT:
    model - a model generated from pipeline

    Description:
    A modelling pipeline that includes text processing steps and 
    a classifier (random forest) used for classifying observations.

    '''

    # create ML pipeline
    pipeline = Pipeline([
    ('features',FeatureUnion([
        ('text_pipeline',Pipeline([
            ('vect',CountVectorizer(tokenizer=tokenize)),
            ('tfidf',TfidfTransformer())
            ])),
        ('disaster_words',DisasterWordExtractor()) # custom transformer - disaster word extractor
        ])),
    ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])

    # get the best model with grid search cv
    # model = grid_search_best_model(pipeline)

    # Note: due to lack of computation resources, grid serach function has not been applied to find the best model
    model = pipeline

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model - the classification model with optimised parameters
    X_test - (pandas Series) feature variable from test set
    y_test - (pandas dataframe) target variables from test set
    category_names - (numpy array) a list of text category name
    
    OUTPUT:
    Classification report and accuracy score

    Description:
    This function evaluates the model performance for each category

    '''

    y_true = Y_test.values
    # predict on test data
    y_pred = model.predict(X_test)

    # f1-score, precision, recall, support...
    print ('Classification Report on Test set: \n')
    print (classification_report(y_true, y_pred, target_names=category_names))

    # accuracy
    print ('Accuracy on Test set: \n')
    print ((y_pred == y_true).mean())


def save_model(model, model_filepath):
    # save the model to disk
    pickle.dump(model, open(model_filepath,'wb'))


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