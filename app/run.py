import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objects import Bar, Sunburst
from sklearn.externals import joblib
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin


app = Flask(__name__)

class DisasterWordExtractor(BaseEstimator, TransformerMixin):

    """
        A custom transformer which will identify buzzwords signaling disaster
    """

    def disaster_words(self, text):
        """
        INPUT: text - string, raw text data
        OUTPUT: bool -bool object, True or False
        """
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


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
database_filepath = 'data/DisasterResponse.db'
table_name = 'DisasterResponse'

engine = create_engine('sqlite:///{}'.format(database_filepath))
df = pd.read_sql_table(table_name, engine)  

# load model
model_filepath = 'models/classifier.pkl'
model = joblib.load(model_filepath)

############################################################
#          Pre-preparation for the visual 1                #
############################################################
def generate_data_for_visual1(df):
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    return (genre_counts, genre_names)

############################################################
#          Pre-preparation for the visual 2                #
############################################################
def generate_data_for_visual2(df):
    
    category_counts = df[df.columns[4:]].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)

    return (category_counts, category_names)

############################################################
#          Pre-preparation for the visual 3                #
############################################################
def convert_category(row):
    
    name = row['category'] + "_" + row['genre']
    
    return name

def generate_data_for_visual3(original_df):

    df = original_df[original_df.columns[3:]]
    df = (pd
        .pivot_table(df,
            values=df.columns[1:], 
            index=['genre'], 
            aggfunc=np.sum, 
            fill_value=0)
        .stack()
        .reset_index()
    )

    df.columns = ['genre', 'category', 'value']

    df = (df
        .groupby('genre')
        .apply(lambda x: x.sort_values(["value"], ascending = False))
        .reset_index(drop=True)
        .groupby('genre')
        .head(10)
    )

    genre_counts = original_df.groupby('genre').count()['message']
    genre_total  = genre_counts.sum()

    df['category_rename'] = df.apply(lambda row: convert_category(row), axis=1)

    characters = np.concatenate((np.array(['total', 'direct', 'news', 'social']), df['category_rename'].values), axis=0)
    parents    = np.concatenate((np.array(['', 'total', 'total', 'total']), df['genre'].values), axis=0)
    values     = np.concatenate((np.array([genre_total]), genre_counts, df['value'].values), axis=0)

    return (characters, parents, values)

############################################################
#                   Webpage Rendering                      #
############################################################
# index webpage displays cool visuals and receives user input text for model => display the visuals
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # visual 1 preparation - distribution of message genres
    genre_counts, genre_names = generate_data_for_visual1(df)

    # visual 2 preparation - distribution of message category
    category_counts, category_names = generate_data_for_visual2(df)

    # visual 3 preparation - top 10 category distribution in each genre
    characters, parents, values = generate_data_for_visual3(df)
    
    # create visuals
    graphs = [
        # visual 1 configuration
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }, 
        # visual 2 configuration
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'tickangle': -30
                },
                'margin': {
                    'b': 100, 
                },
                'height': 600
            }
        },
        # visual 3 configuration
        {
            'data': [
                Sunburst(
                    labels=characters,
                    parents=parents,
                    values=values,
                )
            ],

            'layout': {
                'title': 'Top 10 Category Distribution in each Genre',
                'margin': {
                    't': 70, 
                    'l': 0, 
                    'r': 0, 
                    'b': 0
                },
                'height': 700
            }

        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results => make predictions on text classification
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()