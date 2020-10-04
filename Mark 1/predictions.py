import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv(r'cleaned_tweets.csv')
sentiment_map = pd.read_csv('sentiment_dictionary.csv')
sentiment_dict = dict(zip(sentiment_map.words.values, sentiment_map.sentiment_coff.values))

weights = data.copy()
tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(), norm=None)
tfidf.fit(weights.Tweet)
features = pd.Series(tfidf.get_feature_names())
transformed = tfidf.transform(weights.Tweet)


def create_tfidf_dictionary(x, transformed_file, features):
    vector_coo = transformed_file[x.name].tocoo()
    vector_coo.col = features.iloc[vector_coo.col].values
    dict_from_coo = dict(zip(vector_coo.col, vector_coo.data))
    return dict_from_coo


def replace_tfidf_words(x, transformed_file, features):
    dictionary = create_tfidf_dictionary(x, transformed_file, features)
    return list(map(lambda y: dictionary[f'{y}'], x.Tweet.split()))


replaced_tfidf_words = weights.apply(lambda x: replace_tfidf_words(x, transformed, features), axis=1)


def replace_sentiment_words(word, sentiment_dict):
    try:
        out = sentiment_dict[word]
    except KeyError:
        out = 0
    return out


replaced_closeness_scores = weights.Tweet.apply(lambda x: list(map(lambda y: replace_sentiment_words(y, sentiment_dict), x.split())))
replacement_df = pd.DataFrame(data=[replaced_closeness_scores, replaced_tfidf_words, weights.Tweet]).T
replacement_df.columns = ['sentiment_coeff', 'tfidf_scores', 'sentence']
replacement_df['sentiment_rate'] = replacement_df.apply(lambda x: np.array(x.loc['sentiment_coeff']) @ np.array(x.loc['tfidf_scores']), axis=1)
replacement_df['prediction'] = (replacement_df.sentiment_rate > 0).astype('int8')

data['stars'] = data['stars'].apply(lambda x: 1 if x == 5 else 0)

from sklearn.metrics import classification_report

print(classification_report(data['stars'], replacement_df['prediction']))

replacement_df[['sentence', 'sentiment_rate', 'prediction']].to_csv('predictions.csv', index=False)

