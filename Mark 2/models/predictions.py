import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_transform(weights):
    tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(), norm=None)
    tfidf.fit(weights.Tweet)
    features = pd.Series(tfidf.get_feature_names())
    transformed = tfidf.transform(weights.Tweet)
    return features, transformed


def read_list_of_words(filename):
    file = open(filename, 'r')
    words = file.readlines()
    return [word[:-1] for word in words]


def create_tfidf_dictionary(x, transformed_file, features):
    vector_coo = transformed_file[x.name].tocoo()
    vector_coo.col = features.iloc[vector_coo.col].values
    dict_from_coo = dict(zip(vector_coo.col, vector_coo.data))
    return dict_from_coo


def replace_tfidf_words(x, transformed_file, features):
    dictionary = create_tfidf_dictionary(x, transformed_file, features)
    return list(map(lambda y: dictionary[f'{y}'], x.Tweet.split()))


def replace_sentiment_words(word, sentiment_dict):
    try:
        out = sentiment_dict[word]
    except KeyError:
        out = 0
    return out


def get_sentiment(sentiment_rate):
    if sentiment_rate > 0:
        return 1
    elif sentiment_rate == 0:
        return 0
    else:
        return -1


def make_predictions(weights, sentiment_dict, tfidf_words):
    replaced_closeness_scores = weights.Tweet.apply(lambda x: list(map(lambda y: replace_sentiment_words(y, sentiment_dict), x.split())))
    replacement_df = pd.DataFrame(data=[replaced_closeness_scores, tfidf_words, weights.Tweet]).T
    replacement_df.columns = ['sentiment_coeff', 'tfidf_scores', 'sentence']
    replacement_df['sentiment_rate'] = replacement_df.apply(lambda x: np.array(x.loc['sentiment_coeff']) @ np.array(x.loc['tfidf_scores']), axis=1)
    replacement_df['prediction'] = replacement_df['sentiment_rate'].apply(get_sentiment)
    return replacement_df

def main():
    data = pd.read_csv(r'cleaned_tweets.csv')
    weights = data.copy()
    features, transformed = tfidf_transform(weights)

    positive_words = read_list_of_words('../Positive.txt')
    positive_dict = dict(zip(positive_words, np.ones(len(positive_words))))

    negative_words = read_list_of_words('../Negative.txt')
    negative_dict = dict(zip(negative_words, -1 * np.ones(len(positive_words))))

    sentiment_dict = {**negative_dict, **positive_dict}

    replaced_tfidf_words = weights.apply(lambda x: replace_tfidf_words(x, transformed, features), axis=1)

    final_df = make_predictions(weights, sentiment_dict, replaced_tfidf_words)

    final_df[['sentence', 'sentiment_rate', 'prediction']].to_csv('predictions.csv', index=False)



if __name__ == '__main__':
    main()