import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import multiprocessing

from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec

from time import time


def remove_hashtags_and_usernames(tweet):
    words = tweet.split()
    return ' '.join([word for word in words if word[0] not in '#@'])


def remove_non_aplhanumeric(tweet):
    return (re.sub('[\W_]+', ' ', tweet)).rstrip()


def remove_stop_words(tweet):
    words = tweet.split()
    return ' '.join([word for word in words if word not in stopwords.words('english')])


def stem(tweet):
    words = tweet.split()
    porter = PorterStemmer()
    return ' '.join([porter.stem(word) for word in words])


def remove_duplicates(tweet):
    words = tweet.split()
    new_words = []
    for word in words:
        if len(word) >= 3:
            if word[-3] == word[-2] == word[-1]:
                last_letter = None

                for i, letter in enumerate(word[::-1]):

                    if last_letter is None:
                        last_letter = letter
                        continue

                    if letter != last_letter:
                        words.append(word[:-(i - 1)])
                        break
                    last_letter = letter
    return ' '.join(words)


def clean_data(data):
    data['Tweet'] = data['Tweet'].apply(remove_hashtags_and_usernames)
    data['Tweet'] = data['Tweet'].apply(remove_non_aplhanumeric)
    data['Tweet'] = data['Tweet'].apply(remove_stop_words)
    data['len'] = data['Tweet'].apply(len)
    data['Tweet'] = data['Tweet'].apply(lambda x: x.lower())
    data['Tweet'] = data['Tweet'].apply(stem)
    data['Tweet'] = data[data['Tweet'].str.len() > 1]['Tweet']
    data['Tweet'] = data['Tweet'].apply(remove_duplicates)
    data['Tweet'] = data['Tweet'].apply(str.split)
    return data


def create_sentences(data):
    sent = [tweet for tweet in data.Tweet]
    phrases = Phrases(sent, min_count=1, progress_per=50000)
    bigram = Phraser(phrases)
    data.Tweet = data.Tweet.apply(lambda x: ' '.join(bigram[x]))



def main():
    # Get the data
    data = pd.read_csv(r'Mark 1/Testing/yelp.csv')
    print(data.head())
    print(data.info())
    print(data.describe())

    data['len'].hist(bins=50)


    data = clean_data(data)
    create_sentences(data)

    data.to_csv('cleaned_yelp.csv', index=False)

    plt.show()


if __name__ == '__main__':
    main()
