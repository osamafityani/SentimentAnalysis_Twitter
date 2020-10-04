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
    # data['Tweet'] = data['Tweet'].apply(str.split)
    return data


def create_sentences(data):
    sent = [tweet for tweet in data.Tweet]
    phrases = Phrases(sent, min_count=1, progress_per=50000)
    bigram = Phraser(phrases)
    data.Tweet = data.Tweet.apply(lambda x: ' '.join(bigram[x]))
    return bigram[sent]


def main():
    # Get the data
    data = pd.read_csv(r'Tweets.csv')
    data = data[(data['stars'] == 1) | (data['stars'] == 5)]
    data['Tweet'] = data['text']
    print(data.head())
    print(data.info())
    print(data.describe())

    data['len'].hist(bins=50)


    data = clean_data(data)
    sentences = create_sentences(data)
    print(sentences[:5])

    w2v = Word2Vec(min_count=3,
                   window=4,
                   size=300,
                   alpha=0.03,
                   min_alpha=0.0007,
                   negative=20,
                   workers=multiprocessing.cpu_count()-1)
    start = time()
    w2v.build_vocab(sentences, progress_per=50000)
    print(f'Time to build vocab: {round((time() - start) / 60, 2)} mins')

    start = time()
    w2v.train(sentences, total_examples=w2v.corpus_count, epochs=30, report_delay=1)
    print(f'Time to train the model: {round((time() - start) / 60, 2)} mins')
    w2v.init_sims(replace=True)

    w2v.save('word2vec.model')
    data.to_csv('cleaned_tweets.csv', index=False)

    plt.show()


if __name__ == '__main__':
    main()
