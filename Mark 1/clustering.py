import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

word_vectors = Word2Vec.load('word2vec.model').wv

kmeans = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50)
kmeans.fit(X=word_vectors.vectors)

print(word_vectors.similar_by_vector(kmeans.cluster_centers_[1], topn=10, restrict_vocab=None))

positive_cluster_center = kmeans.cluster_centers_[1]
negative_cluster_center = kmeans.cluster_centers_[0]

words = pd.DataFrame(word_vectors.vocab.keys())
words.columns = ['words']


def cast_vector(row):
    return np.array(list(map(lambda x: x.astype(np.double), row)))


words['vectors'] = words['words'].apply(lambda x: word_vectors.wv[f'{x}'])
words['cluster'] = words['vectors'].apply(lambda x: kmeans.predict([np.array(x, dtype='double')]))
words['cluster'] = words['cluster'].apply(lambda x: x[0])

words['cluster_value'] = [1 if cluster == 1 else -1 for cluster in words['cluster']]
words['closeness_score'] = words.apply(lambda x: 1/(kmeans.transform([x.vectors]).min()), axis=1)
words['sentiment_coff'] = words['closeness_score'] * words['cluster_value']

words.head()
words[['words', 'sentiment_coff']].to_csv('sentiment_dictionary.csv', index=False)
