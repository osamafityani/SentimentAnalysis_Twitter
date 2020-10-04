import requests
import pandas as pd
import numpy as np
import yaml
import time

def create_twitter_url(query, next_token=None):
    max_results = 100
    tweet_fields = "id,text,created_at,public_metrics,source"
    mrf = f"max_results={max_results}"
    q = f"query={query}"
    tf = f"tweet.fields={tweet_fields}"
    if next_token is None:
        url = f"https://api.twitter.com/2/tweets/search/recent?{mrf}&{q}&{tf}"
    else:
        nt = f"next_token={next_token}"
        url = f"https://api.twitter.com/2/tweets/search/recent?{mrf}&{q}&{tf}&{nt}"
    return url


def process_yaml():
    with open("config.yaml") as file:
        return yaml.safe_load(file)


def create_bearer_token(data):
    return data["search_tweets_api"]["bearer_token"]


def twitter_auth_and_connect(bearer_token, url):
    headers = {"Authorization": f"Bearer {bearer_token}"}
    return headers


def get_tweets(url, headers):
    tweets = []
    responce = requests.request("GET", url, headers=headers)
    # We can use this if we want to write the data to a file...But it's not working... Something is wrong and I don't know what it is
    # with open('Tweets.json', 'a') as file:
    #     file.write(responce)
    print(responce.json())
    for tweet in responce.json()["data"]:
        tweets.append(tweet)
    try:
        nt = responce.json()["meta"]["next_token"]
        return tweets, nt
    except():
        return tweets, None


def create_update_dataframe(tweets, df=None):
    temp = pd.DataFrame([tweet["text"] for tweet in tweets], columns=['Tweet'])
    temp['len'] = np.array([len(tweet["text"]) for tweet in tweets])
    temp['metrics'] = np.array([tweet["public_metrics"] for tweet in tweets])
    temp['source'] = np.array([tweet["source"] for tweet in tweets])
    if df is None:
        return temp
    else:
        return df.append(temp, ignore_index=True)


def main():
    nt = None
    df = None
    query = "-is:retweet lang:en (love OR hate OR feel OR feeling OR great OR bad OR better OR think OR opinion OR really OR annoyed OR angry)"  # Set rules here
    data = process_yaml()
    bearer_token = create_bearer_token(data)
    i = 0
    j = 0
    while (i == 0) or (nt is not None):
        url = create_twitter_url(query, next_token=nt)
        headers = twitter_auth_and_connect(bearer_token, url)
        tweets, nt = get_tweets(url, headers=headers)
        df = create_update_dataframe(tweets, df)

        if j >= 3:
            time.sleep(16*60)
            j = -1
        if i >= 100 or (nt is None):
            break
        i += 1
        j += 1
    print(df.head())
    df.to_csv('Tweets.csv')


if __name__ == '__main__':
    main()
