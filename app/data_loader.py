import pandas as pd
import tweepy
import os

# Загрузим ключи API из переменных окружения
API_KEY = os.getenv('QS2VdB8ptH3acU43I2UsAkVsB')
API_SECRET = os.getenv('PFhXL1S1gL3RonFjqUthzbm5ZCbdjB4TVIttdAwEMHowHEOgvD')
ACCESS_TOKEN = os.getenv('1275803974474629120-MmibVe5tBorU01urr5sU6GlhuSeWX4')
ACCESS_TOKEN_SECRET = os.getenv('EAh2TFtuRiWRP99GMLIv94u2RtgVj2oTOnBHpYT6pIUbu')

# Настроим аутентификацию
auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

def load_tweets(query, max_tweets=100):
    tweets = []
    for tweet in tweepy.Cursor(api.search_tweets, q=query, lang='ru', tweet_mode='extended').items(max_tweets):
        tweets.append({'date': tweet.created_at, 'text': tweet.full_text})
    return pd.DataFrame(tweets)
