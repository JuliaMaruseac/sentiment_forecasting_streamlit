import pandas as pd
import snscrape.modules.twitter as sntwitter

def load_tweets(query, max_tweets=1000):
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= max_tweets:
            break
        tweets.append({'date': tweet.date, 'text': tweet.content})
    return pd.DataFrame(tweets)
