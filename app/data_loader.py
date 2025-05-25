import pandas as pd
import tweepy
import os
import streamlit as st

# Получаем Bearer Token (или впиши напрямую)
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAKTr1wEAAAAA%2BbL5xUCHxfIeK%2BLBHVC4IZqQcm4%3DKEI3woOdwKTKK6ikEz2qd2mKKmvYj7HopPKvKc50b3mPAaR8DQ'

# Создаём клиент Tweepy v2
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

def load_tweets(query, max_tweets=100):
    tweets = []
    paginator = tweepy.Paginator(
        client.search_recent_tweets,
        query=query,
        tweet_fields=['created_at', 'text', 'lang'],
        max_results=100  # максимальный лимит для 1 запроса
    )

    # Добавляем прогресс-бар для Streamlit
    progress_bar = st.progress(0)
    loaded = 0

    for page in paginator:
        if page.data is not None:
            for tweet in page.data:
                tweets.append({'date': tweet.created_at, 'text': tweet.text})
                loaded += 1
                progress_bar.progress(min(loaded / max_tweets, 1.0))
                if loaded >= max_tweets:
                    progress_bar.empty()  # убираем прогресс-бар
                    return pd.DataFrame(tweets)
        else:
            st.warning("❗ Нет данных (page.data is None)")

    progress_bar.empty()
    return pd.DataFrame(tweets)
