import streamlit as st
import pandas as pd
from datetime import datetime
from app import data_loader, preprocessing, sentiment_analyzer, trend_forecaster, visualizer

st.set_page_config(page_title="TrendScan", layout="wide")

analyzer = sentiment_analyzer.MultiLangSentimentAnalyzer()
forecaster = trend_forecaster.TrendForecaster()

st.title("📊 TrendScan — Анализ настроений и прогноз трендов")
st.markdown("Введите тему или загрузите CSV, чтобы увидеть динамику обсуждений и эмоциональную окраску.")

tab1, tab2 = st.tabs(["🔍 Анализ по теме", "📁 Загрузка CSV"])

with tab1:
    query = st.text_input("Введите ключевое слово или фразу", "искусственный интеллект")
    if st.button("Анализировать"):
        with st.spinner("Собираем данные..."):
        df = data_loader.load_tweets(query, max_tweets=300)
        df.columns = df.columns.str.lower()
        # Сначала чистим текст
        df["clean"] = df["text"].apply(preprocessing.clean_text)

        # Теперь лемматизируем "пакетом"
        df["clean"] = preprocessing.lemmatize_texts(df["clean"].tolist())

        sentiments = analyzer.batch_predict(df["clean"])
        sent_df = pd.DataFrame(sentiments)
        df_final = pd.concat([df, sent_df], axis=1)

        st.subheader("📊 Распределение тональностей")
        st.plotly_chart(visualizer.plot_sentiment_distribution(df_final), use_container_width=True)

        st.subheader("📈 Прогноз упоминаний")
        ts_data = forecaster.prepare_data(df)
        forecast = forecaster.fit_and_predict(ts_data)
        st.plotly_chart(visualizer.plot_forecast(forecast), use_container_width=True)

        st.download_button("📥 Скачать результат в CSV", df_final.to_csv(index=False), file_name="results.csv")

with tab2:
    uploaded_file = st.file_uploader("Загрузите CSV-файл с колонкой 'text'", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.lower()

        if "text" in df.columns:
            with st.spinner("🔄 Обработка текста..."):
                # Сначала чистим текст
            df["clean"] = df["text"].apply(preprocessing.clean_text)

                # Теперь лемматизируем "пакетом"
            df["clean"] = preprocessing.lemmatize_texts(df["clean"].tolist())
            sentiments = analyzer.batch_predict(df["clean"])
            sent_df = pd.DataFrame(sentiments)
            df_final = pd.concat([df, sent_df], axis=1)

            st.subheader("📊 Тональность загруженного файла")
            st.plotly_chart(visualizer.plot_sentiment_distribution(df_final), use_container_width=True)
            st.download_button("📥 Скачать результат в CSV", df_final.to_csv(index=False), file_name="results.csv")
        else:
            st.error("❌ Колонка 'text' не найдена! Проверьте загруженный CSV-файл.")
