#!/bin/bash
python -m spacy download ru_core_news_sm
streamlit run streamlit_app.py --server.port $PORT --server.enableCORS false
