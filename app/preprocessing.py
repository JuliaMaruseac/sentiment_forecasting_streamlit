import re
import nltk
import spacy
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
STOPWORDS = set(stopwords.words('russian'))
try:
    nlp = spacy.load('ru_core_news_sm')
except OSError:
    from spacy.cli import download
    download('ru_core_news_sm')
    nlp = spacy.load('ru_core_news_sm')


def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.text not in STOPWORDS and token.is_alpha])
