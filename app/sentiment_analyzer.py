from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from langdetect import detect

class MultiLangSentimentAnalyzer:
    def __init__(self):
        self.ru_model = AutoModelForSequenceClassification.from_pretrained("blanchefort/rubert-base-cased-sentiment")
        self.ru_tokenizer = AutoTokenizer.from_pretrained("blanchefort/rubert-base-cased-sentiment")
        self.en_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.en_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

    def detect_language(self, text):
        try:
            return detect(text)
        except:
            return "unknown"

    def predict(self, text):
        lang = self.detect_language(text)
        if lang == "ru":
            tokenizer, model = self.ru_tokenizer, self.ru_model
        else:
            tokenizer, model = self.en_tokenizer, self.en_model
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).squeeze().tolist()
        labels = ['negative', 'neutral', 'positive']
        return dict(zip(labels, probs))

    def batch_predict(self, texts):
        return [self.predict(text) for text in texts]
