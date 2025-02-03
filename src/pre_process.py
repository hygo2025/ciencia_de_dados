import re

import nltk
import unidecode
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


import warnings
warnings.filterwarnings("ignore")


class PreProccessText:
    def __init__(self):
        # TODO: Verificar se é necessário fazer o download dos pacotes do nltk
        # nltk.download('punkt')
        # nltk.download('stopwords')
        # nltk.download('wordnet')
        # nltk.download('punkt_tab')
        pass

    def run(self, text: str) -> str:
        cleaned_text = self.lower_clean(text)
        unidecoded_text = self.unidecode_text(cleaned_text)
        tokens = self.tokenize(unidecoded_text)
        tokens_no_stopwords = self.remove_stopwords(tokens)
        lemmatized_tokens = self.lemmatize(tokens_no_stopwords)  # testar stemming(tokens_no_stopwords)
        return ' '.join(lemmatized_tokens)

    def lower_clean(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)  # Apenas letras e espaços
        return text

    def unidecode_text(self, text):
        return unidecode.unidecode(text)

    def tokenize(self, text: str) -> list:
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list) -> list:
        stop_words = set(stopwords.words('english'))
        return [word for word in tokens if word not in stop_words]

    def lemmatize(self, tokens: list) -> list:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in tokens]

    def stemming(self, tokens: list) -> list:
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in tokens]
