import csv
import string
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))

def clean_term(text):
    text = text.lower()
    return "".join(char for char in text
                   if char not in string.punctuation)


def standardize(text):
    stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    nltk_tokens = nltk.word_tokenize(text)
    result = ''
    for w in nltk_tokens:
        if w not in stop_words:
            text = clean_term(w)
            if not text.isdigit():
                result = result + ' ' + stemmer.stem(wordnet_lemmatizer.lemmatize(text))
    return result

# satd_comment = '// JUnit 4 wraps solo tests this way. We can extract // the original test name with a little hack.'

# ans = standardize(satd_comment)