#!/usr/bin/env python3

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

# Assuming your DataFrame is named df
df = pd.read_csv('customer_reviews.csv')

# Tokenization and lowercase
df['review'] = df['review'].apply(lambda x: word_tokenize(x.lower()))

# Remove stopwords in English and Spanish
stop_words_english = set(stopwords.words('english'))
stop_words_spanish = set(stopwords.words('spanish'))

def remove_stopwords(text):
    words = [word for word in text if word.isalnum() and word not in stop_words_english and word not in stop_words_spanish]
    return words

df['review'] = df['review'].apply(remove_stopwords)

# Create a list of words to exclude
exclude_words = ['servicio','service']

def remove_excluded_words(text):
    words = [word for word in text if word not in exclude_words]
    return words

df['review'] = df['review'].apply(remove_excluded_words)

# Calculate word frequencies
all_words = [word for review in df['review'] for word in review]
fdist = FreqDist(all_words)

# Most common words
common_words = fdist.most_common(10)
print(common_words)

