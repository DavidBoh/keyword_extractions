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
exclude_words = ['servicio', 'service']

def remove_excluded_words(text):
    words = [word for word in text if word not in exclude_words]
    return words

df['review'] = df['review'].apply(remove_excluded_words)

# Tokenize and find bigrams
def get_bigrams(text):
    return list(nltk.ngrams(text, 2))  # Tokenize into bigrams

df['bigrams'] = df['review'].apply(get_bigrams)

# Flatten the list of bigrams
all_bigrams = [bigram for review_bigrams in df['bigrams'] for bigram in review_bigrams]

# Calculate bigram frequencies
fdist = FreqDist(all_bigrams)

# Most common bigrams
common_bigrams = fdist.most_common(20)
print(common_bigrams)

