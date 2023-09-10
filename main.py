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

# Remove stopwords and punctuation
stop_words = set(stopwords.words('english'))
df['review'] = df['review'].apply(lambda x: [word for word in x if word.isalnum() and word not in stop_words])

# Calculate word frequencies
all_words = [word for review in df['review'] for word in review]
fdist = FreqDist(all_words)

# Most common words
common_words = fdist.most_common(10)
print(common_words)


