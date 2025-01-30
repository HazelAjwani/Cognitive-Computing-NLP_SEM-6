# -*- coding: utf-8 -*-
"""CCNLP_Lab-3_V1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oQAEJPzxV-DCnditae-5Cqa3RYhJv6DY
"""

!pip install nltk

import nltk
import string
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')

text = input("Enter the text for preprocessing: ")

normalized_text = text.lower()
print("\nNormalized Text:\n", normalized_text)

sentences = sent_tokenize(normalized_text)
print("\nSentence Tokenization:\n", sentences)

words = word_tokenize(normalized_text)
print("\nWord Tokenization:\n", words)

words_no_punct = [word for word in words if word not in string.punctuation]
print("\nPunctuation Removal:\n", words_no_punct)

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words_no_punct if word not in stop_words]
print("\nStopword Removal:\n", filtered_words)

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
print("\nStemming:\n", stemmed_words)

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
print("\nLemmatization:\n", lemmatized_words)

nltk.download('averaged_perceptron_tagger_eng')

pos_tags = pos_tag(words)
print("\nPOS Tagging:\n", pos_tags)

nltk.download('maxent_ne_chunker_tab')

ner_chunks = ne_chunk(pos_tags)
print("\nNamed Entity Recognition (NER):\n", ner_chunks)

vectorizer = CountVectorizer()

# Fit and transform the filtered text
bow_matrix = vectorizer.fit_transform([" ".join(filtered_words)])

# Display vocabulary (unique words and indices)
print("\nVocabulary (Word to Index Mapping):\n", vectorizer.vocabulary_)

# Convert sparse matrix to array
bow_array = bow_matrix.toarray()

#Display the bow matrix
print("\nBag of Words matrix:\n", bow_array)

# Convert BoW matrix to DataFrame for better readability
bow_df = pd.DataFrame(bow_array, columns=vectorizer.get_feature_names_out())
print("\nBag of Words Representation:\n", bow_df)