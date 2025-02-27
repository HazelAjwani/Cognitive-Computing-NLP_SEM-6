# -*- coding: utf-8 -*-
"""Lab_1 and 2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PE0eF5_rRLHNqknNA2z89Mh_Cfri2F97
"""

!pip install nltk

import nltk
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')

"""Text Input"""

text=input('enter the text: ')

"""Text Normalization"""

normalized_text = text.lower()
print("\nNormalized Text:")
print(normalized_text)

"""Sentence Tokenization"""

sentences=sent_tokenize(normalized_text)
print('\nSentence Tokenization: ')
print(sentences)

"""Word Tokenization"""

words = word_tokenize(normalized_text)
print("\nWord Tokenization:")
print(words)

"""Punctuation Removal"""

words_no_punct = [word for word in words if word not in string.punctuation]
print("\nPunctuation Removal:")
print(words_no_punct)

"""Stemming"""

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in words_no_punct]
print("\nStemming:")
print(stemmed_words)

"""Lemmetization"""

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in words_no_punct]
print("\nLemmatization:")
print(lemmatized_words)

"""POS Tagging"""

nltk.download('averaged_perceptron_tagger_eng')

pos_tags = pos_tag(words_no_punct)
print("\nPOS Tagging:")
print(pos_tags)

"""Named Entity Recognition"""

nltk.download('maxent_ne_chunker_tab')

ner_chunks = ne_chunk(pos_tags)
print("\nNamed Entity Recognition (NER):")
print(ner_chunks)

"""Stopword Removal"""

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words_no_punct if word not in stop_words]
print("\nStopword Removal:")
print(filtered_words)
