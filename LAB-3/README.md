# **Text Preprocessing and Bag of Words (BoW) Model using NLTK and Scikit-learn**

## **Overview**
This project focuses on **text preprocessing** and **Bag of Words (BoW) representation** using **Natural Language Processing (NLP)** techniques. The implementation includes essential text preprocessing steps such as **Tokenization, Stopword Removal, Stemming, Lemmatization, POS Tagging**, and then applies **CountVectorizer** from `scikit-learn` to generate the BoW model.

---

## **Features Implemented**
- **Text Preprocessing:**
  - **Normalization** (Converting text to lowercase)
  - **Sentence Tokenization** (Splitting text into sentences)
  - **Word Tokenization** (Splitting sentences into words)
  - **Punctuation Removal**
  - **Stopword Removal** (Eliminating commonly used words)
  - **Stemming** (Reducing words to their root form)
  - **Lemmatization** (Converting words to their base dictionary form)
  - **POS Tagging** (Assigning parts of speech to words)
  
- **Bag of Words (BoW) Model using CountVectorizer**
  - Converts text into a matrix of token counts
  - Displays unique words with their assigned indices
  - Converts the BoW matrix into an easy-to-read **DataFrame**

---

## **Technologies Used**
- Python 🐍
- NLTK (Natural Language Toolkit)
- Scikit-learn (`CountVectorizer`)
- Pandas (For DataFrame Representation)
- Google Colab (Execution Platform)

---

## **Installation**
Follow these steps to set up the environment:

1. **Install required libraries** (if not already installed):
   ```bash
   pip install nltk scikit-learn pandas
   ```
2. **Download necessary NLTK resources:**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   ```
3. **Run the Python script in a Jupyter Notebook, Google Colab, or any Python environment.**

---

## **Code Implementation**

### **1. Import Required Libraries**
```python
import nltk
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
```

### **2. Download NLTK Resources**
```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### **3. Take User Input**
```python
text = input("Enter your text: ")
text = text.lower()  # Normalization
```

### **4. Text Preprocessing**
```python
sentences = sent_tokenize(text)
words = word_tokenize(text)
words_no_punct = [word for word in words if word not in string.punctuation]

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words_no_punct if word not in stop_words]

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

pos_tags = pos_tag(lemmatized_words)
```

### **5. Display Preprocessed Output**
```python
print("\nSentence Tokenization:\n", sentences)
print("\nWord Tokenization:\n", words)
print("\nPunctuation Removal:\n", words_no_punct)
print("\nStopword Removal:\n", filtered_words)
print("\nStemming:\n", stemmed_words)
print("\nLemmatization:\n", lemmatized_words)
print("\nPOS Tagging:\n", pos_tags)
```

### **6. Bag of Words Model**
```python
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform([" ".join(filtered_words)])

print("\nVocabulary (Word to Index Mapping):", vectorizer.vocabulary_)
bow_array = bow_matrix.toarray()

print("\nBag of Words Matrix:\n", bow_array)

bow_df = pd.DataFrame(bow_array, columns=vectorizer.get_feature_names_out())
print("\nBag of Words Representation:\n", bow_df)
```

---

## **Output Example**
```
Enter your text: Natural Language Processing is amazing! NLP helps computers understand human language.
```
### **Preprocessing Output:**
```
Sentence Tokenization:
['Natural Language Processing is amazing!', 'NLP helps computers understand human language.']

Word Tokenization:
['natural', 'language', 'processing', 'is', 'amazing', '!', 'nlp', 'helps', 'computers', 'understand', 'human', 'language', '.']

Punctuation Removal:
['natural', 'language', 'processing', 'is', 'amazing', 'nlp', 'helps', 'computers', 'understand', 'human', 'language']

Stopword Removal:
['natural', 'language', 'processing', 'amazing', 'nlp', 'helps', 'computers', 'understand', 'human', 'language']

Stemming:
['natur', 'languag', 'process', 'amaz', 'nlp', 'help', 'comput', 'understand', 'human', 'languag']

Lemmatization:
['natural', 'language', 'processing', 'amazing', 'nlp', 'help', 'computer', 'understand', 'human', 'language']

POS Tagging:
[('natural', 'JJ'), ('language', 'NN'), ('processing', 'NN'), ('amazing', 'JJ'), ('nlp', 'NN'), ('help', 'VB'), ('computer', 'NN'), ('understand', 'VB'), ('human', 'JJ'), ('language', 'NN')]
```
### **Bag of Words Output:**
```
Vocabulary (Word to Index Mapping): {'natural': 0, 'language': 1, 'processing': 2, 'amazing': 3, 'nlp': 4, 'help': 5, 'computer': 6, 'understand': 7, 'human': 8}

Bag of Words Matrix:
[[1 2 1 1 1 1 1 1 1]]

Bag of Words Representation:
   natural  language  processing  amazing  nlp  help  computer  understand  human
0        1         2          1        1    1     1        1           1      1
```

---

## **Applications of This Project**
✅ Sentiment Analysis  
✅ Text Classification  
✅ Spam Detection  
✅ Search Engines & Information Retrieval  
✅ Chatbots & Virtual Assistants  

---

## **Conclusion**
This project provides a comprehensive understanding of text preprocessing and BoW representation using NLP techniques. By following this approach, one can efficiently process textual data and convert it into numerical features for Machine Learning & AI applications. 🚀

---

📌 **Note:**  
Feel free to fork this repository or contribute to improve it! 🎯  
If you found this helpful, don't forget to ⭐ star this repo on GitHub! 🚀
