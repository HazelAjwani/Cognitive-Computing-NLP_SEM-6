# 📝 Text Preprocessing using NLTK

## 📌 Project Description  
This project implements **text preprocessing** using the **Natural Language Toolkit (NLTK)** in Python. The script takes user input (a text paragraph) and processes it through multiple **NLP (Natural Language Processing) steps** such as **tokenization, stemming, lemmatization, POS tagging, stopword removal, punctuation removal, text normalization, and Named Entity Recognition (NER)**.

This is useful for **NLP applications** like **text classification, sentiment analysis, machine translation, and chatbots**.

---

## 🚀 Features Implemented

✔ **Text Normalization** (Convert text to lowercase)  
✔ **Sentence Tokenization** (Splitting text into sentences)  
✔ **Word Tokenization** (Splitting sentences into words)  
✔ **Punctuation Removal** (Removing special characters)  
✔ **Stopword Removal** (Filtering out common words like *"the"*, *"is"*)  
✔ **Stemming** (Reducing words to their root form, e.g., *"running"* → *"run"*)  
✔ **Lemmatization** (Reducing words to dictionary form, e.g., *"better"* → *"good"*)  
✔ **POS (Part-of-Speech) Tagging** (Identifying nouns, verbs, adjectives, etc.)  
✔ **Named Entity Recognition (NER)** (Detecting names of people, places, organizations)  

---

## 🛆 Installation & Setup

1️⃣ **Clone this Repository**  
```sh
git clone https://github.com/your-username/text-preprocessing-nltk.git
cd text-preprocessing-nltk
```

2️⃣ **Install Dependencies**  
Make sure you have Python installed, then install NLTK and other required libraries:
```sh
pip install nltk
```

3️⃣ **Download Required NLTK Data**  
The script requires additional NLTK data files (stopwords, tokenizers, etc.). Run:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

---

## 📝 Code Explanation

1️⃣ **Input Text**  
The user enters a text paragraph that will be preprocessed.

2️⃣ **Text Normalization**  
Converts text to lowercase to ensure consistency.

3️⃣ **Sentence Tokenization**  
Splits the paragraph into sentences using `sent_tokenize()`.

4️⃣ **Word Tokenization**  
Splits each sentence into words using `word_tokenize()`.

5️⃣ **Punctuation Removal**  
Removes punctuation marks like `.,!?;:` using `string.punctuation`.

6️⃣ **Stopword Removal**  
Filters out common stopwords (e.g., "is", "the", "and") using `stopwords.words('english')`.

7️⃣ **Stemming**  
Reduces words to their root form using Porter Stemmer (e.g., "running" → "run") with `PorterStemmer().stem()`.

8️⃣ **Lemmatization**  
Converts words to their dictionary base form (e.g., "better" → "good") using `WordNetLemmatizer().lemmatize()`.

9️⃣ **POS (Part-of-Speech) Tagging**  
Assigns grammatical labels (noun, verb, adjective, etc.) to each word using `pos_tag()`.

🔟 **Named Entity Recognition (NER)**  
Identifies names, places, and organizations using `ne_chunk(pos_tag(words))`.

---

## 🛠 How to Run the Script?

1️⃣ Open a Jupyter Notebook / Google Colab / Python Script  
2️⃣ Copy and run the Python script (`text_preprocessing.py`).  
3️⃣ Enter a sample text when prompted.  
4️⃣ View the step-by-step preprocessing output.  

---

## 📌 Sample Input & Output

**Sample Input:**
```text
Elon Musk, the CEO of Tesla and SpaceX, was born in South Africa.
In 2021, he became the world's richest person!
His companies are revolutionizing the electric car and space industries.
However, critics argue that his management style is controversial.
```

**Expected Output:**
```java
✅ Normalized Text:
elon musk, the ceo of tesla and spacex, was born in south africa. in 2021, he became the world's richest person! his companies are revolutionizing the electric car and space industries. however, critics argue that his management style is controversial.

✅ Sentence Tokenization:
['elon musk, the ceo of tesla and spacex, was born in south africa.', 'in 2021, he became the world's richest person!', ...]

✅ Word Tokenization:
['elon', 'musk', ',', 'the', 'ceo', 'of', 'tesla', 'and', 'spacex', ...]

✅ Punctuation Removal:
['elon', 'musk', 'the', 'ceo', 'of', 'tesla', 'and', 'spacex', ...]

✅ Stopword Removal:
['elon', 'musk', 'ceo', 'tesla', 'spacex', 'born', 'south', 'africa', ...]

✅ Stemming:
['elon', 'musk', 'ceo', 'tesla', 'spacex', 'born', 'south', 'africa', ...]

✅ Lemmatization:
['elon', 'musk', 'ceo', 'tesla', 'spacex', 'born', 'south', 'africa', ...]

✅ POS Tagging:
[('elon', 'NN'), ('musk', 'NN'), ('ceo', 'NN'), ('tesla', 'NNP'), ...]

✅ Named Entity Recognition (NER):
(S
  (PERSON elon/NN musk/NN)
  the/DT
  ceo/NN
  of/IN
  (ORGANIZATION tesla/NNP)
  and/CC
  (ORGANIZATION spacex/NNP)
  ...)
```

---

## 🛠 Technologies Used
- **Python** 🐍  
- **NLTK (Natural Language Toolkit)** 📚  
- **Google Colab / Jupyter Notebook** 📓  

---

## 📈 Applications
This text preprocessing pipeline can be used in:
✔ Sentiment Analysis (Analyzing positive/negative reviews)  
✔ Chatbots & Virtual Assistants 🤖  
✔ Search Engines (Improving query results)  
✔ Text Classification & Spam Detection 📩  
✔ Machine Translation & Speech Recognition 🗣️  

---

## 📚 Author
👨‍💻 Your Name  
📧 Email: your.email@example.com  
🔗 GitHub: [your-github-username](https://github.com/your-github-username)  

If you found this project useful, ⭐ star the repo! 😊  
Happy coding! 🚀

