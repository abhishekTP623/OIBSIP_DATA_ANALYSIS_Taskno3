# Sentiment Analysis using Machine Learning

This project focuses on **Sentiment Analysis** of text data using **Machine Learning techniques**.  
The goal is to classify text (tweets) into sentiment categories such as **Positive**, **Negative**, or **Neutral** based on their content.

This project is implemented as a Kaggle notebook and demonstrates a complete NLP pipeline from data preprocessing to model evaluation.

---

## 📌 Project Overview

Sentiment analysis is a Natural Language Processing (NLP) task used to determine the emotional tone behind text.  
In this project, a labeled dataset of tweets is analyzed to build a model that predicts sentiment automatically.

---

## 📊 Dataset Information

- **Source:** Kaggle  
- **Type:** Twitter sentiment dataset  
- **Content:** Tweets with sentiment labels  
- **Target Variable:** Sentiment (Positive / Negative / Neutral)

Each row represents a tweet along with its corresponding sentiment label.

---

## 🧪 Workflow

1. **Data Loading**
   - Import dataset using Pandas

2. **Text Preprocessing**
   - Convert text to lowercase
   - Remove URLs, punctuation, and special characters
   - Remove stopwords
   - Tokenization

3. **Feature Extraction**
   - Bag of Words (BoW)
   - TF-IDF Vectorization

4. **Model Building**
   - Machine Learning classifiers such as:
     - Logistic Regression
     - Naive Bayes
     - Support Vector Machine (SVM)

5. **Model Evaluation**
   - Accuracy score
   - Confusion matrix
   - Classification report

---

## 🛠️ Tools & Libraries Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  
- NLTK  
- Jupyter Notebook / Kaggle Notebook  

---

## 🚀 How to Run

```python
import pandas as pd

# Load dataset
df = pd.read_csv("dataset.csv")
df.head()
