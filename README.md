# Customer-Compliant-Classification

The dataset is from the Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database, which provides publicly available financial complaint records submitted by consumers.

The data includes real customer complaints related to financial products and services such as credit cards, mortgages, bank accounts, and debt collection.

---

## Source
Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database  
https://www.consumerfinance.gov/data-research/consumer-complaints/
---

## 👥 Team Structure & Responsibilities

| Engineer | Role | Responsibilities |
|----------|------|------------------|
| Engineer #1 | Data Preprocessing & EDA Engineer | Data loading, JSON normalization, data cleaning, duplicate removal, handling missing values, EDA, visualization, text preprocessing, lemmatization, noun extraction |
| Engineer #2 | Feature Engineering Engineer | TF-IDF vectorization, label encoding, feature preparation, train-test split |
| Engineer #3 | Machine Learning Engineer | Model training, model evaluation, classification reports, model comparison |
| Engineer #4 | Deployment / UI Engineer | Streamlit GUI, model integration, real-time prediction interface |

---

## 🎯 Problem Statement

The goal of this project is to build an NLP-based machine learning model that automatically classifies customer complaints into the correct financial product category.

Given a free-text complaint written by a consumer, the model predicts one of the predefined categories such as Credit Card, Mortgage, Bank Account, or Debt Collection.

This helps financial institutions better understand customer issues, improve response time, and enhance service quality.

---

## ⚙️ Project Pipeline

### 1. Data Collection
- Dataset loaded from CFPB Consumer Complaint Database  
- Conversion from JSON to structured DataFrame  

### 2. Data Cleaning
- Handling missing values  
- Removing duplicates  
- Standardizing column names  
- Filtering rare classes (less than 500 samples)  

### 3. Exploratory Data Analysis (EDA)
- Analysis of complaint distribution over time  
- Most frequent products and issues  
- State-wise complaint distribution  
- Visualization of trends and patterns  

### 4. Text Preprocessing
- Lowercasing text  
- Removing punctuation, numbers, and special characters  
- Lemmatization using spaCy  
- Removing stopwords  
- Extracting meaningful tokens (nouns)  

### 5. Feature Engineering
- TF-IDF vectorization  
- Preparing input features (X) and target labels (y)  

### 6. Model Building
- Training Logistic Regression model  
- Evaluating performance using accuracy & classification report  

### 7. Visualization
- WordCloud for most frequent words  
- Bar charts and histograms  

---

## 🛠️ Tech Stack

- Python  
- Pandas & NumPy  
- Matplotlib & Seaborn  
- spaCy  
- NLTK  
- Scikit-learn  
- WordCloud  
- TQDM  
- Streamlit  
