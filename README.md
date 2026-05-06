# Customer-Compliant-Classification-
The dataset is from the Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database, which provides publicly available financial complaint records submitted by consumers.

The data includes real customer complaints related to financial products and services such as credit cards, mortgages, bank accounts, and debt collection.

Source:
https://www.consumerfinance.gov/data-research/consumer-complaints/

## 🎯 Problem Statement

The goal of this project is to build an NLP-based machine learning model that automatically classifies customer complaints into the correct financial product category.

Given a free-text complaint written by a consumer, the model predicts one of the predefined categories such as Credit Card, Mortgage, Bank Account, or Debt Collection.

This helps financial institutions better understand customer issues, improve response time, and enhance service quality.

## ⚙️ Project Pipeline

This project follows a complete end-to-end NLP workflow:

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
- Converting text into numerical features using TF-IDF
- Preparing input features (X) and target labels (y)

### 6. Model Building 
- Training machine learning models such as Logistic Regression
- Evaluating model performance using accuracy and classification report

### 7. Visualization
- WordCloud for most frequent words
- Bar charts and histograms for insights

- ## 🛠️ Tech Stack

This project was built using the following tools and libraries:

- Python 
- Pandas & NumPy (Data Manipulation)
- Matplotlib & Seaborn (Data Visualization)
- spaCy (Natural Language Processing)
- NLTK (Text Processing)
- Scikit-learn (Machine Learning)
- WordCloud (Text Visualization)
- TQDM (Progress Tracking)
