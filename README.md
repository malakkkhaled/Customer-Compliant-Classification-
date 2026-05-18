# 🏦 Customer Complaint Classification

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat&logo=streamlit)
![NLP](https://img.shields.io/badge/NLP-spaCy%20%7C%20NLTK-green?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

An **NLP-based machine learning pipeline** that automatically classifies customer financial complaints into the correct product category using the **CFPB Consumer Complaint Database**.

---

## Problem Statement

Given a free-text complaint written by a consumer, the model predicts one of the predefined financial product categories such as:

-  Credit Card
-  Mortgage
-  Bank Account
-  Debt Collection

This helps financial institutions better understand customer issues, improve response time, and enhance service quality.

---

## 📂 Dataset

**Source:** Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database

🔗 [Official CFPB Website](https://www.consumerfinance.gov/data-research/consumer-complaints/)

> ⚠️ The dataset is not included in this repository due to its large size.  
>  **[Download Dataset from Google Drive](https://drive.google.com/drive/folders/1o5gazR7bkwirTYvuL1d2Uu54eG1qg6Qp?usp=sharing)**  
> After downloading, place the file inside a `/data` folder in the project root.

---

##  Project Pipeline

```
Raw Data → Cleaning → EDA → Text Preprocessing → Feature Engineering → Model Training → Streamlit App
```

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

##  Tech Stack

| Category | Libraries |
|----------|-----------|
| **Data Processing** | Pandas, NumPy |
| **NLP** | spaCy, NLTK |
| **Visualization** | Matplotlib, Seaborn, WordCloud |
| **Machine Learning** | Scikit-learn |
| **UI / Deployment** | Streamlit |
| **Utilities** | TQDM |

---

## 👥 Team Structure & Responsibilities

| Engineer | Role | Responsibilities |
|----------|------|-----------------|
| Engineer #1 | Data Preprocessing & EDA | Data loading, JSON normalization, cleaning, duplicate removal, missing values, EDA, visualization, text preprocessing, lemmatization, noun extraction |
| Engineer #2 | Feature Engineering | TF-IDF vectorization, label encoding, feature preparation, train-test split |
| Engineer #3 | Machine Learning | Model training, evaluation, classification reports, model comparison |
| Engineer #4 | Deployment / UI | Streamlit GUI, model integration, real-time prediction interface |

---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/malakkkhaled/Customer-Compliant-Classification-.git
cd Customer-Compliant-Classification-
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 4. Download the dataset

 [Download from Google Drive](https://drive.google.com/drive/folders/1o5gazR7bkwirTYvuL1d2Uu54eG1qg6Qp?usp=sharing)

Place the downloaded file in:
```
data/complaints.json
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

---

##  Project Structure

```
Customer-Compliant-Classification-/
│
├── data/                   # Dataset folder (not tracked by Git)
├── notebooks/              # Jupyter notebooks
├── app.py                  # Streamlit deployment app
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

---

## 📄 License

This project uses publicly available data from the CFPB and is intended for educational purposes only.
