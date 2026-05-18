import streamlit as st
import joblib
import spacy
import re
import os
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION & CSS
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="FinCare AI | Complaint Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Neon/Glassmorphism UI Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global Background matching the dark image */
    .stApp {
        background-color: #110f1a !important;
        color: #ffffff !important;
    }
    
    /* Force standard text to be bright white */
    p, label, .stMarkdown, .st-emotion-cache-1wivap2 {
        color: #ffffff !important;
    }
    
    /* Make the Sidebar text bright white */
    section[data-testid="stSidebar"] li, 
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    
    /* Fix Dropdown Menu (Selectbox) background so white text is readable */
    div[data-baseweb="select"] > div {
        background-color: #110f1a !important;
        color: white !important;
        border: 1px solid rgba(184, 56, 255, 0.3) !important;
    }
    div[role="listbox"] {
        background-color: #1c152e !important;
    }
    ul[role="listbox"] li {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    
    /* Make placeholder text in text area visible and elegant */
    textarea::placeholder {
        color: #cbd5e1 !important;
        opacity: 0.9 !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    
    /* Large Banner similar to the image's top block */
    .title-glow {
        background: linear-gradient(90deg, #6c2cf5 0%, #b838ff 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        font-size: 2.8rem;
        color: white !important;
        box-shadow: 0 8px 32px rgba(184, 56, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #f1f5f9 !important;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        opacity: 0.9;
    }

    /* Card Container matching the dark blocks in image */
    .glass-card {
        background: #1c152e;
        border: 1px solid rgba(184, 56, 255, 0.15);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }
    
    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #110f1a !important;
        color: #ffffff !important;
        border: 1px solid rgba(184, 56, 255, 0.3) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #b838ff !important;
        box-shadow: 0 0 0 2px rgba(184, 56, 255, 0.2) !important;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #6c2cf5 0%, #b838ff 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        width: 100% !important;
        transition: opacity 0.2s;
    }
    .stButton>button:hover {
        opacity: 0.9 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161324 !important;
        border-right: 1px solid rgba(184, 56, 255, 0.1) !important;
    }
    
    /* Prediction Output Styling (Greenish success bar matching image) */
    .prediction-box {
        background: #1c152e;
        border-left: 5px solid #2ea44f !important; /* Green accent for result */
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 2rem;
    }
    .pred-title {
        color: #f8fafc !important;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .pred-result {
        font-size: 2.2rem;
        font-weight: 800;
        color: #b838ff !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# LOAD RESOURCES
# ------------------------------------------------------------------------------
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy model 'en_core_web_sm' is missing. Please run: python -m spacy download en_core_web_sm")
        return None

nlp_model = load_nlp()

@st.cache_resource
def load_models():
    models_dir = "models"
    available_models = {}
    
    # Check for all 4 models (assuming the user will save them with these names)
    model_paths = {
        "SVM – LinearSVC": os.path.join(models_dir, "svm_model.pkl"),
        "Logistic Regression (LR)": os.path.join(models_dir, "logistic_model.pkl"),
        "Random Forest (RF)": os.path.join(models_dir, "rf_model.pkl"),
        "Naive Bayes (NB)": os.path.join(models_dir, "naive_bayes_model.pkl")
    }
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            available_models[name] = joblib.load(path)
            
    # Load LSA model
    lsa_path = os.path.join(models_dir, "lsa_model.pkl")
    lsa_model = joblib.load(lsa_path) if os.path.exists(lsa_path) else None
    
    # Load TF-IDF Vectorizer
    tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    tfidf_model = joblib.load(tfidf_path) if os.path.exists(tfidf_path) else None
    
    return available_models, lsa_model, tfidf_model

models_dict, lsa_model, tfidf_vectorizer = load_models()

# ------------------------------------------------------------------------------
# PREPROCESSING FUNCTIONS (Same as Notebook)
# ------------------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\w*\d\w*", " ", text)
    return text.strip()

def preprocess_input(text):
    cleaned = clean_text(text)
    if not nlp_model:
        return cleaned
    
    doc = nlp_model(cleaned)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# ------------------------------------------------------------------------------
# MAIN UI
# ------------------------------------------------------------------------------
st.markdown("<div class='title-glow'>FinCare AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Intelligent Customer Complaint Classification System</div>", unsafe_allow_html=True)

# Sidebar for Model Settings
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    st.markdown("Select the ML model for classification:")
    
    if models_dict:
        selected_model_name = st.selectbox("Active Model", list(models_dict.keys()))
        selected_model = models_dict[selected_model_name]
        
        # Known accuracies from the notebook
        accuracies = {
            "SVM – LinearSVC": "82.07%",
            "Logistic Regression (LR)": "81.37%",
            "Random Forest (RF)": "77.70%",
            "Naive Bayes (NB)": "69.75%"
        }
        
        acc = accuracies.get(selected_model_name, "N/A")
        
        # Display Accuracy Box
        st.markdown(f'''
        <div style="background: rgba(28, 21, 46, 0.8); border-left: 4px solid #b838ff; padding: 10px 15px; border-radius: 6px; margin-top: 15px;">
            <div style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Model Accuracy</div>
            <div style="color: #ffffff; font-size: 1.6rem; font-weight: 800;">{acc}</div>
        </div>
        ''', unsafe_allow_html=True)
        
    else:
        st.warning("⚠️ No trained models found in 'models' directory.")
        selected_model = None

    st.markdown("---")
    st.markdown("### 📊 Pipeline Status")
    
    def status_icon(condition):
        return "✅ Ready" if condition else "❌ Missing"
        
    st.markdown(f"- **TF-IDF Vectorizer:** {status_icon(tfidf_vectorizer)}")
    st.markdown(f"- **LSA Reducer:** {status_icon(lsa_model)}")
    st.markdown(f"- **SpaCy NLP:** {status_icon(nlp_model)}")
    
    if not tfidf_vectorizer:
        st.error("TF-IDF Vectorizer is missing. It wasn't saved in the original notebook. Please generate it to enable predictions.")

# Main Application Area
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### 📝 Submit a Complaint")

user_input = st.text_area(
    "Describe your financial issue or complaint below:",
    height=150,
    placeholder="e.g., I have been charged an annual fee on my credit card without prior notification..."
)

if st.button("🚀 Analyze & Classify"):
    if not user_input.strip():
        st.warning("Please enter a complaint description to analyze.")
    elif not tfidf_vectorizer:
        st.error("Cannot predict: TF-IDF Vectorizer is missing. You need to save `tfidf_vectorizer.pkl`.")
    elif not selected_model:
        st.error("Cannot predict: No ML model is loaded.")
    else:
        with st.spinner("Analyzing text patterns using NLP..."):
            # 1. Preprocess
            processed_text = preprocess_input(user_input)
            
            # 2. Vectorize
            tfidf_features = tfidf_vectorizer.transform([processed_text])
            
            # 3. Apply LSA if the model was trained on LSA features
            # In the notebook, Random Forest, SVM, and NB were trained on LSA
            final_features = tfidf_features
            if lsa_model and tfidf_features.shape[1] == 5000:
                final_features = lsa_model.transform(tfidf_features)
                
            # 4. Predict
            prediction = selected_model.predict(final_features)[0]
            
            # 5. Display
            st.markdown(f"""
            <div class='prediction-box'>
                <div class='pred-title'>Predicted Category</div>
                <div class='pred-result'>{prediction}</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("🔍 View Extraction Details"):
                st.markdown("**Cleaned & Lemmatized Text:**")
                st.info(processed_text)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 3rem;'>
    Developed for NLP Project | Powered by Streamlit & Scikit-Learn
</div>
""", unsafe_allow_html=True)
