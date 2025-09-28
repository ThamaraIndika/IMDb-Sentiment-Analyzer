import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load classical models
with open(os.path.join(BASE_DIR, "classical_model", "logreg_model.pkl"), "rb") as f:
    logreg = pickle.load(f)
with open(os.path.join(BASE_DIR, "classical_model", "svm_model.pkl"), "rb") as f:
    svm_model = pickle.load(f)
with open(os.path.join(BASE_DIR, "classical_model", "rf_model.pkl"), "rb") as f:
    rf_model = pickle.load(f)
with open(os.path.join(BASE_DIR, "classical_model", "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

# Load DistilBERT
bert_model_path = os.path.join(BASE_DIR, "sentiment_model")
bert_model = DistilBertForSequenceClassification.from_pretrained(bert_model_path)
bert_tokenizer = DistilBertTokenizerFast.from_pretrained(bert_model_path)
bert_classifier = pipeline("sentiment-analysis", model=bert_model, tokenizer=bert_tokenizer)

# Dataset for metrics 
df = pd.read_csv(os.path.join(BASE_DIR, "IMDB Dataset.csv"))
X = vectorizer.transform(df['review'])
y = df['sentiment'].map({'positive': 1, 'negative': 0})


# Streamlit Configuration
st.set_page_config(page_title="IMDb Sentiment Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("IMDb Movie Review Sentiment Analysis")

review = st.text_area("Enter your movie review here:")

if st.button("Predict") and review:
    vect_review = vectorizer.transform([review])   
    # Predictions
    predictions = {
        "Logistic Regression": logreg.predict(vect_review)[0],
        "SVM": svm_model.predict(vect_review)[0],
        "Random Forest": rf_model.predict(vect_review)[0],
    }
    bert_result = bert_classifier(review)[0]
    if bert_result["label"] == "LABEL_0":
        predictions["DistilBERT"] = 0
    else:
        predictions["DistilBERT"] = 1
    st.subheader("Predictions:")
    cols = st.columns(len(predictions))
    for i, (model, pred) in enumerate(predictions.items()):
        label = "POSITIVE" if pred == 1 else "NEGATIVE"
        color = "#4CAF50" if pred == 1 else "#F44336"
        cols[i].markdown(
            f"""
            <div style="background-color:{color};padding:20px;border-radius:10px;text-align:center">
                <h4 style="color:white">{model}</h4>
                <h2 style="color:white">{label}</h2>
            </div>
            """, unsafe_allow_html=True
        )
    # Evaluation Metrics
    st.subheader("Model Performance on Full Dataset:")
    model_metrics = {}
    cms = []
    for model_name, model in zip(["Logistic Regression", "SVM", "Random Forest"], [logreg, svm_model, rf_model]):
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        model_metrics[model_name] = [acc, prec, rec, f1]
        cm = confusion_matrix(y, y_pred)
        cms.append((model_name, cm))
    # Display Metrics
    st.subheader("Confusion Matrices")
    cols = st.columns(3)
    for i, (name, cm) in enumerate(cms):
        fig, ax = plt.subplots(figsize=(3,3))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(name, fontsize=10)
        cols[i].pyplot(fig)
    
    st.subheader("Metrics Comparison")
    metrics_df = pd.DataFrame(model_metrics, index=["Accuracy", "Precision", "Recall", "F1-Score"])
    st.dataframe(metrics_df.style.format("{:.4f}"))
    
    st.subheader("Metrics Bar Chart")
    metrics_long = metrics_df.T.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
    metrics_long.rename(columns={"index": "Model"}, inplace=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = metrics_long["Metric"].unique()
    x = np.arange(len(metrics))
    width = 0.25
    for i, model in enumerate(metrics_long["Model"].unique()):
        scores = metrics_long[metrics_long["Model"] == model]["Score"].values
        ax.bar(x + i*width, scores, width, label=model)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend(title="Model")
    st.pyplot(fig)