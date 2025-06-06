# Aprisity-Assignment

# Resume Screening & Matching AI Agent

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Title
st.set_page_config(page_title="Resume Screening AI", layout="centered")
st.title("📄 Resume Screening & Matching AI Agent")

# Sample Data (You can replace this with file upload later)
st.sidebar.header("Job Description")
job_description = st.sidebar.text_area(
    "Enter Job Description Here:",
    "Looking for a Python developer with data science and machine learning experience."
)

resume_data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'resume_text': [
        "Experienced Python developer with knowledge of ML and data analysis.",
        "Java developer with backend experience and exposure to Spring Boot.",
        "Data scientist with experience in Python, pandas, scikit-learn.",
        "Frontend developer skilled in React, HTML, CSS, and UX design.",
        "AI engineer with NLP, deep learning and TensorFlow background."
    ]
})

st.write("Data Analyst")
st.info(job_description)

# Match Logic
documents = [job_description] + resume_data['resume_text'].tolist()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
resume_data['match_score'] = (similarity_scores * 100).round(2)

# Sort and Display
sorted_matches = resume_data.sort_values(by='match_score', ascending=False)

st.write("###  Top Matching Resumes")
st.dataframe(sorted_matches[['name', 'match_score']].reset_index(drop=True))

# Footer
st.caption(Pavan Kumar | AI Resume Screener")
