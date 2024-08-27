import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from io import BytesIO
import PyPDF2

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Load job skills data
data = pd.read_csv("C:/Users/19452/Downloads/jobss.csv")

# Preprocess data
data = data.dropna(subset=['Key Skills'])
data['Key Skills'] = data['Key Skills'].str.replace('|', ' ')

# Train a TF-IDF vectorizer on the job skills data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Key Skills'])

# Define a function to answer questions
def answer_question(question, data, vectorizer, X):
    query_vec = vectorizer.transform([question])
    cosine_similarities = cosine_similarity(query_vec, X).flatten()
    most_similar_idx = cosine_similarities.argmax()
    answer = data.iloc[most_similar_idx]['Job Title']
    return answer

# Streamlit app
st.title("Resume Question Answering")

uploaded_file = st.file_uploader("Upload your resume (PDF format only)", type=["pdf"])

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.write("Resume Text:")
    st.write(resume_text)

    user_question = st.text_input("Ask a question based on your resume:")
    
    if user_question:
        answer = answer_question(user_question, data, vectorizer, X)
        st.write("Answer:")
        st.write(answer)
