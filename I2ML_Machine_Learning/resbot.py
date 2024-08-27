# Import the necessary libraries
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import PyPDF2
from io import BytesIO
import torch

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2-medium'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to generate answers based on the resume text
def generate_answer(resume_text, question):
    # Extract relevant parts of the resume text
    relevant_text = extract_relevant_text(resume_text, question)
    
    text = relevant_text + " " + question
    input_ids = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=1024)
    
    # Set attention mask and pad token id
    attention_mask = torch.tensor([[1] * len(input_ids[0])])
    pad_token_id = tokenizer.eos_token_id
    
    output = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id, max_length=1024, num_return_sequences=1, no_repeat_ngram_size=2)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Function to extract relevant parts of the resume text
def extract_relevant_text(resume_text, question):
    # Extract the parts of the resume text that are relevant to the question
    # Here, I'm using a simple string matching approach
    relevant_text = ""
    for sentence in resume_text.split('.'):
        if any(word in sentence for word in question.split()):
            relevant_text += sentence + '.'
    
    return relevant_text

# Streamlit app
st.title('Resume Question Answering App')

# Resume PDF upload
uploaded_file = st.file_uploader('Upload your resume (PDF format):', type='pdf')

# Extract text from the uploaded PDF
if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
    resume_text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        resume_text += page.extract_text()
    st.write('Resume Text:')
    st.write(resume_text)

    # User question input
    user_question = st.text_input('Ask a question based on your resume:')

    # Generate answer based on user question and resume text
    if user_question:
        answer = generate_answer(resume_text, user_question)
        st.write('Answer:')
        st.write(answer)
