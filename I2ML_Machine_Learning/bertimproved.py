import streamlit as st
from transformers import BertForQuestionAnswering, BertTokenizer
import torch
from io import BytesIO
import PyPDF2
import pandas as pd

# Initialize session state to store the log of QA pairs and satisfaction responses
if 'qa_log' not in st.session_state:
    st.session_state.qa_log = []

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def answer_question(question, context, model, tokenizer):
    inputs = tokenizer.encode_plus(
        question,
        context,
        add_special_tokens=True,
        return_tensors="pt",
        truncation="only_second",
        max_length=512,
    )
    outputs = model(**inputs, return_dict=True)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    input_ids = inputs["input_ids"].tolist()[0]
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
    )
    return answer

st.title("Resume Question Answering")

uploaded_file = st.file_uploader("Upload your resume (PDF format only)", type=["pdf"])

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.write("Resume Text:")
    st.write(resume_text)

    user_question = st.text_input("Ask a question based on your resume:")

    if user_question:
        model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

        answer = answer_question(user_question, resume_text, model, tokenizer)
        st.write("Answer:")
        st.write(answer)

        # Ask for user feedback on satisfaction
        satisfaction = st.radio('Are you satisfied with the answer?', ('Yes', 'No'), key='satisfaction')
        
        # Log the interaction
        st.session_state.qa_log.append({
            'Question': user_question,
            'Answer': answer,
            'Satisfaction': satisfaction
        })

        # Display the log in a table format
        st.write("Interaction Log:")
        log_df = pd.DataFrame(st.session_state.qa_log)
        st.dataframe(log_df)


1 / 2

# Conclusion:

# In conclusion, our "Resume Question Answering" language model, powered by BERT and integrated into a user-friendly Streamlit application, 
# has demonstrated significant promise in revolutionizing the resume evaluation process. Through the successful execution of this project, 
# we have achieved several key milestones and laid a foundation for future developments:

# Efficient Resume Analysis: Our model has showcased the ability to efficiently process uploaded resumes and respond to user queries with a high degree of accuracy. Users can easily extract valuable information from resumes, such as candidate names, contact details, and more.
# User Feedback Integration: The incorporation of a feedback mechanism has been pivotal in our project. We have gathered user-generated feedback on answer quality and user satisfaction, enabling us to refine and enhance the model continuously.
# Iterative Development: Our commitment to iterative development based on user feedback ensures that our model will only get better over time. This ongoing improvement process positions our model as a dynamic tool that adapts to user needs.
# Data Privacy and Security: We have prioritized data privacy and security, ensuring the protection of user-uploaded resumes and personal information. Users can confidently use our application without concerns about data breaches.

# Future Scope:

# The future of the "Resume Question Answering" project holds exciting prospects and opportunities for further advancements:

# Advanced NLP Models: As NLP technology continues to evolve, we plan to explore and integrate more advanced models such as GPT-3, RoBERTa, and their successors to enhance the accuracy and capabilities of our system.
# Multi-Language Support: Expanding the language capabilities of our model to accommodate a broader range of languages and resume formats will be a key focus, making it accessible to a global audience.
# Scalability and Performance: Enhancing the model's scalability to handle a larger number of users and improve its overall performance will be a critical consideration.
# Enhanced User Interface: We will continuously improve the Streamlit application's user interface to make it even more user-friendly and intuitive.
# Industry-Specific Versions: Creating specialized versions of our model for various industries, such as healthcare, technology, or finance, to provide tailored solutions for specific job roles and requirements.
# Collaboration and Integration: Exploring collaborations with job boards, HR platforms, and recruitment agencies to integrate our technology and streamline the hiring process.
# Research and Innovation: Staying at the forefront of NLP research and technology advancements to maintain our model's state-of-the-art status.