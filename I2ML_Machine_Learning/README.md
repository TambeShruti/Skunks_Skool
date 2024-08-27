# Resume Enhancement Web Application
App Link: https://huggingface.co/spaces/sanjay11/resumesimilarity

![Screenshot 1](screenshots/MicrosoftTeams-image.png)
## Introduction

### Purpose
This project aims to develop a web application designed to assist individuals in enhancing their resumes. Leveraging advanced Natural Language Processing (NLP) techniques, the application analyzes resumes, suggests improvements, and provides answers to user queries based on the resume content.

### Target Audience
The primary audience for this app includes job seekers, career advisors, and anyone interested in refining their resume to better match job descriptions.

## Technology Used

- **Python:** Chosen for its strong support in data science and NLP libraries.
- **Streamlit app:** An efficient framework for building interactive web apps entirely in Python.
- **Spacy:** Utilized for efficient and accurate NLP tasks, particularly in keyword extraction and text analysis.
- **PyPDF2:** Used to handle PDF file reading and text extraction.
- **Transformers (BERT):** Employed for its state-of-the-art performance in NLP tasks, particularly in question answering.

### Resume Question Answering Model
1. **User Uploads Resume:** Users start by uploading a resume.
2. **Question Input:** After uploading the resume, users can input questions related to the resume content.
3. **Model Processing:** The BERT-based model processes the resume and the questions, leveraging its contextual understanding to provide accurate responses.
4. **Answer Display:** The model returns answers to the user's questions, displayed in the Streamlit application.
5. **Feedback Collection:** Users are encouraged to provide feedback on the quality of the answers. They can indicate whether the answers were satisfactory or not.

![Screenshot 2](screenshots/2.png)

## Application Overview

The application provides various features:

- **Text Extraction from PDF Resumes:** Extracts user-uploaded resumes in PDF format.
- **Question Answering:** Uses a BERT model to answer questions based on the resume content.
- **Keyword Extraction:** Extracts keywords from resumes and job descriptions to suggest improvements.
- **Resume and Job Description Analysis:** Compares the two documents for matching keywords.

### NLP Techniques

#### BERT for Question Answering
- The BERT model, specifically the `bert-large-uncased-whole-word-masking-finetuned-squad`, is used for its excellence in understanding the context of a word in a sentence. The model, pre-trained on a vast corpus and fine-tuned on question-answering tasks, can comprehend and provide precise answers to user queries based on their resume.

#### Keyword Extraction and Resume Analysis
- The `extract_keywords_for_sections` function uses Spacy to identify keywords in both the resume and the job description. The app then suggests improvements and identifies potential project ideas based on these keywords.

#### Spacy for Keyword Extraction and Pattern Matching
- Spacy is used for tokenizing the resume and job description texts, tagging each token with its part of speech. It then uses pattern matching to identify skills, technologies, and project ideas, thereby enabling effective keyword extraction. This extraction is pivotal in analyzing and suggesting enhancements for the resume.

## User Interface and Interaction

- Developed using Streamlit, the app provides a clean and interactive interface.
- Users can upload their resumes, input job descriptions, ask questions, and receive tailored suggestions.
- The interface is designed to be user-friendly, allowing seamless navigation through various features.
![Screenshot 2](screenshots/1.png)
![Screenshot 3](screenshots/3.png)
## Challenges and Solutions

- One of the primary challenges was ensuring the accuracy of NLP tasks.
- This was addressed by carefully selecting and fine-tuning the BERT and Spacy models.
- Another challenge involved creating an intuitive user interface, which was resolved using Streamlit's straightforward framework.

## Future Enhancements

Future plans include integrating more advanced NLP models for broader language support, enhancing the app's scalability, and incorporating real-time resume editing features.

## Conclusion

- This project successfully demonstrates the use of cutting-edge NLP techniques in a practical application.
- It offers valuable assistance in resume enhancement, catering to the needs of job seekers and career professionals alike.
