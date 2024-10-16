import streamlit as st
import time
import pdfplumber
import docx
import re
from sklearn.metrics.pairwise import cosine_similarity
import os
import nltk
import spacy
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor

# Download NLTK Data
nltk.download('punkt')
nltk.download('stopwords')

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize models
summarizer_model = ChatGroq(model="llama-3.1-70b-versatile", groq_api_key=groq_api_key)
sentence_transformer_model = SentenceTransformer('paraphrase-mpnet-base-v2')
nlp = spacy.load("en_core_web_sm")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text)
    return text

def advanced_preprocessing(text):
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    return lemmatized_text

def extract_text_from_pdf(pdf_file):
    text = ''
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return preprocess_text(text)

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    raw_text = '\n'.join([para.text for para in doc.paragraphs])
    return preprocess_text(raw_text)

def extract_text_from_file(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    if file_extension == '.pdf':
        return extract_text_from_pdf(file)
    elif file_extension in ['.doc', '.docx']:
        return extract_text_from_docx(file)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def summarize_job_description_with_groq(job_description):
    prompt = f"Summarize the following job description..."
    try:
        response = summarizer_model.invoke(input=prompt)
        return response.content.strip() if response and response.content.strip() else job_description
    except Exception as e:
        st.error(f"Error generating job description summary: {str(e)}")
        return job_description

def summarize_resume_with_groq(resume_text):
    prompt = f"Summarize the following resume..."
    try:
        response = summarizer_model.invoke(input=prompt)
        return response.content.strip() if response and response.content.strip() else resume_text
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return resume_text

def batch_get_embeddings(texts, transformer_model):
    return transformer_model.encode(texts, convert_to_tensor=True, batch_size=8)

def rank_resumes_separately(resume_texts, job_description, transformer_model):
    summarized_job_description = advanced_preprocessing(summarize_job_description_with_groq(job_description))
    job_desc_embs = batch_get_embeddings([summarized_job_description], transformer_model)
    lemmatized_job_description = advanced_preprocessing(job_description)

    # Process resumes sequentially or in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        resume_summaries = list(executor.map(summarize_resume_with_groq, [resume_text for resume_text, _ in resume_texts]))

    resume_embeddings = batch_get_embeddings(resume_summaries, transformer_model)

    scores = []
    vectorizer = TfidfVectorizer()
    for i, (resume_text, file_name) in enumerate(resume_texts):
        lemmatized_resume = advanced_preprocessing(resume_text)
        vectors = vectorizer.fit_transform([lemmatized_job_description, lemmatized_resume])
        similarity_score_keywords = cosine_similarity(vectors[0:1], vectors[1:2])
        keyword_boost = similarity_score_keywords[0][0] * 0.2
        
        similarity_score_embedding = cosine_similarity(job_desc_embs, resume_embeddings[i].reshape(1, -1)).mean()

        total_score = similarity_score_embedding + keyword_boost
        total_score *= 100

        scores.append((file_name, total_score))

    return sorted(scores, key=lambda x: x[1], reverse=True)

# Streamlit UI starts here
st.title("📄 Resume Screening Tool")

st.write("Upload resumes and provide a job description to rank them based on job matching criteria.")

job_description = st.text_area("Please paste the job description here", height=300, placeholder="Enter the job requirements")

resumes = st.file_uploader("Upload one or more resumes (PDF, DOCX)", accept_multiple_files=True, type=["pdf", "docx"])

if st.button("Submit"):
    if job_description and resumes:
        with st.spinner("Processing..."):
            resume_texts = [(extract_text_from_file(resume), resume.name) for resume in resumes]
            ranked_resumes = rank_resumes_separately(resume_texts, job_description, sentence_transformer_model)

        st.subheader("Ranked Resumes")
        for file_name, score in ranked_resumes:
            st.markdown(f"**{file_name}**: {score:.2f}%")
    else:
        st.error("Please provide both a job description and upload resumes.")

# Add Custom CSS Styling
st.markdown('''
    <style>
        /* Page background and fonts */
        body {
            background-color: #f5f5f5;
            color: #333333;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Title styling */
        h1 {
            color: #2E8B57;  /* Darker green */
            font-size: 2.5em;
            text-align: center;
        }

        /* Subheading styling */
        h2 {
            color: #3D85C6;  /* Light blue */
            font-size: 1.8em;
            margin-top: 20px;
        }

        /* Description text area styling */
        textarea {
            font-size: 1.2em;
            padding: 10px;
            border-radius: 10px;
            border: 2px solid #3D85C6;
            width: 100%;
        }

        /* File uploader drop zone styling */
        div[data-testid="stFileUploadDropzone"] {
            border: 2px dashed #2E8B57;  /* Dark green border */
            padding: 20px;
            background-color: #f0fff0;  /* Light green background */
            border-radius: 15px;
            width: 100%;
        }

        /* Submit button styling */
        div.stButton > button {
            background-color: #2E8B57;  /* Dark green background */
            color: white;
            font-size: 1.2em;
            padding: 10px 20px;
            border: none;
            border-radius: 10px;
            width: 100%;
            cursor: pointer;
        }

        div.stButton > button:hover {
            background-color: #3D85C6;  /* Blue on hover */
        }

        /* Styling for ranked resume results */
        .ranked-resume {
            background-color: #e0f7fa;  /* Light cyan background */
            border-left: 4px solid #2E8B57;
            padding: 5px;
            margin-bottom: 5px;
            border-radius: 10px;
            height:90px;
        }

        .ranked-resume h4 {
            color: #2E8B57;
            font-size: 1.2em;
            margin-bottom: 5px;
        }

        .ranked-resume p {
            font-size: 1.1em;
        }

        /* Processing loader styling */
        .processing-loader {
            text-align: center;
            font-size: 1.5em;
            color: #3D85C6;
            margin-top: 20px;
        }
    </style>
''', unsafe_allow_html=True)
