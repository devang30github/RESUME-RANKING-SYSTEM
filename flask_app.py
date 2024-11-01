import pdfplumber
import docx
import re
from sklearn.metrics.pairwise import cosine_similarity
import os
from flask import Flask, request, jsonify
import nltk
import spacy
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from multiprocessing import Pool

nltk.download('punkt')
nltk.download('stopwords')

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize models
summarizer_model = ChatGroq(model="llama-3.1-70b-versatile", groq_api_key=groq_api_key)
sentence_transformer_model = SentenceTransformer('all-mpnet-base-v2')  # Advanced sentence transformer
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f'[{string.punctuation}]', '', text)  # Remove punctuation
    return text

def advanced_preprocessing(text):
    stop_words = set(stopwords.words('english'))
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Lemmatize the remaining text
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    return lemmatized_text



# Text extraction functions
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
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension == '.pdf':
        return extract_text_from_pdf(file)
    elif file_extension in ['.doc', '.docx']:
        return extract_text_from_docx(file)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

# Summarize job description using Groq
def summarize_job_description_with_groq(job_description):
    prompt = f"""
    Summarize the following job description in a structured format. Extract key points only, focusing on specific technical and soft skills. The format should be:

    - Primary Responsibilities: What are the main tasks and duties of the role?
    - Required Technical Skills: List all programming languages, frameworks, tools, and technologies required for this role.
    - Years of Experience: How many years of experience are required?
    - Preferred Qualifications: What are the preferred qualifications or nice-to-haves?
    - Educational Requirements: What degree(s) or certifications are needed or preferred?
    - Job Location and Remote Work Options: Is the job on-site, remote, or hybrid?

    Job Description:
    {job_description}
    """
    try:
        response = summarizer_model.invoke(input=prompt)
        return response.content.strip() if response and response.content.strip() else job_description
    except Exception as e:
        print(f"Error generating job description summary: {str(e)}")
        return job_description

# Summarize resume using Groq
def summarize_resume_with_groq(resume_text):
    prompt = f"""

    Summarize the following resume in a detailed yet concise format. Extract key points that are relevant to the Machine Learning Engineer role. The format should be:

    - Core Technical Skills: Specific tools, languages, and frameworks the candidate has experience with (include proficiency level if possible).
    - Work Experience: Briefly list previous roles, responsibilities, and quantifiable impact (e.g., model accuracy improvement, pipeline optimization, etc.).
    - Education: Highest degree earned and any relevant coursework or specializations.
    - Certifications and Relevant Achievements: Include any certifications, competitions, research papers, or awards relevant to machine learning or AI.
    - Projects: Highlight one or two relevant projects, detailing the tools used and the impact of the project.

    Resume:
    {resume_text}
    """
    try:
        response = summarizer_model.invoke(input=prompt)
        return response.content.strip() if response and response.content.strip() else resume_text
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return resume_text

# Get sentence embeddings using SentenceTransformer
def batch_get_embeddings(texts, transformer_model):
    return transformer_model.encode(texts, convert_to_tensor=True, batch_size=8)


# Parallelized resume preprocessing
def parallel_preprocessing(resume_texts):
    with Pool() as pool:
        return pool.map(advanced_preprocessing, resume_texts)

# Rank resumes based on embeddings and keywords
def rank_resumes_separately(resume_texts, job_description, transformer_model):
    # Summarize and embed the job description
    summarized_job_description = advanced_preprocessing(summarize_job_description_with_groq(job_description))
    job_desc_embs = batch_get_embeddings([summarized_job_description], transformer_model)
    lemmatized_job_description = advanced_preprocessing(job_description)
    
    # Parallel processing for resume summarization and lemmatization
    resume_summaries = parallel_preprocessing([summarize_resume_with_groq(resume_text) for resume_text, _ in resume_texts])
    resume_embeddings = batch_get_embeddings(resume_summaries, transformer_model)

    scores = []
    vectorizer=CountVectorizer(ngram_range=(1, 2))
    vectorizer.fit([lemmatized_job_description])
    vectors_job = vectorizer.transform([lemmatized_job_description])
    for i, (resume_text, file_name) in enumerate(resume_texts):
        # Cosine similarity between summarized job description and resume embeddings
        similarity_score_embedding = cosine_similarity(job_desc_embs, resume_embeddings[i].reshape(1, -1)).mean()

        #cosine similarity between whole text without summarization to not missed any information
        lemmatized_resume = advanced_preprocessing(resume_text)
        vectors_resume = vectorizer.transform([ lemmatized_resume])
        similarity_score_keywords = cosine_similarity(vectors_job, vectors_resume)
        keyword_boost = similarity_score_keywords[0][0] * 0.4  # Boost weight
        
        

        # Calculate the final score with adjusted weights
        total_score = similarity_score_embedding + keyword_boost
        total_score *= 100


        scores.append((file_name, total_score))

    return sorted(scores, key=lambda x: x[1], reverse=True)

# API route to upload resumes and rank them
@app.route('/upload_resume', methods=['POST'])
def upload_files():
    try:
        job_description = request.form['job_description']
        resumes = request.files.getlist('resumes')
        
        resume_texts = []
        for resume in resumes:
            resume_text = extract_text_from_file(resume)
            resume_texts.append((resume_text, resume.filename))
        
        # Rank the resumes and print output for each resume
        ranked_resumes = rank_resumes_separately(resume_texts, job_description, sentence_transformer_model)

        
        return jsonify([{"filename": file_name, "score": f"{float(score):.2f}%"} 
                        for file_name, score in ranked_resumes])
    
    except Exception as e:
        print(f"Error in /upload_resume endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)



