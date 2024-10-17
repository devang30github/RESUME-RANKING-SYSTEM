# RESUME-RANKING-SYSTEM

Overview
This project is a Resume Ranking System that matches and ranks resumes based on a given job description. It leverages Natural Language Processing (NLP), cosine similarity, and embeddings to compute the similarity between resumes and the job description, providing a match score. This allows recruiters to efficiently screen candidates by identifying resumes that closely match job requirements.

The system uses:

Sentence Embeddings for text vectorization using the SentenceTransformer model.
CountVectorizer with N-grams to match keywords between the job description and resumes.
Cosine Similarity to compute similarity scores based on embeddings and keywords.
Flask API to allow uploading of resumes and job descriptions for analysis.

Features
Text Extraction: Extracts text from PDF, DOC, and DOCX files.
NLP Preprocessing: Includes lemmatization, stopword removal, and advanced text preprocessing to clean up and normalize the text.
Embeddings and Keyword Matching: Uses SentenceTransformer for generating embeddings and CountVectorizer for keyword-based similarity.
Cosine Similarity Scoring: Computes a similarity score between the job description and each resume.
Parallelized Processing: Processes resumes in parallel to enhance performance.
API Interface: A Flask API endpoint to upload resumes and job descriptions for ranking.

Tech Stack
Programming Language: Python
Libraries:
Flask: For API development.
pdfplumber and python-docx: For extracting text from PDF and DOCX files.
spacy, nltk: For NLP preprocessing.
sentence-transformers: For generating sentence embeddings.
sklearn: For CountVectorizer and cosine_similarity.
multiprocessing: For parallel processing.

Model:
SentenceTransformer (all-mpnet-base-v2): For generating high-quality embeddings.
Groq API (optional): For summarization of resumes and job descriptions using large language models.
