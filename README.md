# Resume Ranking System

## Live Application

You can access the hosted version of the Resume Ranking System here:

[**Resume Ranking System - Live Demo**](https://resume-ranking-system-devang-gawade.streamlit.app/)

## Overview

The **Resume Ranking System** is an AI-driven tool designed to streamline the resume screening process for recruiters by matching and ranking resumes against a given job description. This tool leverages Natural Language Processing (NLP), embeddings, and cosine similarity to compute how well a resume aligns with the job requirements, returning a match score for each resume.

The system uses:

- **Sentence Embeddings** for text vectorization using `SentenceTransformer`.
- **CountVectorizer with N-grams** for keyword matching between the job description and resumes.
- **Cosine Similarity** to compute similarity scores.
- **Flask API** to provide a user-friendly interface for uploading resumes and job descriptions.

## Features

- **Text Extraction**: Supports extracting text from PDF, DOC, and DOCX resumes.
- **NLP Preprocessing**: Includes lemmatization, stopword removal, and other advanced text preprocessing techniques.
- **Embeddings and Keyword Matching**: Combines `SentenceTransformer` for embeddings and `CountVectorizer` for keyword-based similarity scoring.
- **Cosine Similarity Scoring**: Calculates similarity between job descriptions and resumes using cosine similarity.
- **API Interface**: Allows uploading of resumes and job descriptions via a Flask API.
- **Parallelized Processing**: Efficiently processes multiple resumes in parallel.

## Tech Stack

- **Programming Language**: Python
- **Libraries**:
  - `Flask`: For building the API.
  - `pdfplumber`, `python-docx`: For extracting text from PDF and DOCX files.
  - `spacy`, `nltk`: For Natural Language Processing tasks like tokenization and lemmatization.
  - `sentence-transformers`: For generating sentence embeddings.
  - `scikit-learn (sklearn)`: For `CountVectorizer` and `cosine_similarity`.
  - `multiprocessing`: For parallelizing resume processing.
- **Models**:
  - `SentenceTransformer` (`all-mpnet-base-v2`): Used for generating high-quality sentence embeddings.
  - `Groq API` (optional): For summarizing resumes and job descriptions using large language models.
