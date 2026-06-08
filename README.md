# Bioinformatics Chatbot

## Overview
This bioinformatics chatbot is developed using Python, Ollama, LangChain, ChromaDB, and Gradio. It answers questions from uploaded bioinformatics documents by retrieving relevant contex and generating helpful responses.

## Features
- Answers bioinformatics questions from local documents.
- Runs locally with Ollama.
- Uses ChromaDB for retrieval.
- Simple chat interface with Gradio.

## Installation
1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies.
4. Pull the Ollama models.

## Usage
Run python ingest.py to process the PDF files, then run python app.py to start the chatbot. Open the local browser link, ask a question, and get answers based on the uploaded bioinformatics documents.

## Project Structure
- app.py
- ingest.py
- requirements.txt
- data/
- images/
