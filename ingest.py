import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = "bioinfo_chatbot"
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
NOTES_DIR = os.path.join(BASE_DIR, "data", "notes")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

def load_documents():
    documents = []

    if os.path.exists(PDF_DIR):
        for file in os.listdir(PDF_DIR):
            if file.lower().endswith(".pdf"):
                path = os.path.join(PDF_DIR, file)
                loader = PyPDFLoader(path)
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = file
                documents.extend(docs)

    if os.path.exists(NOTES_DIR):
        for file in os.listdir(NOTES_DIR):
            if file.lower().endswith(".txt"):
                path = os.path.join(NOTES_DIR, file)
                loader = TextLoader(path, encoding="utf-8")
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = file
                documents.extend(docs)

    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

def store_embeddings(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    db.persist()
    return db

if __name__ == "_main_":
    docs = load_documents()

    if not docs: 
        print("No documents found in data/pdfs or data/notes")
        raise SystemExit

    chunks = chunk_documents(docs)
    store_embeddings(chunks)

    print(f"Done. Stored {len(chunks)} chunks in ChromaDB.")