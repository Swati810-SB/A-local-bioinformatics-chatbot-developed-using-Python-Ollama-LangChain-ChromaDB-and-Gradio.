import gradio as gr
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

BASE_DIR = "bioinfo_chatbot"
CHROMA_DIR = f"{BASE_DIR}/chroma_db"

def build_qa_chain():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    retriever = db.as_retriever()

    llm = Ollama(model="llama3")  # or another supported model
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"   # default chain type
    )
    return qa_chain

qa_chain = build_qa_chain()

def chat(message, history):
    result = qa_chain.invoke({"query": message})
    answer = result["result"]

    sources = []
    for doc in result.get("source_documents", []):
        src = doc.metadata.get("source", "unknown")
        if src not in sources:
            sources.append(src)
    if sources:
         answer += "\nSources:\n" + "\n".join("- " + src for src in sources[:3])

    return answer

demo  = gr.ChatInterface(
    fn=chat,
    title="Bioinformatics Chatbot for Students",
    description="Ask questions from your bioinformatics PDFs and get answers with sources."
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)