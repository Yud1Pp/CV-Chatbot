from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nomic import NomicEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")

embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", nomic_api_key=NOMIC_API_KEY)

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def index_docs(documents):
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

file = "data/file_name.pdf"
documents = load_pdf(file)
chunked_documents = split_text(documents)
index = index_docs(chunked_documents)

index.save_local("faiss-index")