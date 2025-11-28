# vectordb_build.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1️⃣ Load your PDF documents
docs = []
for file in os.listdir("docs"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join("docs", file))
        docs.extend(loader.load())

# 2️⃣ Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(docs)

# 3️⃣ Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4️⃣ Create FAISS vectorstore
vectorstore = FAISS.from_documents(docs, embeddings)

# 5️⃣ Save it locally
if not os.path.exists("vectorstore"):
    os.makedirs("vectorstore")

vectorstore.save_local("vectorstore")

print("Vectorstore built successfully!")
