from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("medicine_dataset.csv")
print(df.columns)

db_location = "./chrome_langchain_db"
embedding = OllamaEmbeddings(model="nomic-embed-text")
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = [] 
    for i, row in df.iterrows():
        document = Document(
            page_content=str(row["id"]),
            metadata={
                "name": str(row["name"]),
                "substitute0": str(row["substitute0"]),
                "substitute1": str(row["substitute1"]),
                "substitute2": str(row["substitute2"]),
            }
        )
        documents.append(document)
        ids.append(str(i))

vector_store = Chroma(
    collection_name="medicine_collection",
    embedding_function=embedding,
    persist_directory=db_location
    )  
if add_documents:
    vector_store.add_documents(documents, ids=ids)

# connection between llm and vector store

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
    )