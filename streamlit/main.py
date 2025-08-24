
from fastapi import FastAPI, HTTPException
import os
import csv
import pickle
from typing import List
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from dotenv import load_dotenv

app = FastAPI()

# Load environment variables from .env file
load_dotenv()
class RecommendationRequest(BaseModel):
    query: str

def load_csv_data(file_path: str) -> List[Document]:
    documents = []
    try:
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                content = (
                    f"Restaurant: {row['name']}, Rating: {row['rate']}, "
                    f"Cost for two: {row['approx_cost(for two people)']}, "
                    f"Online Order: {row['online_order']}, Book Table: {row['book_table']}, "
                    f"Votes: {row['votes']}, Cuisines: {row['cuisines']}, "
                    f"Type: {row['rest_type']}, Location: {row['location']}"
                )
                documents.append(Document(page_content=content, metadata=row))
        return documents
    except FileNotFoundError:
        raise Exception(f"CSV file not found at: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading CSV: {str(e)}")

def save_faiss_index(vector_store, file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(vector_store, f)

def load_faiss_index(file_path: str):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

rag_chain = None

@app.on_event("startup")
async def startup_event():
    global rag_chain
    csv_file_path = "D:\\job\\Restaurant_Recommendation_System\\data\\zomato_cleaned_features.csv"
    index_path = "faiss_index.pkl"
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        documents = load_csv_data(csv_file_path)
        text_splitter = SemanticChunker(embeddings)
        split_documents = text_splitter.split_documents(documents)
        vector_store = load_faiss_index(index_path)
        if vector_store is None:
            vector_store = FAISS.from_documents(split_documents, embeddings)
            save_faiss_index(vector_store, index_path)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
        prompt_template = """
        You are a restaurant recommendation assistant. Based on the user's query and the provided restaurant data, recommend similar restaurants. Consider cuisines, restaurant type, location, rating, cost, and whether they offer online ordering or table booking.

        **User Query**: {query}

        **Context**:
        {context}

        **Instructions**:
        - Recommend up to 5 restaurants that closely match the user's preferences.
        - For each recommendation, provide the restaurant name, cuisines, type, location, rating, cost for two, online order availability, and table booking availability.
        - Format the response as a bulleted list for clarity.
        - If no close matches are found, suggest alternatives with similar characteristics.

        **Response**:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["query", "context"])
        rag_chain = (
            {"context": retriever | (lambda docs: "\n".join([doc.page_content for doc in docs])), "query": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    except Exception as e:
        raise Exception(f"Failed to initialize RAG system: {str(e)}")

@app.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    try:
        if rag_chain is None:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        response = rag_chain.invoke(request.query)
        return {"recommendations": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")