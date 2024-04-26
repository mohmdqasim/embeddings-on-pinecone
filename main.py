# Importing Libraries
import os
import openai
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone
from fastapi import FastAPI, UploadFile, File
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from fastapi import HTTPException
from _utils.document import get_chunks, get_chunks_pdf
from llama_index.vector_stores.pinecone import PineconeVectorStore

from PyPDF2 import PdfReader


load_dotenv() # Loading Enviroment Variables

pc = Pinecone(api_key=os.environ.get('PINE_CONE_API_KEY')) # Setting Pine Cone API Key
embed_model = OpenAIEmbedding(api_key=os.environ.get('OPENAI_API_KEY')) # Setting OpenAI API Key for Embed Model
openai.api_key = os.environ.get('OPENAI_API_KEY') # Setting OpenAI API Key 



app = FastAPI() # Creating An Intance of Fast API

@app.get("/")
def read_root():
    return {'response': "API RUNNING"}


@app.post("/generate_embeddings/")
async def generate_embeddings(useCaseId:str, files: List[UploadFile] = File(...)):
    try:
        try:
            pc.describe_index(useCaseId)
        except:
            pc.create_index(useCaseId)
        index = pc.Index(useCaseId, pool_threads = 32)
        nodes = await get_chunks(files)
        embeddings_list = [embed_model.get_text_embedding(node.text) for node in nodes]

        new_nodes=[]
        for node, embd in zip(nodes, embeddings_list):
            node.embedding = embd 
            new_nodes.append(node)

        vector_store = PineconeVectorStore(
            pinecone_index=index,
        )

        vector_store.add(nodes=new_nodes)

        print(len(new_nodes ))
        return {"response":'Embedding Added Succesfully'}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Yo Man Internal Server Error Man")




@app.post("/generate_embeddings_pdf/")
async def generate_embeddings(useCaseId:str, files: List[UploadFile] = File(...)):
    try:
        try:
            pc.describe_index(useCaseId)
        except:
            pc.create_index(useCaseId)
        index = pc.Index(useCaseId, pool_threads = 32)
        nodes = await get_chunks_pdf(files)
        embeddings_list = [embed_model.get_text_embedding(node.text) for node in nodes]

        new_nodes=[]
        for node, embd in zip(nodes, embeddings_list):
            node.embedding = embd 
            new_nodes.append(node)
        vector_store = PineconeVectorStore(
            pinecone_index=index,
        )
        vector_store.add(nodes=new_nodes)
        return {"response":'Embedding Added Succesfully'}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Yo Man Internal Server Error Man")

    

@app.post("/test/")
async def test(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        reader = PdfReader(file.file)
        page = reader.pages[1]
        text = page.extract_text()
        print(text)
    return {"apple":"yes"}

