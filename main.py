import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

import gradio as gr
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub

from utils import sync_google_drive_files, load_documents_from_folder

app = FastAPI()

#CORS 設定（允許 iframe 嵌入）
app.add_middleware(
CORSMiddleware,
allow_origins=[""],
allow_credentials=True,
allow_methods=[""],
allow_headers=["*"],
)

#初始化向量資料庫
VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"

os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
llm = HuggingFaceHub(
repo_id="THUDM/chatglm3-6b",
model_kwargs={"temperature": 0.5, "max_length": 2048}
)

def build_vector_store():
sync_google_drive_files(DOCUMENTS_PATH)
documents = load_documents_from_folder(DOCUMENTS_PATH)
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = splitter.split_documents(documents)
db = FAISS.from_documents(texts, embedding_model)
db.save_local(VECTOR_STORE_PATH)
return db

def load_vector_store():
return FAISS.load_local(VECTOR_STORE_PATH, embedding_model)

if not os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
vectorstore = build_vector_store()
else:
vectorstore = load_vector_store()

qa = RetrievalQA.from_chain_type(
llm=llm,
chain_type="stuff",
retriever=vectorstore.as_retriever()
)

def rag_answer(question):
return qa.run(question)

Gradio UI
def chat_fn(msg):
response = rag_answer(msg)
return response

gradio_app = gr.Interface(
fn=chat_fn,
inputs=gr.Textbox(lines=2, label="請輸入問題"),
outputs=gr.Textbox(label="AI 回答"),
title="RAG AI 機器人 (ChatGLM3-6B)",
)

@app.get("/", response_class=HTMLResponse)
async def index():
return gradio_app.launch(share=False, inline=True, prevent_thread_lock=True)

#Webhook 接收範例（LINE/Telegram 可共用）
@app.post("/webhook")
async def webhook(request: Request):
payload = await request.json()
user_message = payload.get("message", "")
reply = rag_answer(user_message)
return {"reply": reply}