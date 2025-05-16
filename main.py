import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from utils import load_documents_from_folder

import gradio as gr

# 初始化 FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 環境參數
VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

# 初始化嵌入模型與 LLM
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

# 初始化向量資料庫
vectorstore = None
qa = None

def build_vector_store():
    documents = load_documents_from_folder(DOCUMENTS_PATH)
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embedding_model)
    db.save_local(VECTOR_STORE_PATH)
    return db

def load_vector_store():
    return FAISS.load_local(VECTOR_STORE_PATH, embedding_model)

def ensure_qa():
    global vectorstore, qa
    if vectorstore is None:
        if os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
            vectorstore = load_vector_store()
        else:
            vectorstore = build_vector_store()
    if qa is None:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

def rag_answer(question):
    ensure_qa()
    return qa.run(question)

# Gradio 介面
with gr.Blocks() as demo:
    gr.Markdown("# GPT-3.5 向量檢索問答機器人")
    with gr.Row():
        with gr.Column():
            question_box = gr.Textbox(label="輸入問題", placeholder="請輸入問題")
            submit_btn = gr.Button("送出")
        with gr.Column():
            answer_box = gr.Textbox(label="AI 回答")
    submit_btn.click(fn=rag_answer, inputs=question_box, outputs=answer_box)

@app.get("/", response_class=HTMLResponse)
async def index():
    return demo.launch(share=False, inline=True, prevent_thread_lock=True)

@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()
    user_message = payload.get("message", "")
    reply = rag_answer(user_message)
    return {"reply": reply}
