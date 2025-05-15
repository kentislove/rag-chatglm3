import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

import gradio as gr
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
from utils import sync_google_drive_files, load_documents_from_folder, load_webpage

# 憑證初始化
creds_content = os.getenv("GOOGLE_CREDENTIALS_JSON")
if creds_content:
    with open("credentials.json", "w") as f:
        f.write(creds_content)

# FastAPI 初始化
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 參數與資料夾設定
VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

# 嵌入與語言模型
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
llm = HuggingFaceHub(
    repo_id="THUDM/chatglm3-6b",
    model_kwargs={"temperature": 0.5, "max_length": 2048}
)

# 延遲載入 QA 模組
vectorstore = None
qa = None

def ensure_qa():
    global vectorstore, qa
    if vectorstore is None or qa is None:
        if os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
            vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embedding_model)
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )

# 問答核心
def rag_answer(question):
    ensure_qa()
    return qa.run(question) if qa else "目前尚未建立向量資料庫，請先點選重載資料庫"

# Chat 用函式
def chat_fn(msg):
    return rag_answer(msg)

# 建立資料庫（可手動觸發）
def build_vector_store():
    sync_google_drive_files(DOCUMENTS_PATH)
    documents = load_documents_from_folder(DOCUMENTS_PATH)
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embedding_model)
    db.save_local(VECTOR_STORE_PATH)
    return db

# 手動更新資料庫
def refresh_fn():
    global vectorstore, qa
    vectorstore = build_vector_store()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return "資料庫已重新建立"

# 新增網址向量
def ingest_url_fn(url):
    from langchain.schema import Document
    docs = load_webpage(url)
    if not docs:
        return "讀取失敗或無內容"
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_documents(docs)
    global vectorstore, qa
    if vectorstore is None:
        vectorstore = FAISS.from_documents(texts, embedding_model)
    else:
        vectorstore.add_documents(texts)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return f"網址已加入向量庫，共新增 {len(texts)} 段"

# Gradio 多分頁介面
chat_tab = gr.Interface(
    fn=chat_fn,
    inputs=gr.Textbox(lines=2, label="請輸入問題"),
    outputs=gr.Textbox(label="AI 回答"),
    title="RAG AI 機器人 (ChatGLM3-6B)"
)

refresh_tab = gr.Interface(
    fn=refresh_fn,
    inputs=[],
    outputs="text",
    title="重新載入向量資料庫"
)

url_tab = gr.Interface(
    fn=ingest_url_fn,
    inputs=gr.Textbox(label="請輸入網址（新聞/文章）"),
    outputs="text",
    title="新增網頁內容至向量庫"
)

@app.get("/", response_class=HTMLResponse)
async def index():
    return gr.TabbedInterface(
        [chat_tab, refresh_tab, url_tab],
        ["對話機器人", "重載資料庫", "新增網頁"]
    ).launch(
        share=False, inline=True, prevent_thread_lock=True
    )

@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()
    user_message = payload.get("message", "")
    reply = rag_answer(user_message)
    return {"reply": reply}
