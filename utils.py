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


creds_content = os.getenv("GOOGLE_CREDENTIALS_JSON")
if creds_content:
    with open("credentials.json", "w") as f:
        f.write(creds_content)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def chat_fn(msg):
    response = rag_answer(msg)
    return response

def refresh_fn():
    global vectorstore, qa
    vectorstore = build_vector_store()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return "資料庫已重新載入"

def ingest_url_fn(url):
    from langchain.vectorstores import FAISS
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.schema import Document
    docs = load_webpage(url)
    if not docs:
        return "讀取失敗或無內容"
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_documents(docs)
    global vectorstore, qa
    vectorstore.add_documents(texts)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return f"網址內容已加入向量庫，共新增 {len(texts)} 段"

# 建立 Gradio 多分頁 UI
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
    inputs=gr.Textbox(label="請輸入網址（例如新聞/網頁）"),
    outputs="text",
    title="新增網頁內容至向量庫"
)

@app.get("/", response_class=HTMLResponse)
async def index():
    return gr.TabbedInterface([chat_tab, refresh_tab, url_tab], ["對話機器人", "重載資料庫", "新增網頁"]).launch(
        share=False, inline=True, prevent_thread_lock=True
    )

@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()
    user_message = payload.get("message", "")
    reply = rag_answer(user_message)
    return {"reply": reply}
