import os
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from langchain_cohere import CohereEmbeddings, Cohere
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from utils import load_documents_from_folder
import gradio as gr
from typing import List

# 你的 Cohere API Key，建議實際用時設在環境變數以保安全
COHERE_API_KEY = "DS1Ess8AcMXnuONkQKdQ56GmHXI7u7tkQekQrZDJ"

VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"
DOCS_STATE_PATH = os.path.join(VECTOR_STORE_PATH, "last_docs.json")
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

# 改用 Cohere Embedding + LLM
embedding_model = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)
llm = Cohere(cohere_api_key=COHERE_API_KEY, model="command-r-plus", temperature=0.3)  # 你也可以改 model 參數

vectorstore = None
qa = None

def get_current_docs_state() -> dict:
    docs_state = {}
    for f in os.listdir(DOCUMENTS_PATH):
        path = os.path.join(DOCUMENTS_PATH, f)
        if os.path.isfile(path):
            docs_state[f] = os.path.getmtime(path)
    return docs_state

def load_last_docs_state() -> dict:
    if os.path.exists(DOCS_STATE_PATH):
        with open(DOCS_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_docs_state(state: dict):
    with open(DOCS_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f)

def get_new_or_updated_files(current: dict, last: dict) -> List[str]:
    changed = []
    for name, mtime in current.items():
        if name not in last or last[name] < mtime:
            changed.append(name)
    return changed

def build_vector_store(docs_state: dict = None):
    documents = load_documents_from_folder(DOCUMENTS_PATH)
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_documents(documents)
    if not texts:
        raise RuntimeError("docs 資料夾內沒有可用文件，無法建立向量資料庫，請至少放入一份 txt/pdf/docx/xlsx/csv 檔案！")
    db = FAISS.from_documents(texts, embedding_model)
    db.save_local(VECTOR_STORE_PATH)
    if docs_state:
        save_docs_state(docs_state)
    return db

def add_new_files_to_vector_store(db, new_files: List[str], docs_state: dict):
    from langchain.schema import Document
    from langchain_community.document_loaders import (
        TextLoader,
        UnstructuredPDFLoader,
        UnstructuredWordDocumentLoader,
        UnstructuredExcelLoader
    )
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    new_documents = []
    for file in new_files:
        filepath = os.path.join(DOCUMENTS_PATH, file)
        if os.path.isfile(filepath):
            ext = os.path.splitext(file)[1].lower()
            if ext == ".txt":
                loader = TextLoader(filepath, autodetect_encoding=True)
                docs = loader.load()
            elif ext == ".pdf":
                loader = UnstructuredPDFLoader(filepath)
                docs = loader.load()
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(filepath)
                docs = loader.load()
            elif ext in [".xlsx", ".xls"]:
                loader = UnstructuredExcelLoader(filepath)
                docs = loader.load()
            elif ext == ".csv":
                from utils import parse_csv_file
                docs = parse_csv_file(filepath)
            else:
                print(f"不支援的格式：{file}")
                continue
            new_documents.extend(docs)
    if new_documents:
        texts = splitter.split_documents(new_documents)
        db.add_documents(texts)
        db.save_local(VECTOR_STORE_PATH)
        save_docs_state(docs_state)
    return db

def ensure_qa():
    global vectorstore, qa
    current_docs_state = get_current_docs_state()
    last_docs_state = load_last_docs_state()
    if vectorstore is None:
        if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
            vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embedding_model)
            new_files = get_new_or_updated_files(current_docs_state, last_docs_state)
            if new_files:
                print(f"偵測到新文件或文件變動：{new_files}，自動增量加入向量庫！")
                vectorstore = add_new_files_to_vector_store(vectorstore, new_files, current_docs_state)
            elif len(current_docs_state) != len(last_docs_state):
                print(f"文件數變動，重新建構向量庫")
                vectorstore = build_vector_store(current_docs_state)
        else:
            vectorstore = build_vector_store(current_docs_state)
    else:
        new_files = get_new_or_updated_files(current_docs_state, last_docs_state)
        if new_files:
            print(f"偵測到新文件或文件變動：{new_files}，自動增量加入向量庫！")
            vectorstore = add_new_files_to_vector_store(vectorstore, new_files, current_docs_state)
        elif len(current_docs_state) != len(last_docs_state):
            print(f"文件數變動，重新建構向量庫")
            vectorstore = build_vector_store(current_docs_state)

    if qa is None:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

def rag_answer(question):
    ensure_qa()
    return qa.run(question)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Cohere 向量檢索問答機器人")
    with gr.Row():
        with gr.Column():
            question_box = gr.Textbox(label="輸入問題", placeholder="請輸入問題")
            submit_btn = gr.Button("送出")
        with gr.Column():
            answer_box = gr.Textbox(label="AI 回答")
    submit_btn.click(fn=rag_answer, inputs=question_box, outputs=answer_box)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from gradio.routes import mount_gradio_app
app = mount_gradio_app(app, demo, path="/gradio")

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/gradio")

@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()
    user_message = payload.get("message", "")
    reply = rag_answer(user_message)
    return {"reply": reply}
