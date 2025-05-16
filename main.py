import os
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from utils import load_documents_from_folder
import gradio as gr
from typing import List

COHERE_API_KEY = "DS1Ess8AcMXnuONkQKdQ56GmHXI7u7tkQekQrZDJ"

VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"
DOCS_STATE_PATH = os.path.join(VECTOR_STORE_PATH, "last_docs.json")
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

embedding_model = CohereEmbeddings(
    cohere_api_key=COHERE_API_KEY,
    model="embed-multilingual-v3.0"
)
llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-r-plus",
    temperature=0.3
)
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
        raise RuntimeError("docs 資料夾內沒有可用文件，無法建立向量資料庫，請至少放入一份 txt/pdf/docx/xlsx/csv/url 檔案！")
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
        UnstructuredExcelLoader,
        WebBaseLoader
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
            elif ext == ".url":
                with open(filepath, "r", encoding="utf-8") as f:
                    urls = [line.strip() for line in f if line.strip()]
                    for url in urls:
                        try:
                            web_loader = WebBaseLoader(url)
                            docs = web_loader.load()
                            new_documents.extend(docs)
                        except Exception as e:
                            print(f"讀取網址失敗 {url}: {e}")
                continue
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

def manual_update_vector():
    global vectorstore, qa
    vectorstore = build_vector_store(get_current_docs_state())
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return "向量資料庫已手動重建完成"

def rag_answer(question):
    ensure_qa()
    try:
        result = qa.run(question)
    except Exception as e:
        return f"【系統錯誤】{e}"
    # Fallback：若結果明顯查無內容（可自訂條件）
    if not result or result.strip().lower() in ["", "無相關內容", "no relevant content"]:
        try:
            direct = llm.invoke(question)
            return f"【外部網路搜尋結果】\n{direct}"
        except Exception as e:
            return f"RAG查無結果且外部查詢失敗：{e}"
    return result

with gr.Blocks() as demo:
    gr.Markdown("# Cohere 向量檢索問答機器人")
    with gr.Row():
        with gr.Column():
            question_box = gr.Textbox(label="輸入問題", placeholder="請輸入問題")
            submit_btn = gr.Button("送出")
            update_btn = gr.Button("手動更新向量庫")  # 新增手動按鈕
        with gr.Column():
            answer_box = gr.Textbox(label="AI 回答")
    submit_btn.click(fn=rag_answer, inputs=question_box, outputs=answer_box)
    update_btn.click(fn=lambda: manual_update_vector(), inputs=None, outputs=None)

def manual_update_vector():
    global vectorstore, qa
    vectorstore = build_vector_store(get_current_docs_state())
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return "向量資料庫已手動重建完成"
 def rag_answer(question):
    ensure_qa()
    # 嘗試用RAG
    result = qa.run(question)
    # 檢查答案太短或沒命中時 fallback
    if not result or result.strip().lower() in ["", "無相關內容", "no relevant content"]:
        # 使用 Cohere chat 直接查（有網路知識）
        try:
            direct = llm.invoke(question)
            return f"【外部網路搜尋結果】\n{direct}"
        except Exception as e:
            return f"RAG查無結果且外部查詢失敗：{e}"
    return result
   
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
