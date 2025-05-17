import os
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from utils import (
    load_documents_from_folder,
    crawl_links_from_homepage,
    fetch_urls_from_sitemap,
    save_url_list
)
import gradio as gr
import tiktoken
from typing import List
import requests

MAX_CONTEXT_TOKENS = 12000

def safe_context_chunks(chunks, max_tokens=MAX_CONTEXT_TOKENS):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    total = 0
    output = []
    for chunk in chunks:
        n = len(tokenizer.encode(chunk.page_content))
        if total + n > max_tokens:
            break
        total += n
        output.append(chunk)
    return output

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
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=30)
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
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=30)
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
            retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        )

def manual_update_vector():
    global vectorstore, qa
    vectorstore = build_vector_store(get_current_docs_state())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return "向量資料庫已手動重建完成"

def duckduckgo_search(query):
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    try:
        res = requests.get(url, timeout=8)
        data = res.json()
        answer = data.get("AbstractText") or data.get("Answer")
        # 先取摘要，再取相關主題
        if not answer:
            related = data.get("RelatedTopics", [])
            if isinstance(related, list) and related:
                if isinstance(related[0], dict):
                    answer = related[0].get("Text", "")
        return answer if answer else "查無 DuckDuckGo 即時答案"
    except Exception as e:
        return f"DuckDuckGo 查詢失敗: {e}"

def rag_answer(question):
    ensure_qa()
    try:
        docs = qa.retriever.get_relevant_documents(question)
        safe_docs = safe_context_chunks(docs)
        result = qa.combine_documents_chain.run(input_documents=safe_docs, question=question)
    except Exception as e:
        return f"【系統錯誤】{e}"
    # 判斷無結果（空、預設句、內容太短），再 fallback
    if (not result) or (result.strip().lower() in ["", "無相關內容", "no relevant content"]) or (len(result.strip()) < 10):
        # 1. 先用 Cohere 直接問
        try:
            direct = llm.invoke(question)
            if direct and len(direct.strip()) > 10:
                return f"【來自外部 Cohere LLM】\n{direct}"
        except Exception as e:
            direct = None
        # 2. 再用 DuckDuckGo 查詢
        duck_ans = duckduckgo_search(question)
        if duck_ans and len(duck_ans.strip()) > 5 and "查無" not in duck_ans:
            return f"【來自外部 DuckDuckGo】\n{duck_ans}"
        return "【查無內容】RAG 與外部查詢都沒有相關結果。"
    # 有內部 RAG 回答
    return f"【來自 RAG 向量資料庫】\n{result}"


def crawl_and_save_urls_homepage(start_url, filename, max_pages=100):
    if not filename or filename.strip() == "":
        filename = "homepage_auto.url"
    if not filename.endswith('.url'):
        filename = filename + '.url'
    file_path = os.path.join(DOCUMENTS_PATH, filename)
    urls = crawl_links_from_homepage(start_url, max_pages=max_pages)
    save_url_list(urls, file_path)
    return f"{len(urls)} 筆網址已存入 {file_path}，請點手動更新向量庫。"

def crawl_and_save_urls_sitemap(sitemap_url, filename):
    if not filename or filename.strip() == "":
        filename = "sitemap_auto.url"
    if not filename.endswith('.url'):
        filename = filename + '.url'
    file_path = os.path.join(DOCUMENTS_PATH, filename)
    urls = fetch_urls_from_sitemap(sitemap_url)
    save_url_list(urls, file_path)
    return f"{len(urls)} 筆網址已存入 {file_path}，請點手動更新向量庫。"

with gr.Blocks() as demo:
    gr.Markdown("# Cohere 向量檢索問答機器人")
    with gr.Row():
        with gr.Column():
            question_box = gr.Textbox(label="輸入問題", placeholder="請輸入問題")
            submit_btn = gr.Button("送出")
            update_btn = gr.Button("手動更新向量庫")
            gr.Markdown("---")
            homepage_url = gr.Textbox(label="全站首頁網址(含http)")
            homepage_filename = gr.Textbox(label=".url檔名(如 demo_homepage.url )")
            homepage_maxpages = gr.Number(label="最大爬頁數", value=30)
            crawl_btn = gr.Button("用首頁爬子頁並產生 .url")
            sitemap_url = gr.Textbox(label="sitemap.xml網址")
            sitemap_filename = gr.Textbox(label=".url檔名(如 demo_sitemap.url )")
            crawl_sitemap_btn = gr.Button("用sitemap自動產生 .url")
        with gr.Column():
            answer_box = gr.Textbox(label="AI 回答")
    submit_btn.click(fn=rag_answer, inputs=question_box, outputs=answer_box)
    update_btn.click(fn=manual_update_vector, inputs=None, outputs=answer_box)
    crawl_btn.click(
        fn=crawl_and_save_urls_homepage,
        inputs=[homepage_url, homepage_filename, homepage_maxpages],
        outputs=None
    )
    crawl_sitemap_btn.click(
        fn=crawl_and_save_urls_sitemap,
        inputs=[sitemap_url, sitemap_filename],
        outputs=None
    )
# update_btn click 放最後，且 outputs=answer_box（顯示提示）
    update_btn.click(fn=manual_update_vector, inputs=None, outputs=answer_box)
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
