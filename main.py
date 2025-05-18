import os
import json
import shutil
import uuid
import sqlite3
import psutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
print("utils path", os.path.abspath("utils.py"))
from utils import (
    load_documents_from_folder,
    crawl_links_from_homepage,
    fetch_urls_from_sitemap,
    save_url_list
)
import gradio as gr
from typing import List
import cohere

# ===========================
# 多語 LABELS
# ===========================
LABELS = {
    "zh-TW": {
        "lang": "繁體中文",
        "title": "太盛昌AI助理",
        "login": "登入",
        "username": "帳號",
        "password": "密碼",
        "login_fail": "帳號或密碼錯誤！",
        "ai_qa": "AI 問答",
        "input_question": "請輸入問題",
        "submit": "送出",
        "admin_panel": "管理員功能",
        "upload": "上傳文件（doc, docx, xls, xlsx, pdf, txt）",
        "update_vector": "手動更新向量庫",
        "homepage_url": "全站首頁網址(含http)",
        "sitemap_url": "sitemap.xml網址",
        "success_upload": "檔案已上傳！",
        "logout": "登出",
        "choose_lang": "選擇語言",
        "user_mode": "使用者模式",
        "admin_mode": "管理員模式",
        "homepage_filename": ".url檔名",
        "sitemap_filename": ".url檔名",
        "homepage_crawl": "用首頁爬子頁並產生 .url",
        "sitemap_crawl": "用sitemap自動產生 .url",
        "uploaded": "已上傳：",
        "update_notice": "請點「手動更新向量庫」導入向量資料庫。"
    },
    "zh-CN": {
        "lang": "简体中文",
        "title": "太盛昌AI助理",
        "login": "登录",
        "username": "账号",
        "password": "密码",
        "login_fail": "账号或密码错误！",
        "ai_qa": "AI 问答",
        "input_question": "请输入问题",
        "submit": "提交",
        "admin_panel": "管理员功能",
        "upload": "上传文件（doc, docx, xls, xlsx, pdf, txt）",
        "update_vector": "手动更新向量库",
        "homepage_url": "全站首页网址(含http)",
        "sitemap_url": "sitemap.xml URL",
        "success_upload": "文件已上传！",
        "logout": "登出",
        "choose_lang": "选择语言",
        "user_mode": "使用者模式",
        "admin_mode": "管理员模式",
        "homepage_filename": ".url档名",
        "sitemap_filename": ".url档名",
        "homepage_crawl": "用首页爬子页并产生 .url",
        "sitemap_crawl": "用sitemap自动产生 .url",
        "uploaded": "已上传：",
        "update_notice": "请点“手动更新向量库”导入向量数据库。"
    },
    "en": {
        "lang": "English",
        "title": "KentWare AI BOX",
        "login": "Login",
        "username": "Username",
        "password": "Password",
        "login_fail": "Wrong username or password!",
        "ai_qa": "AI QA",
        "input_question": "Type your question here",
        "submit": "Submit",
        "admin_panel": "Admin Functions",
        "upload": "Upload Files (doc, docx, xls, xlsx, pdf, txt)",
        "update_vector": "Manual Vector Update",
        "homepage_url": "Homepage URL (with http)",
        "sitemap_url": "sitemap.xml URL",
        "success_upload": "File uploaded!",
        "logout": "Logout",
        "choose_lang": "Choose language",
        "user_mode": "User mode",
        "admin_mode": "Admin mode",
        "homepage_filename": ".url filename",
        "sitemap_filename": ".url filename",
        "homepage_crawl": "Crawl homepage & save .url",
        "sitemap_crawl": "Crawl sitemap & save .url",
        "uploaded": "Uploaded:",
        "update_notice": "Please click 'Manual Vector Update' to import."
    },
    "ja": {
        "lang": "日本語",
        "title": "タイセイショウAIアシスタント",
        "login": "ログイン",
        "username": "ユーザー名",
        "password": "パスワード",
        "login_fail": "ユーザー名またはパスワードが間違っています！",
        "ai_qa": "AI 質問",
        "input_question": "質問を入力してください",
        "submit": "送信",
        "admin_panel": "管理者機能",
        "upload": "ファイルをアップロード（doc, docx, xls, xlsx, pdf, txt）",
        "update_vector": "ベクトル手動更新",
        "homepage_url": "ホームページURL（http含む）",
        "sitemap_url": "sitemap.xmlのURL",
        "success_upload": "ファイルがアップロードされました！",
        "logout": "ログアウト",
        "choose_lang": "言語を選択",
        "user_mode": "ユーザーモード",
        "admin_mode": "管理者モード",
        "homepage_filename": ".urlファイル名",
        "sitemap_filename": ".urlファイル名",
        "homepage_crawl": "ホームページクロールして .url 作成",
        "sitemap_crawl": "sitemap から .url 作成",
        "uploaded": "アップロード済み：",
        "update_notice": "「ベクトル手動更新」を押して反映してください。"
    },
    "ko": {
        "lang": "한국어",
        "title": "태성창 AI 어시스턴트",
        "login": "로그인",
        "username": "아이디",
        "password": "비밀번호",
        "login_fail": "아이디 또는 비밀번호가 틀렸습니다!",
        "ai_qa": "AI 질문",
        "input_question": "질문을 입력하세요",
        "submit": "제출",
        "admin_panel": "관리자 기능",
        "upload": "파일 업로드 (doc, docx, xls, xlsx, pdf, txt)",
        "update_vector": "벡터 수동 업데이트",
        "homepage_url": "홈페이지 URL (http 포함)",
        "sitemap_url": "sitemap.xml URL",
        "success_upload": "파일이 업로드되었습니다!",
        "logout": "로그아웃",
        "choose_lang": "언어 선택",
        "user_mode": "사용자 모드",
        "admin_mode": "관리자 모드",
        "homepage_filename": ".url 파일명",
        "sitemap_filename": ".url 파일명",
        "homepage_crawl": "홈페이지 크롤링 및 .url 저장",
        "sitemap_crawl": "sitemap으로 .url 저장",
        "uploaded": "업로드됨:",
        "update_notice": "‘벡터 수동 업데이트’를 눌러 반영하세요."
    }
}
DEFAULT_LANG = "zh-TW"

def detect_lang():
    return DEFAULT_LANG

def check_login(username, password):
    if username == "admin" and password == "AaAa691027!!":
        return "admin"
    if username == "user" and password == "":
        return "user"
    return None

# ===========================
# Session ID, DB, 向量庫/LLM/NLU/NLG
# ===========================
def generate_session_id(method="uuid", username=None, token=None):
    if method == "uuid":
        return str(uuid.uuid4())
    elif method == "username" and username:
        return f"user-{username}-{uuid.uuid4()}"
    elif method == "token" and token:
        return f"token-{token}"
    else:
        return str(uuid.uuid4())

def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        question TEXT,
        answer TEXT,
        intent TEXT,
        entities TEXT,
        summary TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        session_id TEXT,
        login_type TEXT
    )''')
    conn.commit()
    conn.close()

def save_chat(username, question, answer, intent, entities, summary, session_id, login_type):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('INSERT INTO chat_history (username, question, answer, intent, entities, summary, session_id, login_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
              (username, question, answer, intent, entities, summary, session_id, login_type))
    conn.commit()
    conn.close()

def get_recent_chats(session_id, n=5):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('SELECT question, answer FROM chat_history WHERE session_id=? ORDER BY timestamp DESC LIMIT ?', (session_id, n))
    rows = c.fetchall()
    conn.close()
    return rows[::-1]

def get_db_size():
    if not os.path.exists('chat_history.db'):
        return 0
    return os.path.getsize('chat_history.db')

def get_vectorstore_file_count():
    if not os.path.exists('faiss_index'):
        return 0
    return len([f for f in os.listdir('faiss_index') if os.path.isfile(os.path.join('faiss_index', f))])

# Cohere
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("請設定 Cohere API Key 到 COHERE_API_KEY 環境變數！")
co = cohere.Client(COHERE_API_KEY)

def classify_intent(question):
    examples=[
        cohere.ClassifyExample(text="I want to check order status", label="query"),
        cohere.ClassifyExample(text="Place an order for me", label="order"),
        cohere.ClassifyExample(text="Contact customer service", label="contact"),
        cohere.ClassifyExample(text="What can you do?", label="about"),
        cohere.ClassifyExample(text="Who are you?", label="about"),
    ]
    try:
        response = co.classify(
            inputs=[question],
            examples=examples
        )
        return response.classifications[0].prediction
    except Exception as e:
        return "unknown"


def extract_entities(question):
    import re
    if re.search('[\u4e00-\u9fff]', question):  # 有中文字就跳過
        return "[]"
    try:
        response = co.extract(
            texts=[question],
            examples=[{"text": "I want 5 apples to Taipei", "entities": [
                {"type": "item", "text": "apples"},
                {"type": "quantity", "text": "5"},
                {"type": "location", "text": "Taipei"}
            ]}]
        )
        return str(response[0].entities) if response else "[]"
    except Exception as e:
        return "[]"

def cohere_generate(prompt):
    response = co.generate(
        model="command-r-plus",
        prompt=prompt,
        max_tokens=300,
        temperature=0.3
    )
    return response.generations[0].text.strip()

def summarize_qa(question, answer):
    prompt = f"Summarize the following conversation in one sentence:\nQ: {question}\nA: {answer}"
    return cohere_generate(prompt)

# FAISS 向量庫
VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"
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

def build_vector_store():
    documents = load_documents_from_folder(DOCUMENTS_PATH)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    if not texts:
        raise RuntimeError("No document to build vectorstore.")
    db = FAISS.from_documents(texts, embedding_model)
    db.save_local(VECTOR_STORE_PATH)
    return db

def ensure_qa():
    global vectorstore, qa
    if vectorstore is None:
        if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
            vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embedding_model)
        else:
            vectorstore = build_vector_store()
    if qa is None:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 1})
        )

def add_chats_to_vectorstore():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('SELECT question, answer, summary FROM chat_history')
    rows = c.fetchall()
    conn.close()
    from langchain.schema import Document
    chat_docs = [Document(page_content=f"Q: {q}\nA: {a}\nSummary: {s}") for q, a, s in rows if q and a]
    if chat_docs:
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        texts = splitter.split_documents(chat_docs)
        vectorstore.add_documents(texts)
        vectorstore.save_local(VECTOR_STORE_PATH)

def build_multi_turn_prompt(current_question, session_id):
    history = get_recent_chats(session_id)
    dialog = ""
    for q, a in history:
        dialog += f"User: {q}\nBot: {a}\n"
    dialog += f"User: {current_question}\nBot:"
    return dialog

# ===========================
# RAG 多語入口
# ===========================
def rag_answer_lang(question, lang_code, username="user", session_id=None, login_type="user"):
    force_english = lang_code in ["en", "ja", "ko"]
    q = question if not force_english else f"Please answer the following question in English:\n{question}"
    ensure_qa()
    try:
        docs = qa.retriever.get_relevant_documents(q)
        rag_result = qa.combine_documents_chain.run(input_documents=docs, question=q)
    except Exception as e:
        rag_result = f"【RAG錯誤】{e}"
    prompt = build_multi_turn_prompt(question, session_id)
    try:
        cohere_msg = cohere_generate(prompt)
    except Exception as e:
        cohere_msg = "[LLM error]"
    intent = classify_intent(question)
    entities = extract_entities(question)
    summary = summarize_qa(question, cohere_msg)
    save_chat(username, question, cohere_msg, intent, entities, summary, session_id or "default", login_type)
    return f"【RAG】\n{rag_result}\n\n【LLM】\n{cohere_msg}\n\n【意圖】{intent}\n【實體】{entities}\n【摘要】{summary}\n"

def ai_chat(question, username="user", session_id=None, login_type="user"):
    if not session_id:
        session_id = generate_session_id("uuid", username=username)
    ensure_qa()
    prompt = build_multi_turn_prompt(question, session_id)
    intent = classify_intent(question)
    entities = extract_entities(question)
    llm_result = cohere_generate(prompt)
    summary = summarize_qa(question, llm_result)
    rag_result = qa.run(question)
    save_chat(username, question, llm_result, intent, entities, summary, session_id, login_type)
    return {
        "session_id": session_id,
        "意圖": intent,
        "實體": entities,
        "LLM": llm_result,
        "RAG": rag_result,
        "摘要": summary
    }

# ===========================
# 網站爬蟲功能
# ===========================
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

def manual_update_vector():
    global vectorstore, qa
    vectorstore = build_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return "向量資料庫已手動重建完成"

# ===========================
# Gradio
# ===========================
init_db()
with gr.Blocks(title="Cohere AI 多語助理") as demo:
    with gr.Tab("AI Chat 多語"):
        question_box = gr.Textbox(label="請輸入問題")
        username_box = gr.Textbox(label="使用者帳號", value="user")
        session_id_box = gr.Textbox(label="Session ID (自動產生)", value="")
        login_type_box = gr.Textbox(label="登入類型", value="user")
        output_box = gr.JSON(label="AI 回應")
        submit_btn = gr.Button("送出")
        submit_btn.click(
            fn=ai_chat,
            inputs=[question_box, username_box, session_id_box, login_type_box],
            outputs=output_box
        )
    with gr.Tab("RAG QA 多語"):
        question_box2 = gr.Textbox(label="請輸入問題")
        lang_box = gr.Textbox(label="語言代碼（en/zh-TW/zh-CN/ja/ko）", value="zh-TW")
        username_box2 = gr.Textbox(label="使用者帳號", value="user")
        session_id_box2 = gr.Textbox(label="Session ID", value="")
        login_type_box2 = gr.Textbox(label="登入類型", value="user")
        output_box2 = gr.Textbox(label="回應", lines=10)
        submit_btn2 = gr.Button("送出")
        submit_btn2.click(
            fn=rag_answer_lang,
            inputs=[question_box2, lang_box, username_box2, session_id_box2, login_type_box2],
            outputs=output_box2
        )
    with gr.Tab("管理員功能"):
        add_vec_btn = gr.Button("將所有對話餵進知識庫")
        status_box = gr.Textbox(label="狀態")
        add_vec_btn.click(fn=lambda: (add_chats_to_vectorstore() or "已成功將所有問答導入知識庫！"), outputs=status_box)
        # 系統狀態
        dbsize = gr.Textbox(label="資料庫大小（Bytes）")
        vcount = gr.Textbox(label="向量庫檔案數")
        cpu = gr.Textbox(label="CPU使用率")
        ram = gr.Textbox(label="RAM使用情形")
        disk = gr.Textbox(label="磁碟使用情形")
        stats_btn = gr.Button("立即更新狀態")
        def get_stats():
            return [
                str(get_db_size()),
                str(get_vectorstore_file_count()),
                str(psutil.cpu_percent()),
                str(psutil.virtual_memory()._asdict()),
                str(psutil.disk_usage('/')._asdict())
            ]
        stats_btn.click(fn=get_stats, outputs=[dbsize, vcount, cpu, ram, disk])
        # 手動重建向量庫
        update_vec_btn = gr.Button("手動更新向量庫")
        update_status = gr.Textbox(label="向量庫狀態")
        update_vec_btn.click(fn=manual_update_vector, outputs=update_status)
        # 網站爬蟲
        homepage_url = gr.Textbox(label="全站首頁網址(含http)")
        homepage_filename = gr.Textbox(label=".url檔名")
        homepage_maxpages = gr.Number(label="最大爬頁數", value=30)
        crawl_btn = gr.Button("用首頁爬子頁並產生 .url")
        crawl_status = gr.Textbox(label="爬蟲狀態")
        crawl_btn.click(
            fn=crawl_and_save_urls_homepage,
            inputs=[homepage_url, homepage_filename, homepage_maxpages],
            outputs=crawl_status
        )
        sitemap_url = gr.Textbox(label="sitemap.xml網址")
        sitemap_filename = gr.Textbox(label=".url檔名")
        crawl_sitemap_btn = gr.Button("用sitemap自動產生 .url")
        crawl_sitemap_status = gr.Textbox(label="爬蟲狀態")
        crawl_sitemap_btn.click(
            fn=crawl_and_save_urls_sitemap,
            inputs=[sitemap_url, sitemap_filename],
            outputs=crawl_sitemap_status
        )

    # 上傳功能（跳過重名不覆蓋）
    with gr.Tab("文件上傳"):
        upload_file = gr.File(label="上傳文件（doc, docx, xls, xlsx, pdf, txt）", file_count="multiple")
        upload_status = gr.Textbox(label="狀態")
        def save_uploaded_files(files):
            allowed_exts = {".doc",".docx",".xls",".xlsx",".pdf",".txt"}
            saved = []
            if not files:
                return "請選擇要上傳的文件！"
            if not isinstance(files, list):
                files = [files]
            for f in files:
                filename = os.path.basename(f.name)
                ext = os.path.splitext(filename)[1].lower()
                save_path = os.path.join(DOCUMENTS_PATH, filename)
                if ext not in allowed_exts or os.path.exists(save_path):
                    continue  # 跳過重名或不支援格式
                shutil.copy(f.name, save_path)
                saved.append(filename)
            if saved:
                return f"已上傳：{', '.join(saved)}\n請手動更新向量庫。"
            else:
                return "沒有支援的檔案被上傳，或全部檔案已存在（未覆蓋）"
        upload_btn = gr.Button("上傳")
        upload_btn.click(fn=save_uploaded_files, inputs=upload_file, outputs=upload_status)

# FastAPI + Gradio
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
