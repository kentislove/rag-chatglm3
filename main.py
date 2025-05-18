import os
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
from utils import (
    load_documents_from_folder,
    crawl_links_from_homepage,
    fetch_urls_from_sitemap,
    save_url_list
)
import gradio as gr
import cohere

LABELS = {
    "zh-TW": {
        "lang": "繁體中文",
        "title": "太盛昌AI助理",
        "ai_qa": "AI 問答",
        "rag_qa": "RAG 問答",
        "input_question": "請輸入問題",
        "username": "帳號",
        "password": "密碼",
        "login": "登入",
        "login_fail": "帳號或密碼錯誤！",
        "submit": "送出",
        "admin_panel": "管理員功能",
        "logout": "登出",
        "upload": "上傳文件（doc, docx, xls, xlsx, pdf, txt）",
        "update_vector": "手動更新向量庫",
        "homepage_url": "全站首頁網址(含http)",
        "sitemap_url": "sitemap.xml網址",
        "success_upload": "檔案已上傳！",
        "homepage_filename": ".url檔名",
        "sitemap_filename": ".url檔名",
        "homepage_crawl": "用首頁爬子頁並產生 .url",
        "sitemap_crawl": "用sitemap自動產生 .url",
        "uploaded": "已上傳：",
        "update_notice": "請點「手動更新向量庫」導入向量資料庫。",
        "db_size": "資料庫大小（Bytes）",
        "vec_count": "向量庫檔案數",
        "cpu": "CPU使用率",
        "ram": "RAM使用情形",
        "disk": "磁碟使用情形",
        "update_status": "向量庫狀態",
        "rag_reply": "RAG 回應",
        "ai_reply": "AI 回應",
        "file_status": "狀態",
        "crawl_status": "爬蟲狀態",
        "lang_select": "🌐 語言 Language",
        "login_as_admin": "請先以管理員登入",
        "admin_locked": "管理員已登出",
    },
    "zh-CN": {
        "lang": "简体中文",
        "title": "太盛昌AI助理",
        "ai_qa": "AI 问答",
        "rag_qa": "RAG 问答",
        "input_question": "请输入问题",
        "username": "账号",
        "password": "密码",
        "login": "登录",
        "login_fail": "账号或密码错误！",
        "submit": "提交",
        "admin_panel": "管理员功能",
        "logout": "登出",
        "upload": "上传文件（doc, docx, xls, xlsx, pdf, txt）",
        "update_vector": "手动更新向量库",
        "homepage_url": "全站首页网址(含http)",
        "sitemap_url": "sitemap.xml URL",
        "success_upload": "文件已上传！",
        "homepage_filename": ".url档名",
        "sitemap_filename": ".url档名",
        "homepage_crawl": "用首页爬子页并产生 .url",
        "sitemap_crawl": "用sitemap自动产生 .url",
        "uploaded": "已上传：",
        "update_notice": "请点“手动更新向量库”导入向量数据库。",
        "db_size": "数据库大小（Bytes）",
        "vec_count": "向量库文件数",
        "cpu": "CPU使用率",
        "ram": "RAM使用情况",
        "disk": "磁盘使用情况",
        "update_status": "向量库状态",
        "rag_reply": "RAG 回复",
        "ai_reply": "AI 回复",
        "file_status": "状态",
        "crawl_status": "爬虫状态",
        "lang_select": "🌐 语言 Language",
        "login_as_admin": "请先以管理员登录",
        "admin_locked": "管理员已登出",
    },
    "en": {
        "lang": "English",
        "title": "KentWare AI BOX",
        "ai_qa": "AI QA",
        "rag_qa": "RAG QA",
        "input_question": "Type your question here",
        "username": "Username",
        "password": "Password",
        "login": "Login",
        "login_fail": "Wrong username or password!",
        "submit": "Submit",
        "admin_panel": "Admin Functions",
        "logout": "Logout",
        "upload": "Upload Files (doc, docx, xls, xlsx, pdf, txt)",
        "update_vector": "Manual Vector Update",
        "homepage_url": "Homepage URL (with http)",
        "sitemap_url": "sitemap.xml URL",
        "success_upload": "File uploaded!",
        "homepage_filename": ".url filename",
        "sitemap_filename": ".url filename",
        "homepage_crawl": "Crawl homepage & save .url",
        "sitemap_crawl": "Crawl sitemap & save .url",
        "uploaded": "Uploaded:",
        "update_notice": "Please click 'Manual Vector Update' to import.",
        "db_size": "DB size (Bytes)",
        "vec_count": "Vector file count",
        "cpu": "CPU usage",
        "ram": "RAM usage",
        "disk": "Disk usage",
        "update_status": "Vector status",
        "rag_reply": "RAG Reply",
        "ai_reply": "AI Reply",
        "file_status": "Status",
        "crawl_status": "Crawl status",
        "lang_select": "🌐 Language",
        "login_as_admin": "Please login as admin first",
        "admin_locked": "Admin logged out",
    },
    "ja": {
        "lang": "日本語",
        "title": "タイセイショウAIアシスタント",
        "ai_qa": "AI 質問",
        "rag_qa": "RAG 質問",
        "input_question": "質問を入力してください",
        "username": "ユーザー名",
        "password": "パスワード",
        "login": "ログイン",
        "login_fail": "ユーザー名またはパスワードが間違っています！",
        "submit": "送信",
        "admin_panel": "管理者機能",
        "logout": "ログアウト",
        "upload": "ファイルをアップロード（doc, docx, xls, xlsx, pdf, txt）",
        "update_vector": "ベクトル手動更新",
        "homepage_url": "ホームページURL（http含む）",
        "sitemap_url": "sitemap.xmlのURL",
        "success_upload": "ファイルがアップロードされました！",
        "homepage_filename": ".urlファイル名",
        "sitemap_filename": ".urlファイル名",
        "homepage_crawl": "ホームページクロールして .url 作成",
        "sitemap_crawl": "sitemap から .url 作成",
        "uploaded": "アップロード済み：",
        "update_notice": "「ベクトル手動更新」を押して反映してください。",
        "db_size": "DBサイズ（Bytes）",
        "vec_count": "ベクトルファイル数",
        "cpu": "CPU使用率",
        "ram": "RAM使用状況",
        "disk": "ディスク使用状況",
        "update_status": "ベクトル状態",
        "rag_reply": "RAG返答",
        "ai_reply": "AI返答",
        "file_status": "状態",
        "crawl_status": "クロール状態",
        "lang_select": "🌐 言語 Language",
        "login_as_admin": "管理者でログインしてください",
        "admin_locked": "管理者ログアウト",
    },
    "ko": {
        "lang": "한국어",
        "title": "태성창 AI 어시스턴트",
        "ai_qa": "AI 질문",
        "rag_qa": "RAG 질문",
        "input_question": "질문을 입력하세요",
        "username": "아이디",
        "password": "비밀번호",
        "login": "로그인",
        "login_fail": "아이디 또는 비밀번호가 틀렸습니다!",
        "submit": "제출",
        "admin_panel": "관리자 기능",
        "logout": "로그아웃",
        "upload": "파일 업로드 (doc, docx, xls, xlsx, pdf, txt)",
        "update_vector": "벡터 수동 업데이트",
        "homepage_url": "홈페이지 URL (http 포함)",
        "sitemap_url": "sitemap.xml URL",
        "success_upload": "파일이 업로드되었습니다!",
        "homepage_filename": ".url 파일명",
        "sitemap_filename": ".url 파일명",
        "homepage_crawl": "홈페이지 크롤링 및 .url 저장",
        "sitemap_crawl": "sitemap으로 .url 저장",
        "uploaded": "업로드됨:",
        "update_notice": "‘벡터 수동 업데이트’를 눌러 반영하세요.",
        "db_size": "DB크기（Bytes）",
        "vec_count": "벡터파일수",
        "cpu": "CPU사용률",
        "ram": "RAM사용상황",
        "disk": "디스크사용상황",
        "update_status": "벡터상태",
        "rag_reply": "RAG 답변",
        "ai_reply": "AI 답변",
        "file_status": "상태",
        "crawl_status": "크롤상태",
        "lang_select": "🌐 언어 Language",
        "login_as_admin": "관리자로 로그인하세요",
        "admin_locked": "관리자 로그아웃",
    }
}
DEFAULT_LANG = "zh-TW"

def get_label(lang, key):
    return LABELS.get(lang, LABELS[DEFAULT_LANG]).get(key, key)

def check_login(username, password):
    return username == "admin" and password == "AaAa691027!!"

def generate_session_id():
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

def ai_chat_llm_only(question, username="user", lang=DEFAULT_LANG):
    session_id = generate_session_id()
    login_type = "user"
    ensure_qa()
    prompt = build_multi_turn_prompt(question, session_id)
    llm_result = cohere_generate(prompt)
    intent = classify_intent(question)
    entities = extract_entities(question)
    summary = summarize_qa(question, llm_result)
    rag_result = qa.invoke({"query": question})
    save_chat(username, question, llm_result, intent, entities, summary, session_id, login_type)
    return llm_result

def rag_answer_rag_only(question, lang_code, username="user", lang=DEFAULT_LANG):
    session_id = generate_session_id()
    login_type = "user"
    ensure_qa()
    force_english = lang_code in ["en", "ja", "ko"]
    q = question if not force_english else f"Please answer the following question in English:\n{question}"
    try:
        docs = qa.retriever.invoke(q)
        rag_result = qa.combine_documents_chain.run(input_documents=docs, question=q)
    except Exception as e:
        rag_result = f"【RAG錯誤】{e}"
    prompt = build_multi_turn_prompt(question, session_id)
    cohere_msg = cohere_generate(prompt)
    intent = classify_intent(question)
    entities = extract_entities(question)
    summary = summarize_qa(question, cohere_msg)
    save_chat(username, question, cohere_msg, intent, entities, summary, session_id, login_type)
    return rag_result

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

init_db()
with gr.Blocks(title="AI 多語助理") as demo:
    lang_map = {LABELS[k]["lang"]: k for k in LABELS}
    lang_names = list(lang_map.keys())
    lang_dropdown = gr.Dropdown(
        choices=lang_names,
        value=LABELS[DEFAULT_LANG]['lang'],
        label=LABELS[DEFAULT_LANG]["lang_select"]
    )

    # --------- AI Chat 多語 ----------
    with gr.Tab(get_label(DEFAULT_LANG, "ai_qa")) as tab_ai_qa:
        ai_question = gr.Textbox(label=get_label(DEFAULT_LANG, "input_question"))
        ai_username = gr.Textbox(label=get_label(DEFAULT_LANG, "username"), value="user")
        ai_output = gr.Textbox(label=get_label(DEFAULT_LANG, "ai_reply"))
        ai_submit = gr.Button(get_label(DEFAULT_LANG, "submit"))

    def ai_chat_ui(question, username, lang):
        lang_key = lang_map.get(lang, DEFAULT_LANG)
        return ai_chat_llm_only(question, username, lang_key)

    ai_submit.click(
        ai_chat_ui,
        inputs=[ai_question, ai_username, lang_dropdown],
        outputs=ai_output
    )

    # --------- RAG QA 多語 ----------
    with gr.Tab(get_label(DEFAULT_LANG, "rag_qa")) as tab_rag_qa:
        rag_question = gr.Textbox(label=get_label(DEFAULT_LANG, "input_question"))
        rag_lang = gr.Textbox(label="語言代碼（en/zh-TW/zh-CN/ja/ko）", value=DEFAULT_LANG)
        rag_username = gr.Textbox(label=get_label(DEFAULT_LANG, "username"), value="user")
        rag_output = gr.Textbox(label=get_label(DEFAULT_LANG, "rag_reply"))
        rag_submit = gr.Button(get_label(DEFAULT_LANG, "submit"))

    def rag_chat_ui(question, lang_code, username, lang):
        lang_key = lang_map.get(lang, DEFAULT_LANG)
        return rag_answer_rag_only(question, lang_code, username, lang_key)

    rag_submit.click(
        rag_chat_ui,
        inputs=[rag_question, rag_lang, rag_username, lang_dropdown],
        outputs=rag_output
    )

    # =========== 管理員登入 UI & TAB ===========
    with gr.Tab(get_label(DEFAULT_LANG, "admin_panel"), visible=False) as tab_admin:
        admin_logout_btn = gr.Button(get_label(DEFAULT_LANG, "logout"))
        add_vec_btn = gr.Button("將所有對話餵進知識庫")
        status_box = gr.Textbox(label="狀態")
        add_vec_btn.click(fn=lambda: (add_chats_to_vectorstore() or "已成功將所有問答導入知識庫！"), outputs=status_box)
        dbsize = gr.Textbox(label=get_label(DEFAULT_LANG, "db_size"))
        vcount = gr.Textbox(label=get_label(DEFAULT_LANG, "vec_count"))
        cpu = gr.Textbox(label=get_label(DEFAULT_LANG, "cpu"))
        ram = gr.Textbox(label=get_label(DEFAULT_LANG, "ram"))
        disk = gr.Textbox(label=get_label(DEFAULT_LANG, "disk"))
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
        update_vec_btn = gr.Button(get_label(DEFAULT_LANG, "update_vector"))
        update_status = gr.Textbox(label=get_label(DEFAULT_LANG, "update_status"))
        update_vec_btn.click(fn=manual_update_vector, outputs=update_status)
        homepage_url = gr.Textbox(label=get_label(DEFAULT_LANG, "homepage_url"))
        homepage_filename = gr.Textbox(label=get_label(DEFAULT_LANG, "homepage_filename"))
        homepage_maxpages = gr.Number(label="最大爬頁數", value=30)
        crawl_btn = gr.Button(get_label(DEFAULT_LANG, "homepage_crawl"))
        crawl_status = gr.Textbox(label=get_label(DEFAULT_LANG, "crawl_status"))
        crawl_btn.click(
            fn=crawl_and_save_urls_homepage,
            inputs=[homepage_url, homepage_filename, homepage_maxpages],
            outputs=crawl_status
        )
        sitemap_url = gr.Textbox(label=get_label(DEFAULT_LANG, "sitemap_url"))
        sitemap_filename = gr.Textbox(label=get_label(DEFAULT_LANG, "sitemap_filename"))
        crawl_sitemap_btn = gr.Button(get_label(DEFAULT_LANG, "sitemap_crawl"))
        crawl_sitemap_status = gr.Textbox(label=get_label(DEFAULT_LANG, "crawl_status"))
        crawl_sitemap_btn.click(
            fn=crawl_and_save_urls_sitemap,
            inputs=[sitemap_url, sitemap_filename],
            outputs=crawl_sitemap_status
        )
        def admin_logout():
            tab_admin.visible = False
            return False
        admin_logout_btn.click(fn=admin_logout, outputs=None)

    with gr.Tab(get_label(DEFAULT_LANG, "upload")):
        upload_file = gr.File(label=get_label(DEFAULT_LANG, "upload"), file_count="multiple")
        upload_status = gr.Textbox(label=get_label(DEFAULT_LANG, "file_status"))
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
                    continue
                shutil.copy(f.name, save_path)
                saved.append(filename)
            if saved:
                return f"已上傳：{', '.join(saved)}\n請手動更新向量庫。"
            else:
                return "沒有支援的檔案被上傳，或全部檔案已存在（未覆蓋）"
        upload_btn = gr.Button(get_label(DEFAULT_LANG, "submit"))
        upload_btn.click(fn=save_uploaded_files, inputs=upload_file, outputs=upload_status)

    # 登入框
    with gr.Row(visible=True) as admin_login_row:
        admin_username = gr.Textbox(label=get_label(DEFAULT_LANG, "username"))
        admin_password = gr.Textbox(label=get_label(DEFAULT_LANG, "password"), type="password")
        admin_login_btn = gr.Button(get_label(DEFAULT_LANG, "login"))
        admin_login_status = gr.Textbox(label="Admin", interactive=False)

    def admin_login_fn(username, password, lang):
        if check_login(username, password):
            tab_admin.visible = True
            admin_login_row.visible = False
            return "", True
        else:
            tab_admin.visible = False
            admin_login_row.visible = True
            return get_label(lang_map.get(lang, DEFAULT_LANG), "login_fail")

    admin_login_btn.click(
        admin_login_fn,
        inputs=[admin_username, admin_password, lang_dropdown],
        outputs=[admin_login_status]
    )

    def switch_lang(selected_lang):
        # AI QA
        tab_ai_qa.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "ai_qa")
        ai_question.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "input_question")
        ai_username.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "username")
        ai_output.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "ai_reply")
        ai_submit.value = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "submit")
        # RAG QA
        tab_rag_qa.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "rag_qa")
        rag_question.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "input_question")
        rag_username.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "username")
        rag_output.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "rag_reply")
        rag_submit.value = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "submit")
        # 管理面
        tab_admin.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "admin_panel")
        dbsize.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "db_size")
        vcount.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "vec_count")
        cpu.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "cpu")
        ram.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "ram")
        disk.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "disk")
        update_vec_btn.value = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "update_vector")
        update_status.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "update_status")
        homepage_url.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "homepage_url")
        homepage_filename.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "homepage_filename")
        crawl_btn.value = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "homepage_crawl")
        crawl_status.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "crawl_status")
        sitemap_url.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "sitemap_url")
        sitemap_filename.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "sitemap_filename")
        crawl_sitemap_btn.value = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "sitemap_crawl")
        crawl_sitemap_status.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "crawl_status")
        upload_file.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "upload")
        upload_status.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "file_status")
        upload_btn.value = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "submit")
        demo.title = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "title")
        admin_username.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "username")
        admin_password.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "password")
        admin_login_btn.value = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "login")
        admin_login_status.label = get_label(lang_map.get(selected_lang, DEFAULT_LANG), "admin_panel")

    lang_dropdown.change(
        switch_lang,
        inputs=[lang_dropdown],
        outputs=[]
    )

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
