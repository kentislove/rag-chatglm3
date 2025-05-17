import os
import json
import shutil
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


# 多語 LABELS
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
        "sitemap_url": "sitemap.xml网址",
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

def detect_lang(request: gr.Request):
    accept_language = request.headers.get('accept-language', '').lower()
    if accept_language.startswith("zh-tw"):
        return "zh-TW"
    elif accept_language.startswith("zh-cn"):
        return "zh-CN"
    elif accept_language.startswith("ja"):
        return "ja"
    elif accept_language.startswith("ko"):
        return "ko"
    elif accept_language.startswith("zh"):
        return "zh-TW"
    else:
        return "en"

def check_login(username, password):
    if username == "admin" and password == "AaAa691027!!":
        return "admin"
    if username == "user" and password == "":
        return "user"
    return None

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

VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"
DOCS_STATE_PATH = os.path.join(VECTOR_STORE_PATH, "last_docs.json")
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("請設定 Cohere API Key 到 COHERE_API_KEY 環境變數！")

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
                vectorstore = add_new_files_to_vector_store(vectorstore, new_files, current_docs_state)
            elif len(current_docs_state) != len(last_docs_state):
                vectorstore = build_vector_store(current_docs_state)
        else:
            vectorstore = build_vector_store(current_docs_state)
    else:
        new_files = get_new_or_updated_files(current_docs_state, last_docs_state)
        if new_files:
            vectorstore = add_new_files_to_vector_store(vectorstore, new_files, current_docs_state)
        elif len(current_docs_state) != len(last_docs_state):
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
        if not answer:
            related = data.get("RelatedTopics", [])
            if isinstance(related, list) and related:
                if isinstance(related[0], dict):
                    answer = related[0].get("Text", "")
        return answer if answer else "查無 DuckDuckGo 即時答案"
    except Exception as e:
        return f"DuckDuckGo 查詢失敗: {e}"

def rag_answer_lang(question, lang_code):
    force_english = lang_code in ["en", "ja", "ko"]
    if force_english:
        q = f"Please answer the following question in English:\n{question}"
    else:
        q = question
    ensure_qa()
    # RAG 部分，永遠嘗試，出錯也給出明確提示
    try:
        docs = qa.retriever.get_relevant_documents(q)
        safe_docs = safe_context_chunks(docs)
        rag_result = qa.combine_documents_chain.run(input_documents=safe_docs, question=q)
        if not rag_result or str(rag_result).strip() == "":
            rag_result = "查無內容" if not force_english else "No content"
    except Exception as e:
        rag_result = f"【RAG錯誤】{e}"
    # Cohere LLM 部分，不影響主流程
    try:
        cohere_result = llm.invoke(q)
        cohere_msg = f"{getattr(cohere_result,'content',str(cohere_result)).strip()}"
        if not cohere_msg:
            raise Exception("Empty LLM result")
    except Exception as e:
        if lang_code == "en":
            cohere_msg = "[LLM Error] External AI temporarily unavailable."
        elif lang_code == "ja":
            cohere_msg = "[LLMエラー] 外部AIは一時的に利用できません。"
        elif lang_code == "ko":
            cohere_msg = "[LLM 오류] 외부 AI를 일시적으로 사용할 수 없습니다."
        elif lang_code == "zh-CN":
            cohere_msg = "[LLM错误] 外部AI暂时无法响应。"
        else:
            cohere_msg = "[LLM錯誤] 外部AI暫時無法回應。"

    # DuckDuckGo 可失敗但不影響主流程
    try:
        duck_result = duckduckgo_search(q)
        if not duck_result or str(duck_result).strip() == "" or "查無" in duck_result:
            duck_msg = "查無內容" if not force_english else "No content"
        else:
            duck_msg = duck_result.strip()
    except Exception as e:
        duck_msg = "查無內容" if not force_english else "No content"

    # 最終多來源並列回傳
    msg = f"【RAG】\n{rag_result}\n\n"
    msg += f"【LLM】\n{cohere_msg}\n\n"
    msg += f"【DuckDuckGo】\n{duck_msg}\n"
    return msg

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

with gr.Blocks(title="太盛昌AI助理") as demo:
    lang = gr.State(DEFAULT_LANG)
    role = gr.State("user")
    login_status = gr.State("")
    with gr.Row():
        with gr.Column():
            ai_title = gr.Markdown(f"<h1 id='ai_title'>{LABELS[DEFAULT_LANG]['title']}</h1>")
            lang_selector = gr.Radio(
                choices=[LABELS[k]["lang"] for k in LABELS],
                value=LABELS[DEFAULT_LANG]["lang"],
                label=LABELS[DEFAULT_LANG]["choose_lang"]
            )
            username_input = gr.Textbox(label=LABELS[DEFAULT_LANG]["username"])
            password_input = gr.Textbox(label=LABELS[DEFAULT_LANG]["password"], type="password")
            login_btn = gr.Button(LABELS[DEFAULT_LANG]["login"])
            login_fail_tip = gr.Markdown("")
    with gr.Row():
        with gr.Column(visible=False) as main_left:
            qa_box = gr.Textbox(label=LABELS[DEFAULT_LANG]["input_question"], placeholder=LABELS[DEFAULT_LANG]["input_question"])
            submit_btn = gr.Button(LABELS[DEFAULT_LANG]["submit"])
            answer_box = gr.Textbox(label=LABELS[DEFAULT_LANG]["ai_qa"], elem_id="answer_area", interactive=False, show_copy_button=True)
        with gr.Column(visible=False) as admin_panel:
            upload_file = gr.File(label=LABELS[DEFAULT_LANG]["upload"], file_count="multiple", file_types=[".doc",".docx",".xls",".xlsx",".pdf",".txt"])
            upload_btn = gr.Button(LABELS[DEFAULT_LANG]["upload"])
            update_btn = gr.Button(LABELS[DEFAULT_LANG]["update_vector"])
            homepage_url = gr.Textbox(label=LABELS[DEFAULT_LANG]["homepage_url"])
            homepage_filename = gr.Textbox(label=LABELS[DEFAULT_LANG]["homepage_filename"])
            homepage_maxpages = gr.Number(label="最大爬頁數", value=30)
            crawl_btn = gr.Button(LABELS[DEFAULT_LANG]["homepage_crawl"])
            sitemap_url = gr.Textbox(label=LABELS[DEFAULT_LANG]["sitemap_url"])
            sitemap_filename = gr.Textbox(label=LABELS[DEFAULT_LANG]["sitemap_filename"])
            crawl_sitemap_btn = gr.Button(LABELS[DEFAULT_LANG]["sitemap_crawl"])
            admin_status = gr.Markdown("")

    # ===== 登入邏輯 =====
    def do_login(username, password, lang_str):
        lang_code = None
        for k, v in LABELS.items():
            if v["lang"] == lang_str:
                lang_code = k
                break
        if not lang_code:
            lang_code = DEFAULT_LANG
        role_val = check_login(username, password)
        title = LABELS[lang_code]["title"]
        if role_val == "admin":
            return lang_code, role_val, "", gr.update(visible=True), gr.update(visible=True), gr.update(value=f"<h1 id='ai_title'>{title}</h1>")
        elif role_val == "user":
            return lang_code, role_val, "", gr.update(visible=True), gr.update(visible=False), gr.update(value=f"<h1 id='ai_title'>{title}</h1>")
        else:
            return lang_code, "user", LABELS[lang_code]["login_fail"], gr.update(visible=False), gr.update(visible=False), gr.update(value=f"<h1 id='ai_title'>{title}</h1>")
    login_btn.click(
        fn=do_login,
        inputs=[username_input, password_input, lang_selector],
        outputs=[lang, role, login_fail_tip, main_left, admin_panel, ai_title]
    )
    # AI 問答
    submit_btn.click(fn=rag_answer_lang, inputs=[qa_box, lang], outputs=answer_box)

    # ===== 檔案上傳 =====
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
            if ext not in allowed_exts:
                continue
            save_path = os.path.join(DOCUMENTS_PATH, filename)
            shutil.copy(f.name, save_path)
            saved.append(filename)
        if saved:
            return f"{LABELS[DEFAULT_LANG]['uploaded']} {', '.join(saved)}\n{LABELS[DEFAULT_LANG]['update_notice']}"
        else:
            return "沒有支援的檔案被上傳！僅允許 doc, docx, xls, xlsx, pdf, txt"
    upload_btn.click(fn=save_uploaded_files, inputs=upload_file, outputs=admin_status)
    # ===== 手動更新向量 =====
    update_btn.click(fn=manual_update_vector, inputs=None, outputs=admin_status)
    # ===== 網站爬蟲 =====
    crawl_btn.click(
        fn=crawl_and_save_urls_homepage,
        inputs=[homepage_url, homepage_filename, homepage_maxpages],
        outputs=admin_status
    )
    crawl_sitemap_btn.click(
        fn=crawl_and_save_urls_sitemap,
        inputs=[sitemap_url, sitemap_filename],
        outputs=admin_status
    )
    # ===== CSS =====
    demo.load(None, None, None, js="""
        function() {
            let box = document.querySelector('#answer_area textarea');
            if (box) {
                if (window.innerWidth <= 700) {
                    box.style.height = '200px';
                    box.style.overflowY = 'scroll';
                } else {
                    box.style.height = (window.innerHeight - 320) + 'px';
                    box.style.overflowY = 'auto';
                }
            }
            // 標題即時切換
            let title = document.getElementById('ai_title');
            if (title && box) {
                document.title = title.innerText;
            }
        }
    """)
    # 自動語言偵測
    demo.load(lambda request: (detect_lang(request), "user"), outputs=[lang, role])

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
