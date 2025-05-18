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
import tiktoken

def get_token_count(text, model="gpt-3.5-turbo"):
    # 以 gpt-3.5-turbo 為例，若 Cohere 有自己的 tokenizer 請替換
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

LABELS = {
    "zh-TW": {
        "lang": "繁體中文",
        "ai_qa": "網路搜尋",
        "rag_qa": "FAQ搜尋",
        "input_question": "請輸入問題",
        "submit": "送出",
        "rag_reply": "RAG 回應",
        "ai_reply": "AI 回應",
    },
    "zh-CN": {
        "lang": "简体中文",
        "ai_qa": "网络搜索",
        "rag_qa": "FAQ搜索",
        "input_question": "请输入问题",
        "submit": "提交",
        "rag_reply": "RAG 回复",
        "ai_reply": "AI 回复",
    },
    "en": {
        "lang": "English",
        "ai_qa": "Web Search",
        "rag_qa": "FAQ Search",
        "input_question": "Type your question here",
        "submit": "Submit",
        "rag_reply": "RAG Reply",
        "ai_reply": "AI Reply",
    },
    "ja": {
        "lang": "日本語",
        "ai_qa": "ウェブ検索",
        "rag_qa": "FAQ検索",
        "input_question": "質問を入力してください",
        "submit": "送信",
        "rag_reply": "RAG返答",
        "ai_reply": "AI返答",
    },
    "ko": {
        "lang": "한국어",
        "ai_qa": "웹 검색",
        "rag_qa": "FAQ 검색",
        "input_question": "질문을 입력하세요",
        "submit": "제출",
        "rag_reply": "RAG 답변",
        "ai_reply": "AI 답변",
    }
}

DEFAULT_LANG = "zh-TW"

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
    splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=50)
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
        splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=50)
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
        docs = docs[:3]
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
    admin_status = gr.State(False)

    with gr.Group(visible=True) as qa_group:
        langs = ["zh-TW", "zh-CN", "en", "ja", "ko"]
        def make_language_tab(lang):
            with gr.Tab(LABELS[lang]["lang"]):
                with gr.Tab(LABELS[lang]["ai_qa"]):
                    ai_question = gr.Textbox(label=LABELS[lang]["input_question"])
                    ai_output = gr.Textbox(label=LABELS[lang]["ai_reply"])
                    ai_submit = gr.Button(LABELS[lang]["submit"])
                    def ai_chat_ui(question):
                        return ai_chat_llm_only(question, "user", lang)
                    ai_submit.click(
                        ai_chat_ui,
                        inputs=[ai_question],
                        outputs=ai_output
                    )
                with gr.Tab(LABELS[lang]["rag_qa"]):
                    rag_question = gr.Textbox(label=LABELS[lang]["input_question"])
                    rag_lang = gr.Textbox(label="語言代碼（en/zh-TW/zh-CN/ja/ko）", value=lang)
                    rag_output = gr.Textbox(label=LABELS[lang]["rag_reply"])
                    rag_submit = gr.Button(LABELS[lang]["submit"])
                    def rag_chat_ui(question, lang_code):
                        return rag_answer_rag_only(question, lang_code, "user", lang)
                    rag_submit.click(
                        rag_chat_ui,
                        inputs=[rag_question, rag_lang],
                        outputs=rag_output
                    )
        for lang in langs:
            make_language_tab(lang)
        # 管理員登入區（底部唯一，不分語系）
        with gr.Row():
            admin_username = gr.Textbox(label="帳號")
            admin_password = gr.Textbox(label="密碼", type="password")
            admin_login_btn = gr.Button("登入")
            admin_login_status = gr.Textbox(label="", value="未登入", interactive=False)

    with gr.Group(visible=False) as admin_group:
        admin_logout_btn = gr.Button("登出")
        add_vec_btn = gr.Button("將所有對話餵進知識庫")
        status_box = gr.Textbox(label="狀態")
        add_vec_btn.click(fn=lambda: (add_chats_to_vectorstore() or "已成功將所有問答導入知識庫！"), outputs=status_box)
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
        update_vec_btn = gr.Button("手動更新向量庫")
        update_status = gr.Textbox(label="向量庫狀態")
        update_vec_btn.click(fn=manual_update_vector, outputs=update_status)

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
                save_path = os.path.join("./docs", filename)
                if ext not in allowed_exts or os.path.exists(save_path):
                    continue
                shutil.copy(f.name, save_path)
                saved.append(filename)
            if saved:
                return f"已上傳：{', '.join(saved)}\n請手動更新向量庫。"
            else:
                return "沒有支援的檔案被上傳，或全部檔案已存在（未覆蓋）"
        upload_btn = gr.Button("送出")
        upload_btn.click(fn=save_uploaded_files, inputs=upload_file, outputs=upload_status)

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

    def do_login(username, password):
        if check_login(username, password):
            return gr.update(visible=False), gr.update(visible=True), "已登入"
        else:
            return gr.update(visible=True), gr.update(visible=False), "帳號或密碼錯誤"
    admin_login_btn.click(
        do_login,
        inputs=[admin_username, admin_password],
        outputs=[qa_group, admin_group, admin_login_status]
    )

    def do_logout():
        return gr.update(visible=True), gr.update(visible=False)
    admin_logout_btn.click(
        do_logout,
        outputs=[qa_group, admin_group]
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
