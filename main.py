import os
import shutil
import uuid
import sqlite3
import psutil
import csv
import tempfile
import time
import threading

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, PlainTextResponse

from linebot.exceptions import InvalidSignatureError
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage

import gradio as gr
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

import cohere

# === 多語 label 及樣式 ===
LABELS = {
    "zh-TW": {"lang": "繁體中文", "ai_qa": "網路搜尋", "rag_qa": "FAQ搜尋", "input_question": "請輸入問題", "submit": "送出", "rag_reply": "RAG 回應", "ai_reply": "AI 回應"},
    "zh-CN": {"lang": "简体中文", "ai_qa": "网络搜索", "rag_qa": "FAQ搜索", "input_question": "请输入问题", "submit": "提交", "rag_reply": "RAG 回复", "ai_reply": "AI 回复"},
    "en": {"lang": "English", "ai_qa": "Web Search", "rag_qa": "FAQ Search", "input_question": "Type your question here", "submit": "Submit", "rag_reply": "RAG Reply", "ai_reply": "AI Reply"},
    "ja": {"lang": "日本語", "ai_qa": "ウェブ検索", "rag_qa": "FAQ検索", "input_question": "質問を入力してください", "submit": "送信", "rag_reply": "RAG返答", "ai_reply": "AI返答"},
    "ko": {"lang": "한국어", "ai_qa": "웹 검색", "rag_qa": "FAQ 검색", "input_question": "질문을 입력하세요", "submit": "제출", "rag_reply": "RAG 답변", "ai_reply": "AI 답변"},
}
DEFAULT_LANG = "zh-TW"
STYLE_PROMPT = {
    "zh-TW": "請以溫暖、貼心、鼓勵、分析、細膩、簡短扼要但不失重點的方式回答，**並以不超過30字**精簡扼要回應：",
    "zh-CN": "请以温暖、贴心、鼓励、分析、细腻、简短扼要但不失重点的方式回答，**且不超过30字**简短扼要回复：",
     "en":  "Please answer in less than 30 words, concise and clear: "
}

# === DB ===
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
        login_type TEXT,
        ip4 TEXT,
        platform TEXT,
        line_display_name TEXT
    )''')
    conn.commit()
    conn.close()

def save_chat(
    username, question, answer, intent, entities, summary,
    session_id, login_type, ip4=None, platform=None, line_display_name=None
):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO chat_history
        (username, question, answer, intent, entities, summary, session_id, login_type, ip4, platform, line_display_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (username, question, answer, intent, entities, summary, session_id, login_type, ip4, platform, line_display_name))
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

def export_chat_history_csv():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("SELECT session_id, platform, username, line_display_name, question, answer, timestamp FROM chat_history ORDER BY timestamp ASC")
    rows = c.fetchall()
    conn.close()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    with open(tmp.name, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id', 'platform', 'username', 'line_display_name', 'question', 'answer', 'timestamp'])
        for row in rows:
            writer.writerow(row)
    return tmp.name

# === 向量庫已上傳過的檔案清單 ===
DOCUMENTS_PATH = "./docs"
def get_uploaded_doc_files():
    if not os.path.exists(DOCUMENTS_PATH):
        return []
    return [f for f in os.listdir(DOCUMENTS_PATH) if os.path.isfile(os.path.join(DOCUMENTS_PATH, f))]

# === AI / LLM 部分 ===
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
        response = co.classify(inputs=[question], examples=examples)
        return response.classifications[0].prediction
    except Exception:
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
    except Exception:
        return "[]"

def cohere_generate(prompt):
    response = co.generate(
        model="command-r",
        prompt=prompt,
        max_tokens=128,
        temperature=0.3
    )
    return response.generations[0].text.strip()

def summarize_qa(question, answer):
    prompt = f"Summarize the following conversation in one sentence:\nQ: {question}\nA: {answer}"
    return cohere_generate(prompt)

VECTOR_STORE_PATH = "./faiss_index"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)
embedding_model = CohereEmbeddings(
    cohere_api_key=COHERE_API_KEY,
    model="embed-multilingual-v3.0"
)
llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-r",
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
            # 允许从本地 pickle 文件反序列化
            vectorstore = FAISS.load_local(
                VECTOR_STORE_PATH,
                embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            vectorstore = build_vector_store()
    if qa is None:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 1})
        )

def build_multi_turn_prompt(current_question, session_id):
    history = get_recent_chats(session_id)
    dialog = ""
    for q, a in history:
        dialog += f"User: {q}\nBot: {a}\n"
    dialog += f"User: {current_question}\nBot:"
    return dialog

def ai_chat_llm_only(
    question,
    username="user",
    lang=DEFAULT_LANG,
    platform="gradio",
    line_display_name=None
):
    session_id = username
    login_type = "user"
    ensure_qa()
    style_prefix = STYLE_PROMPT.get(lang, "")
    if style_prefix:
        prompt = f"{style_prefix}\n{build_multi_turn_prompt(question, session_id)}"
    else:
        prompt = build_multi_turn_prompt(question, session_id)
    llm_result = cohere_generate(prompt)
    intent = classify_intent(question)
    entities = extract_entities(question)
    summary = summarize_qa(question, llm_result)
    rag_result = qa.invoke({"query": question})
    save_chat(
        username, question, llm_result, intent, entities, summary,
        session_id, login_type, ip4=None, platform=platform, line_display_name=line_display_name
    )
    return llm_result

def faq_chat_only(
    question,
    username="user",
    lang=DEFAULT_LANG,
    platform="gradio",
    line_display_name=None
):
    ensure_qa()
    rag_result = qa.invoke({"query": question})
    if isinstance(rag_result, dict) and "result" in rag_result:
        return rag_result["result"]
    else:
        return str(rag_result)

# === FastAPI app 須先宣告 ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Gradio 介面組裝 ===
init_db()
with gr.Blocks(title="AI 多語助理") as demo:
    admin_status = gr.State(False)

    with gr.Group(visible=True) as qa_group:
        langs = ["zh-TW", "zh-CN", "en", "ja", "ko"]
        def make_language_tab(lang):
            with gr.Tab(LABELS[lang]["lang"]):
                # AI 回應 tab
                with gr.Tab(LABELS[lang]["ai_qa"]):
                    ai_question = gr.Textbox(label=LABELS[lang]["input_question"])
                    ai_output = gr.Textbox(label=LABELS[lang]["ai_reply"])
                    ai_submit = gr.Button(LABELS[lang]["submit"])
                    def ai_chat_ui(question):
                        return ai_chat_llm_only(question, "user", lang, platform="gradio", line_display_name=None)
                    ai_submit.click(
                        ai_chat_ui,
                        inputs=[ai_question],
                        outputs=ai_output
                    )
                # FAQ（RAG回應）tab
                with gr.Tab(LABELS[lang]["rag_qa"]):
                    faq_question = gr.Textbox(label=LABELS[lang]["input_question"])
                    faq_output = gr.Textbox(label=LABELS[lang]["rag_reply"])
                    faq_submit = gr.Button(LABELS[lang]["submit"])
                    def faq_chat_ui(question):
                        return faq_chat_only(question, "user", lang, platform="gradio", line_display_name=None)
                    faq_submit.click(
                        faq_chat_ui,
                        inputs=[faq_question],
                        outputs=faq_output
                    )
        for lang in langs:
            make_language_tab(lang)
        with gr.Row():
            admin_username = gr.Textbox(label="帳號")
            admin_password = gr.Textbox(label="密碼", type="password")
            admin_login_btn = gr.Button("登入")
            admin_login_status = gr.Textbox(label="", value="未登入", interactive=False)

    with gr.Group(visible=False) as admin_group:
        admin_logout_btn = gr.Button("登出")
        dbsize = gr.Textbox(label="資料庫大小（Bytes）")
        vcount = gr.Textbox(label="向量庫檔案數")
        cpu = gr.Textbox(label="CPU使用率")
        ram = gr.Textbox(label="RAM使用情形")
        disk = gr.Textbox(label="磁碟使用情形")
        uploaded_files_list = gr.Textbox(label="向量資料庫已上傳檔案", interactive=False)
        stats_btn = gr.Button("立即更新狀態")
        def get_stats():
            return [
                str(get_db_size()),
                str(get_vectorstore_file_count()),
                str(psutil.cpu_percent()),
                str(psutil.virtual_memory()._asdict()),
                str(psutil.disk_usage('/')._asdict()),
                "\n".join(get_uploaded_doc_files())
            ]
        stats_btn.click(fn=get_stats, outputs=[dbsize, vcount, cpu, ram, disk, uploaded_files_list])
        update_vec_btn = gr.Button("手動更新向量庫")
        update_status = gr.Textbox(label="向量庫狀態")

        upload_file = gr.File(label="上傳文件（doc, docx, xls, xlsx, pdf, txt）", file_count="multiple")
        upload_status = gr.Textbox(label="狀態")

        homepage_url = gr.Textbox(label="全站首頁網址(含http)")
        homepage_filename = gr.Textbox(label=".url檔名")
        homepage_maxpages = gr.Number(label="最大爬頁數", value=30)
        crawl_btn = gr.Button("用首頁爬子頁並產生 .url")
        crawl_status = gr.Textbox(label="爬蟲狀態")

        sitemap_url = gr.Textbox(label="sitemap.xml網址")
        sitemap_filename = gr.Textbox(label=".url檔名")
        crawl_sitemap_btn = gr.Button("用sitemap自動產生 .url")
        crawl_sitemap_status = gr.Textbox(label="爬蟲狀態")

        # 匯出問答紀錄
        export_btn = gr.Button("一鍵匯出問答紀錄 (CSV)")
        export_file = gr.File(label="下載匯出檔", interactive=True)
        export_btn.click(fn=export_chat_history_csv, inputs=[], outputs=export_file)
                # 手動更新向量庫
        def admin_update_vectorstore():
            try:
                db = build_vector_store()
                global vectorstore, qa
                vectorstore = None
                qa = None
                return "向量資料庫已重新建立"
            except Exception as e:
                return f"重建失敗：{e}"

        update_vec_btn.click(
            admin_update_vectorstore,
            inputs=[],
            outputs=update_status
        )


    def check_login(username, password):
        return username == "admin" and password == "AaAa691027!!"

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

# === Gradio 綁定到 FastAPI ===
from gradio.routes import mount_gradio_app
app = mount_gradio_app(app, demo, path="/gradio")

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/gradio")

# === LINE BOT 整合 ===
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

if not LINE_CHANNEL_SECRET or not LINE_CHANNEL_ACCESS_TOKEN:
    raise RuntimeError("請設定 LINE_CHANNEL_SECRET 與 LINE_CHANNEL_ACCESS_TOKEN 到環境變數")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# === 新增「reply_token 去重」 ===
recent_reply_tokens = {}
recent_reply_tokens_lock = threading.Lock()

def is_duplicate_token(token, expire=180):
    now = time.time()
    with recent_reply_tokens_lock:
        expired = [t for t, ts in recent_reply_tokens.items() if now - ts > expire]
        for t in expired:
            del recent_reply_tokens[t]
        if token in recent_reply_tokens:
            return True
        recent_reply_tokens[token] = now
    return False

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # --- 加入去重邏輯 ---
    if is_duplicate_token(event.reply_token):
        print(f"Skip duplicated event, token: {event.reply_token}")
        return

    user_id = event.source.user_id
    username = f"line_{user_id}"
    try:
        profile = line_bot_api.get_profile(user_id)
        display_name = profile.display_name
    except Exception:
        display_name = None
    user_text = event.message.text
    reply = ai_chat_llm_only(
        user_text,
        username=username,
        lang=DEFAULT_LANG,
        platform="line",
        line_display_name=display_name
    )
    try:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
    except Exception as e:
        print("Reply error:", e)

@app.post("/callback/line")
async def line_callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    body_str = body.decode("utf-8")
    try:
        handler.handle(body_str, signature)
    except InvalidSignatureError:
        return PlainTextResponse("Invalid signature", status_code=400)
    except Exception as e:
        print("LINE handler error:", e)
        # 不管出什麼錯，都回 200，避免 LINE 重送同一個事件，否則無窮回圈
        return PlainTextResponse("OK", status_code=200)
    return PlainTextResponse("OK")
