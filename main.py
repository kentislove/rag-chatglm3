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
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from utils import (
    load_documents_from_folder,
    crawl_links_from_homepage,
    fetch_urls_from_sitemap,
    save_url_list
)

import cohere

# === 多語 label 及樣式 ===
LABELS = {
    "zh-TW": {"lang": "繁體中文", "ai_qa": "網路搜尋", "rag_qa": "FAQ搜尋", "input_question": "請輸入問題", "submit": "送出", "rag_reply": "RAG 回應", "ai_reply": "AI 回應", "chat_history": "問答紀錄", "select_session": "選擇對話", "multi_turn_prompt": "多輪對話上下文", "feedback": "回饋", "good": "👍", "bad": "👎", "vector_store_management": "向量庫管理", "uploaded_files": "已上傳檔案", "select_files_to_delete": "選擇要刪除的檔案", "delete_selected": "刪除選取檔案", "rebuild_vector_store": "重建向量庫", "role": "角色", "default_role": "預設", "customer_service": "客服", "psychologist": "心理師", "sales": "業務", "doctor": "醫師"},
    "zh-CN": {"lang": "简体中文", "ai_qa": "网络搜索", "rag_qa": "FAQ搜索", "input_question": "请输入问题", "submit": "提交", "rag_reply": "RAG 回复", "ai_reply": "AI 回复", "chat_history": "问答记录", "select_session": "选择对话", "multi_turn_prompt": "多轮对话上下文", "feedback": "反馈", "good": "👍", "bad": "👎", "vector_store_management": "向量库管理", "uploaded_files": "已上传文件", "select_files_to_delete": "选择要删除的文件", "delete_selected": "删除选取文件", "rebuild_vector_store": "重建向量库", "role": "角色", "default_role": "默认", "customer_service": "客服", "psychologist": "心理师", "sales": "业务", "doctor": "医师"},
    "en":    {"lang": "English",   "ai_qa": "Web Search",  "rag_qa": "FAQ Search",   "input_question": "Type your question here", "submit": "Submit", "rag_reply": "RAG Reply", "ai_reply": "AI Reply", "chat_history": "Chat History", "select_session": "Select Session", "multi_turn_prompt": "Multi-turn Prompt Context", "feedback": "Feedback", "good": "👍", "bad": "👎", "vector_store_management": "Vector Store Management", "uploaded_files": "Uploaded Files", "select_files_to_delete": "Select files to delete", "delete_selected": "Delete Selected", "rebuild_vector_store": "Rebuild Vector Store", "role": "Role", "default_role": "Default", "customer_service": "Customer Service", "psychologist": "Psychologist", "sales": "Sales", "doctor": "Doctor"},
    "ja":    {"lang": "日本語",     "ai_qa": "ウェブ検索",   "rag_qa": "FAQ検索",    "input_question": "質問を入力してください", "submit": "送信", "rag_reply": "RAG返答",  "ai_reply": "AI返答", "chat_history": "チャット履歴", "select_session": "セッションを選択", "multi_turn_prompt": "多ターンプロンプトコンテキスト", "feedback": "フィードバック", "good": "👍", "bad": "👎", "vector_store_management": "ベクトルストア管理", "uploaded_files": "アップロード済みファイル", "select_files_to_delete": "削除するファイルを選択", "delete_selected": "選択したファイルを削除", "rebuild_vector_store": "ベクトルストアを再構築", "role": "役割", "default_role": "デフォルト", "customer_service": "カスタマーサービス", "psychologist": "心理学者", "sales": "営業", "doctor": "医師"},
    "ko":    {"lang": "한국어",     "ai_qa": "웹 검색",     "rag_qa": "FAQ 검색",    "input_question": "질문을 입력하세요",      "submit": "제출", "rag_reply": "RAG 답변", "ai_reply": "AI 답변", "chat_history": "채팅 기록", "select_session": "세션 선택", "multi_turn_prompt": "다중 턴 프롬프트 컨텍스트", "feedback": "피드백", "good": "👍", "bad": "👎", "vector_store_management": "벡터 스토어 관리", "uploaded_files": "업로드된 파일", "select_files_to_delete": "삭제할 파일 선택", "delete_selected": "선택 항목 삭제", "rebuild_vector_store": "벡터 스토어 재구축", "role": "역할", "default_role": "기본", "customer_service": "고객 서비스", "psychologist": "심리학자", "sales": "영업", "doctor": "의사"},
}
DEFAULT_LANG = "zh-TW"
STYLE_PROMPT = {
    "zh-TW": {
        "default": "請以溫暖、貼心、鼓勵、分析、細膩、簡短扼要但不失重點的方式回答，**並以不超過30字**精簡扼要回應：",
        "customer_service": "請以專業、有禮、清晰、簡潔的方式回答，並提供必要的資訊：",
        "psychologist": "請以同理、支持、溫暖、鼓勵的方式回應，並提供正向的觀點：",
        "sales": "請以積極、熱情、專業、有說服力的方式回答，並強調產品或服務的優勢：",
        "doctor": "請以嚴謹、專業、客觀、清晰的方式回答，並提供相關的健康資訊（請強調這不是醫療建議）："
    },
    "zh-CN": {
        "default": "请以温暖、贴心、鼓励、分析、细腻、简短扼要但不失重点的方式回答，**且不超过30字**简短扼要回复：",
        "customer_service": "请以专业、有礼、清晰、简洁的方式回答，并提供必要的信息：",
        "psychologist": "请以同理、支持、温暖、鼓励的方式回应，并提供正向的观点：",
        "sales": "请以积极、热情、专业、有说服力的方式回答，并强调产品或服务的优势：",
        "doctor": "请以严谨、专业、客观、清晰的方式回答，并提供相关的健康信息（请强调这不是医疗建议）："
    },
    "en":    {
        "default": "Please answer in less than 30 words, concise and clear: ",
        "customer_service": "Please answer in a professional, polite, clear, and concise manner, providing necessary information: ",
        "psychologist": "Please respond with empathy, support, warmth, and encouragement, offering positive perspectives: ",
        "sales": "Please answer in an active, enthusiastic, professional, and persuasive manner, highlighting product or service advantages: ",
        "doctor": "Please answer in a rigorous, professional, objective, and clear manner, providing relevant health information (please emphasize this is not medical advice): "
    }
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
        line_display_name TEXT,
        feedback INTEGER DEFAULT 0 -- 0: no feedback, 1: good, -1: bad
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
    # Return the ID of the newly inserted row
    chat_id = c.lastrowid
    conn.close()
    return chat_id

def save_feedback(chat_id, feedback_value):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('UPDATE chat_history SET feedback = ? WHERE id = ?', (feedback_value, chat_id))
    conn.commit()
    conn.close()

def get_recent_chats(session_id, n=5):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('SELECT question, answer FROM chat_history WHERE session_id=? ORDER BY timestamp DESC LIMIT ?', (session_id, n))
    rows = c.fetchall()
    conn.close()
    return rows[::-1]

# New function to get all session IDs
def get_all_session_ids():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('SELECT DISTINCT session_id FROM chat_history ORDER BY session_id')
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]

# New function to get chat history for a specific session
def get_chat_history_by_session(session_id):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    # Include ID for feedback mechanism
    c.execute('SELECT id, question, answer, timestamp, feedback FROM chat_history WHERE session_id=? ORDER BY timestamp ASC', (session_id,))
    rows = c.fetchall()
    conn.close()
    # Format for display, maybe as a list of dicts or tuples
    history = []
    for row in rows:
        history.append({
            "id": row[0],
            "question": row[1],
            "answer": row[2],
            "timestamp": row[3],
            "feedback": row[4] # Add feedback status
        })
    return history


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
    # Include feedback column in export
    c.execute("SELECT session_id, platform, username, line_display_name, question, answer, timestamp, feedback FROM chat_history ORDER BY timestamp ASC")
    rows = c.fetchall()
    conn.close()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    with open(tmp.name, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id', 'platform', 'username', 'line_display_name', 'question', 'answer', 'timestamp', 'feedback'])
        for row in rows:
            writer.writerow(row)
    return tmp.name

# === 向量庫已上傳過的檔案清單 ===
DOCUMENTS_PATH = "./docs"
def get_uploaded_doc_files():
    if not os.path.exists(DOCUMENTS_PATH):
        return []
    return [f for f in os.listdir(DOCUMENTS_PATH) if os.path.isfile(os.path.join(DOCUMENTS_PATH, f))]

# New function to delete files from docs folder
def delete_doc_files(filenames):
    deleted_count = 0
    for filename in filenames:
        file_path = os.path.join(DOCUMENTS_PATH, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    return deleted_count

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
        cohere.ClassifyExample(text="Tell me about your products", label="product_info"),
        cohere.ClassifyExample(text="How to use this feature?", label="how_to"),
        cohere.ClassifyExample(text="What is your price?", label="pricing"),
        cohere.ClassifyExample(text="I have a problem with my account", label="account_issue"),
        cohere.ClassifyExample(text="Where is your store located?", label="location"),
        cohere.ClassifyExample(text="Can I return this item?", label="returns"),
        cohere.ClassifyExample(text="What are your business hours?", label="hours"),
        cohere.ClassifyExample(text="I want to give feedback", label="feedback"),
        cohere.ClassifyExample(text="Hello", label="greeting"),
        cohere.ClassifyExample(text="Thank you", label="gratitude"),
        cohere.ClassifyExample(text="Goodbye", label="farewell"),
    ]
    try:
        response = co.classify(inputs=[question], examples=examples)
        # Return the prediction and confidence score
        if response and response.classifications:
             return response.classifications[0].prediction, response.classifications[0].confidence
        return "unknown", 0.0
    except Exception:
        return "unknown", 0.0

def extract_entities(question):
    import re
    # Check for Chinese characters and skip if present, as the example is English-centric
    if re.search(r'[\u4e00-\u9fff]', question):
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
        return str(response[0].entities) if response and response[0].entities else "[]"
    except Exception:
        return "[]"

def cohere_generate(prompt: str) -> str:
    response = co.chat(
        model="command-r7b-12-2024",
        message=prompt,
        max_tokens=512, # Increased max_tokens for potentially longer responses
        temperature=0.3
    )
    # NonStreamedChatResponse 使用 .message.content 取回內容
    return response.message.content.strip()
def cohere_generate(prompt: str) -> str:
    response = co.chat(
        model="command-r7b-12-2024",
        message=prompt,
        max_tokens=512,
        temperature=0.3
    )
    # 非串流模式下，使用 response.text 屬性取回生成內容
    return response.text.strip()

def summarize_qa(question: str, answer: str) -> str:
    prompt = (
        "Summarize the following conversation in one sentence:\n"
        f"Q: {question}\n"
        f"A: {answer}"
    )
    # Use a smaller model or lower max_tokens for summarization if needed
    return cohere_generate(prompt)

# 路徑準備
VECTOR_STORE_PATH = "./faiss_index"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

# 嵌入模型
embedding_model = CohereEmbeddings(
    cohere_api_key=COHERE_API_KEY,
    model="embed-multilingual-v3.0"
)

# LLM wrapper 使用指定 model
llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-r7b-12-2024",
    temperature=0.3
)

vectorstore = None
qa = None

# Supported document loaders mapping
LOADERS = {
    ".csv": CSVLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".epub": UnstructuredEPubLoader,
    ".html": UnstructuredHTMLLoader,
    ".md": UnstructuredMarkdownLoader,
    ".odt": UnstructuredODTLoader,
    ".pdf": PDFMinerLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".txt": TextLoader,
    # Add more loaders as needed
}

def load_documents_from_folder(folder_path):
    all_documents = []
    if not os.path.exists(folder_path):
        return all_documents
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension in LOADERS:
                try:
                    loader = LOADERS[file_extension](file_path)
                    documents = loader.load()
                    all_documents.extend(documents)
                    print(f"Loaded {len(documents)} documents from {filename}")
                except Exception as e:
                    print(f"Error loading document {filename}: {e}")
            else:
                print(f"Skipping unsupported file type: {filename}")
    return all_documents


def build_vector_store():
    # Clean up existing index before building a new one
    if os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH)
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    documents = load_documents_from_folder(DOCUMENTS_PATH)
    if not documents:
        # If no documents, create an empty vector store or handle appropriately
        # For now, raise an error as the original code did
        raise RuntimeError("No document to build vectorstore.")

    splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=100) # Increased chunk size and overlap
    texts = splitter.split_documents(documents)
    if not texts:
         raise RuntimeError("No text chunks to build vectorstore.")

    db = FAISS.from_documents(texts, embedding_model)
    db.save_local(VECTOR_STORE_PATH)
    return db

def ensure_qa():
    global vectorstore, qa
    if vectorstore is None:
        if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
            # 允许从本地 pickle 文件反序列化
            try:
                vectorstore = FAISS.load_local(
                    VECTOR_STORE_PATH,
                    embedding_model,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading vector store: {e}")
                vectorstore = None # Reset vectorstore if loading fails
        # If loading failed or directory was empty, try to build
        if vectorstore is None:
             try:
                 vectorstore = build_vector_store()
             except RuntimeError as e:
                 print(f"Could not build vector store: {e}")
                 vectorstore = None # Ensure vectorstore is None if build fails

    if qa is None and vectorstore is not None:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}) # Increased k for more context
        )

def build_multi_turn_prompt(current_question, session_id, lang=DEFAULT_LANG, role="default"):
    history = get_recent_chats(session_id)
    dialog = ""
    for q, a in history:
        dialog += f"User: {q}\nBot: {a}\n"
    dialog += f"User: {current_question}\nBot:"

    # Get the style prefix based on language and role
    style_prefix = STYLE_PROMPT.get(lang, {}).get(role, STYLE_PROMPT.get(DEFAULT_LANG, {}).get("default", ""))

    if style_prefix:
        return f"{style_prefix}\n{dialog}"
    else:
        return dialog

def ai_chat_llm_only(
    question,
    username="user",
    lang=DEFAULT_LANG,
    platform="gradio",
    line_display_name=None,
    role="default" # Added role parameter
):
    session_id = username # Using username as session_id for simplicity in Gradio
    login_type = "user"

    # --- Intent Classification and Flow Guidance ---
    intent, confidence = classify_intent(question)
    print(f"Classified intent: {intent} (Confidence: {confidence:.2f})")

    # Simple flow guidance based on intent
    # If intent is likely related to FAQ/knowledge base (e.g., product_info, how_to, pricing, location, returns, hours)
    # and confidence is reasonably high, maybe prioritize RAG?
    # Or, if intent is "about", maybe give a canned response or use RAG?
    # For this implementation, we will use intent to decide if RAG is *also* performed,
    # but the primary response will still come from the LLM, potentially informed by RAG results.
    # A more complex flow would involve conditional UI updates or redirects, which is harder in this structure.

    # For now, let's just log the intent and entities, and proceed with the chat.
    # We can use the intent later for analysis or more complex routing if the UI allows.
    entities = extract_entities(question)
    print(f"Extracted entities: {entities}")

    ensure_qa() # Ensure vector store and QA chain are ready

    # Build the prompt including style and history
    prompt = build_multi_turn_prompt(question, session_id, lang, role)

    llm_result = cohere_generate(prompt)

    # Perform RAG query regardless, maybe the LLM can use it
    # Note: The current RetrievalQA chain is separate from the chat model.
    # A more advanced setup would integrate retrieval into the chat prompt or use a tool.
    # For now, we just run it and could potentially use its result.
    # Let's keep the original logic of just getting the LLM result for the main response.
    # rag_result = qa.invoke({"query": question})
    # print(f"RAG Result: {rag_result}") # Log RAG result

    summary = summarize_qa(question, llm_result)

    # Save chat history and get the chat_id
    chat_id = save_chat(
        username, question, llm_result, intent, entities, summary,
        session_id, login_type, ip4=None, platform=platform, line_display_name=line_display_name
    )

    # Return the LLM result and the chat_id for feedback
    return llm_result, chat_id, prompt # Also return the prompt for visualization

def faq_chat_only(
    question,
    username="user",
    lang=DEFAULT_LANG,
    platform="gradio",
    line_display_name=None
):
    session_id = username # Using username as session_id for simplicity in Gradio
    login_type = "user"

    # Intent and entities for logging, not used for routing in this function
    intent, confidence = classify_intent(question)
    entities = extract_entities(question)
    summary = summarize_qa(question, "N/A") # Summarize question only if no answer yet

    ensure_qa()
    rag_result = qa.invoke({"query": question})
    answer = ""
    if isinstance(rag_result, dict) and "result" in rag_result:
        answer = rag_result["result"]
    else:
        answer = str(rag_result)

    # Save chat history for RAG queries as well
    chat_id = save_chat(
        username, question, answer, intent, entities, summary,
        session_id, login_type, ip4=None, platform=platform, line_display_name=line_display_name
    )

    return answer, chat_id # Return RAG answer and chat_id for feedback

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
    current_chat_id = gr.State(None) # State to hold the ID of the last chat entry for feedback

    with gr.Group(visible=True) as qa_group:
        langs = ["zh-TW", "zh-CN", "en", "ja", "ko"]
        roles = ["default", "customer_service", "psychologist", "sales", "doctor"] # Define available roles

        def make_language_tab(lang):
            with gr.Tab(LABELS[lang]["lang"]):
                # Role selection dropdown
                role_dropdown = gr.Dropdown(
                    label=LABELS[lang]["role"],
                    choices=[(LABELS[lang].get(role_key, role_key), role_key) for role_key in roles],
                    value="default",
                    interactive=True
                )
                # AI 回應 tab
                with gr.Tab(LABELS[lang]["ai_qa"]):
                    ai_question = gr.Textbox(label=LABELS[lang]["input_question"])
                    ai_output = gr.Textbox(label=LABELS[lang]["ai_reply"])
                    multi_turn_viz = gr.Textbox(label=LABELS[lang]["multi_turn_prompt"], interactive=False, lines=5) # Multi-turn visualization
                    ai_submit = gr.Button(LABELS[lang]["submit"])

                    # Feedback buttons for AI response
                    with gr.Row():
                        ai_feedback_good = gr.Button(LABELS[lang]["good"], interactive=False)
                        ai_feedback_bad = gr.Button(LABELS[lang]["bad"], interactive=False)

                    def ai_chat_ui(question, role):
                        # Assume user is "gradio_user" for session ID in Gradio UI
                        answer, chat_id, prompt_viz = ai_chat_llm_only(
                            question,
                            username="gradio_user",
                            lang=lang,
                            platform="gradio",
                            role=role # Pass selected role
                        )
                        # Enable feedback buttons after a response
                        return answer, chat_id, prompt_viz, gr.update(interactive=True), gr.update(interactive=True)

                    ai_submit.click(
                        ai_chat_ui,
                        inputs=[ai_question, role_dropdown], # Pass role dropdown value
                        outputs=[ai_output, current_chat_id, multi_turn_viz, ai_feedback_good, ai_feedback_bad] # Output chat_id and prompt_viz
                    )

                    # Feedback button handlers
                    def handle_feedback(chat_id, feedback_value):
                        if chat_id is not None:
                            save_feedback(chat_id, feedback_value)
                            print(f"Feedback saved for chat ID {chat_id}: {feedback_value}")
                        # Disable buttons after feedback is given (optional)
                        return gr.update(interactive=False), gr.update(interactive=False)

                    ai_feedback_good.click(
                        lambda chat_id: handle_feedback(chat_id, 1), # 1 for good
                        inputs=[current_chat_id],
                        outputs=[ai_feedback_good, ai_feedback_bad]
                    )
                    ai_feedback_bad.click(
                        lambda chat_id: handle_feedback(chat_id, -1), # -1 for bad
                        inputs=[current_chat_id],
                        outputs=[ai_feedback_good, ai_feedback_bad]
                    )


                # FAQ（RAG回應）tab
                with gr.Tab(LABELS[lang]["rag_qa"]):
                    faq_question = gr.Textbox(label=LABELS[lang]["input_question"])
                    faq_output = gr.Textbox(label=LABELS[lang]["rag_reply"])
                    faq_submit = gr.Button(LABELS[lang]["submit"])

                    # Feedback buttons for RAG response
                    with gr.Row():
                        faq_feedback_good = gr.Button(LABELS[lang]["good"], interactive=False)
                        faq_feedback_bad = gr.Button(LABELS[lang]["bad"], interactive=False)

                    def faq_chat_ui(question):
                         # Assume user is "gradio_user" for session ID in Gradio UI
                        answer, chat_id = faq_chat_only(
                            question,
                            username="gradio_user",
                            lang=lang,
                            platform="gradio"
                        )
                        # Enable feedback buttons after a response
                        return answer, chat_id, gr.update(interactive=True), gr.update(interactive=True)

                    faq_submit.click(
                        faq_chat_ui,
                        inputs=[faq_question],
                        outputs=[faq_output, current_chat_id, faq_feedback_good, faq_feedback_bad] # Output chat_id
                    )

                    # Feedback button handlers for RAG
                    faq_feedback_good.click(
                        lambda chat_id: handle_feedback(chat_id, 1), # 1 for good
                        inputs=[current_chat_id],
                        outputs=[faq_feedback_good, faq_feedback_bad]
                    )
                    faq_feedback_bad.click(
                        lambda chat_id: handle_feedback(chat_id, -1), # -1 for bad
                        inputs=[current_chat_id],
                        outputs=[faq_feedback_good, faq_feedback_bad]
                    )

        for lang in langs:
            make_language_tab(lang)

        # === Chat History Interface ===
        with gr.Tab(LABELS[DEFAULT_LANG]["chat_history"]): # Use default lang for tab label
            session_dropdown = gr.Dropdown(
                label=LABELS[DEFAULT_LANG]["select_session"],
                choices=get_all_session_ids(), # Populate with existing session IDs
                interactive=True
            )
            # Use a Dataframe to display history
            history_display = gr.Dataframe(
                headers=["ID", "Timestamp", "Question", "Answer", "Feedback"],
                interactive=False
            )
            load_history_btn = gr.Button("載入紀錄") # Button to trigger loading

            def load_session_history(session_id):
                if not session_id:
                    return []
                history = get_chat_history_by_session(session_id)
                # Format history for Dataframe
                formatted_history = [[h['id'], h['timestamp'], h['question'], h['answer'], h['feedback']] for h in history]
                return formatted_history

            load_history_btn.click(
                load_session_history,
                inputs=[session_dropdown],
                outputs=[history_display]
            )

            # Periodically update session dropdown (optional, could be manual button)
            # demo.load(lambda: gr.Dropdown(choices=get_all_session_ids()), outputs=[session_dropdown])


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

        # === Vector Store Management Interface ===
        with gr.Tab(LABELS[DEFAULT_LANG]["vector_store_management"]):
            uploaded_files_checkboxes = gr.CheckboxGroup(
                label=LABELS[DEFAULT_LANG]["select_files_to_delete"],
                choices=get_uploaded_doc_files(), # Populate with current files
                interactive=True
            )
            delete_files_btn = gr.Button(LABELS[DEFAULT_LANG]["delete_selected"])
            rebuild_vec_btn = gr.Button(LABELS[DEFAULT_LANG]["rebuild_vector_store"])
            vec_management_status = gr.Textbox(label="狀態")

            def update_file_list():
                 files = get_uploaded_doc_files()
                 return gr.CheckboxGroup(choices=files, value=[]) # Update choices and clear selection

            def delete_and_rebuild(filenames):
                if not filenames:
                    return "請選擇要刪除的檔案。", gr.CheckboxGroup(choices=get_uploaded_doc_files(), value=[]) # Update list
                deleted_count = delete_doc_files(filenames)
                status_msg = f"已刪除 {deleted_count} 個檔案。"
                # After deleting, rebuild the vector store
                try:
                    db = build_vector_store()
                    global vectorstore, qa
                    vectorstore = None # Reset to force reload
                    qa = None # Reset to force reload
                    ensure_qa() # Reload/rebuild
                    status_msg += " 向量資料庫已重新建立。"
                except RuntimeError as e:
                    status_msg += f" 重建向量資料庫失敗：{e}"
                except Exception as e:
                     status_msg += f" 重建向量資料庫時發生錯誤：{e}"

                # Update the file list checkbox
                updated_files = get_uploaded_doc_files()
                return status_msg, gr.CheckboxGroup(choices=updated_files, value=[])

            delete_files_btn.click(
                delete_and_rebuild,
                inputs=[uploaded_files_checkboxes],
                outputs=[vec_management_status, uploaded_files_checkboxes]
            )

            def admin_rebuild_vectorstore():
                try:
                    db = build_vector_store()
                    global vectorstore, qa
                    vectorstore = None # Reset to force reload
                    qa = None # Reset to force reload
                    ensure_qa() # Reload/rebuild
                    # Update the file list checkbox after rebuild
                    updated_files = get_uploaded_doc_files()
                    return "向量資料庫已重新建立", gr.CheckboxGroup(choices=updated_files, value=[])
                except RuntimeError as e:
                    # Update the file list checkbox even if rebuild fails
                    updated_files = get_uploaded_doc_files()
                    return f"重建失敗：{e}", gr.CheckboxGroup(choices=updated_files, value=[])
                except Exception as e:
                     # Update the file list checkbox even if rebuild fails
                    updated_files = get_uploaded_doc_files()
                    return f"重建時發生錯誤：{e}", gr.CheckboxGroup(choices=updated_files, value=[])


            rebuild_vec_btn.click(
                admin_rebuild_vectorstore,
                inputs=[],
                outputs=[vec_management_status, uploaded_files_checkboxes] # Also update file list
            )

            # Initial load of file list
            demo.load(update_file_list, outputs=[uploaded_files_checkboxes])


        update_vec_btn = gr.Button("手動更新向量庫") # Keep the old button for simplicity, links to rebuild
        update_status = gr.Textbox(label="向量庫狀態")
        # Link the old button to the new rebuild function
        update_vec_btn.click(
            admin_rebuild_vectorstore,
            inputs=[],
            outputs=[update_status, uploaded_files_checkboxes] # Update status and file list
        )


        upload_file = gr.File(label="上傳文件（doc, docx, xls, xlsx, pdf, txt）", file_count="multiple")
        upload_status = gr.Textbox(label="狀態")

        # Handle file upload
        def handle_upload(files):
            if not files:
                return "請選擇要上傳的檔案。"
            uploaded_count = 0
            for file_obj in files:
                try:
                    # Gradio provides a NamedTemporaryFile object
                    temp_path = file_obj.name
                    filename = os.path.basename(file_obj.orig_name) # Use original filename
                    dest_path = os.path.join(DOCUMENTS_PATH, filename)
                    shutil.copy(temp_path, dest_path)
                    uploaded_count += 1
                except Exception as e:
                    print(f"Error uploading file {file_obj.orig_name}: {e}")
            status_msg = f"成功上傳 {uploaded_count} 個檔案。"
            # After upload, update the file list checkbox
            updated_files = get_uploaded_doc_files()
            return status_msg, gr.CheckboxGroup(choices=updated_files, value=[])

        upload_file.upload(
            handle_upload,
            inputs=[upload_file],
            outputs=[upload_status, uploaded_files_checkboxes] # Update status and file list
        )


        homepage_url = gr.Textbox(label="全站首頁網址(含http)")
        homepage_filename = gr.Textbox(label=".url檔名")
        homepage_maxpages = gr.Number(label="最大爬頁數", value=30)
        crawl_btn = gr.Button("用首頁爬子頁並產生 .url")
        crawl_status = gr.Textbox(label="爬蟲狀態")

        # Handle homepage crawl
        def handle_homepage_crawl(url, filename, max_pages):
            if not url or not filename:
                return "請輸入網址和檔名。"
            try:
                links = crawl_links_from_homepage(url, max_pages)
                save_url_list(links, filename)
                # After crawling, update the file list checkbox
                updated_files = get_uploaded_doc_files()
                return f"成功從 {url} 爬取 {len(links)} 個連結並儲存到 {filename}", gr.CheckboxGroup(choices=updated_files, value=[])
            except Exception as e:
                # Update the file list checkbox even if crawl fails
                updated_files = get_uploaded_doc_files()
                return f"爬取失敗：{e}", gr.CheckboxGroup(choices=updated_files, value=[])

        crawl_btn.click(
            handle_homepage_crawl,
            inputs=[homepage_url, homepage_filename, homepage_maxpages],
            outputs=[crawl_status, uploaded_files_checkboxes] # Update status and file list
        )


        sitemap_url = gr.Textbox(label="sitemap.xml網址")
        sitemap_filename = gr.Textbox(label=".url檔名")
        crawl_sitemap_btn = gr.Button("用sitemap自動產生 .url")
        crawl_sitemap_status = gr.Textbox(label="爬蟲狀態")

        # Handle sitemap crawl
        def handle_sitemap_crawl(url, filename):
            if not url or not filename:
                return "請輸入網址和檔名。"
            try:
                links = fetch_urls_from_sitemap(url)
                save_url_list(links, filename)
                 # After crawling, update the file list checkbox
                updated_files = get_uploaded_doc_files()
                return f"成功從 {url} 爬取 {len(links)} 個連結並儲存到 {filename}", gr.CheckboxGroup(choices=updated_files, value=[])
            except Exception as e:
                 # Update the file list checkbox even if crawl fails
                updated_files = get_uploaded_doc_files()
                return f"爬取失敗：{e}", gr.CheckboxGroup(choices=updated_files, value=[])

        crawl_sitemap_btn.click(
            handle_sitemap_crawl,
            inputs=[sitemap_url, sitemap_filename],
            outputs=[crawl_sitemap_status, uploaded_files_checkboxes] # Update status and file list
        )


        # 匯出問答紀錄
        export_btn = gr.Button("一鍵匯出問答紀錄 (CSV)")
        export_file = gr.File(label="下載匯出檔", interactive=True)
        export_btn.click(fn=export_chat_history_csv, inputs=[], outputs=export_file)

    def check_login(username, password):
        return username == "admin" and password == "AaAa691027!!"

    def do_login(username, password):
        if check_login(username, password):
            # Update session dropdown choices on login
            session_choices = get_all_session_ids()
            return gr.update(visible=False), gr.update(visible=True), "已登入", gr.Dropdown(choices=session_choices)
        else:
            return gr.update(visible=True), gr.update(visible=False), "帳號或密碼錯誤", gr.Dropdown(choices=[]) # Clear dropdown on failed login

    admin_login_btn.click(
        do_login,
        inputs=[admin_username, admin_password],
        outputs=[qa_group, admin_group, admin_login_status, session_dropdown] # Update session dropdown
    )

    def do_logout():
        # Clear session dropdown on logout
        return gr.update(visible=True), gr.update(visible=False), gr.Dropdown(choices=[])
    admin_logout_btn.click(
        do_logout,
        outputs=[qa_group, admin_group, session_dropdown] # Update session dropdown
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
    # Allow running without LINE keys for Gradio testing
    print("LINE_CHANNEL_SECRET or LINE_CHANNEL_ACCESS_TOKEN not set. LINE bot functionality will be disabled.")
    # raise RuntimeError("請設定 LINE_CHANNEL_SECRET 與 LINE_CHANNEL_ACCESS_TOKEN 到環境變數")

if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
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
        username = f"line_{user_id}" # Use line_user_id as session_id for LINE
        try:
            profile = line_bot_api.get_profile(user_id)
            display_name = profile.display_name
        except Exception:
            display_name = None
        user_text = event.message.text

        # LINE bot uses default role and language for now
        # Intent classification is done inside ai_chat_llm_only
        reply, chat_id, _ = ai_chat_llm_only( # We don't need chat_id or prompt_viz for LINE reply
            user_text,
            username=username,
            lang=DEFAULT_LANG, # Use default language for LINE
            platform="line",
            line_display_name=display_name,
            role="default" # Use default role for LINE
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
else:
    # Placeholder if LINE keys are not set
    @app.post("/callback/line")
    async def line_callback_disabled():
        return PlainTextResponse("LINE bot is disabled due to missing environment variables.", status_code=503)
