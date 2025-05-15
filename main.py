import os
import gradio as gr
import psutil

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")

def build_vector_store():
    docs = []
    for fname in os.listdir(DOCUMENTS_PATH):
        if fname.endswith(".txt"):
            with open(os.path.join(DOCUMENTS_PATH, fname), "r", encoding="utf-8") as f:
                docs.append(f.read())
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = splitter.create_documents(docs)
    db = FAISS.from_documents(texts, embedding_model)
    db.save_local(VECTOR_STORE_PATH)
    return db

def load_vector_store():
    return FAISS.load_local(VECTOR_STORE_PATH, embedding_model)

if os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
    vectorstore = load_vector_store()
else:
    vectorstore = build_vector_store()

def print_ram_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    mem_mb = mem_bytes / 1024 / 1024
    print(f"[RAM] 啟動時記憶體佔用：{mem_mb:.2f} MB")
    for mark in range(100, int(mem_mb)+100, 100):
        if mem_mb > mark:
            print(f"[RAM 警告] 記憶體已超過 {mark} MB")
print_ram_usage()

def search_answer(query):
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=2)
    if not docs_and_scores:
        return "找不到相關內容"
    result = "\n---\n".join(
        f"Score: {score:.2f}\n{doc.page_content}" for doc, score in docs_and_scores
    )
    return result

gr.Interface(
    fn=search_answer,
    inputs=gr.Textbox(lines=2, label="請輸入問題"),
    outputs=gr.Textbox(label="最相關內容片段"),
    title="知識文件相似搜尋（不含AI自動回答）"
).launch(server_name="0.0.0.0", server_port=10000)
