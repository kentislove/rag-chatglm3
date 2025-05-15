import os
import gradio as gr
import psutil
import faiss
import requests
import numpy as np

DOCUMENTS_PATH = "./docs"
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "你的 HuggingFace Token")
EMBEDDING_MODEL = "BAAI/bge-small-zh"

doc_texts = []
doc_vectors = []
index = None

def get_embedding(text):
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    response = requests.post(api_url, headers=headers, json={"inputs": text, "options": {"wait_for_model": True}})
    if response.status_code == 200:
        result = response.json()
        if isinstance(result[0], float):
            return [result]
        return result
    else:
        return [[0.0]*384]  # or raise error

def build_index():
    global doc_texts, doc_vectors, index
    doc_texts = []
    doc_vectors = []
    for fname in os.listdir(DOCUMENTS_PATH):
        if fname.endswith(".txt"):
            with open(os.path.join(DOCUMENTS_PATH, fname), "r", encoding="utf-8") as f:
                doc_texts.append(f.read())
    for text in doc_texts:
        emb = get_embedding(text)[0]
        doc_vectors.append(emb)
    if doc_vectors:
        dim = len(doc_vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(doc_vectors, dtype="float32"))
    else:
        index = None

def print_ram_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    mem_mb = mem_bytes / 1024 / 1024
    print(f"[RAM] 啟動時記憶體佔用：{mem_mb:.2f} MB")
    for mark in range(100, int(mem_mb)+100, 100):
        if mem_mb > mark:
            print(f"[RAM 警告] 記憶體已超過 {mark} MB")
print_ram_usage()

build_index()

def search_answer(query):
    if not doc_texts or index is None:
        return "沒有可搜尋的文件"
    q_emb = get_embedding(query)[0]
    D, I = index.search(np.array([q_emb], dtype="float32"), k=2)
    result = []
    for idx, dist in zip(I[0], D[0]):
        if idx < 0 or idx >= len(doc_texts): continue
        result.append(f"Score: {dist:.2f}\n{doc_texts[idx]}")
    return "\n---\n".join(result) if result else "找不到相關內容"

port = int(os.environ.get("PORT", 10000))
gr.Interface(
    fn=search_answer,
    inputs=gr.Textbox(lines=2, label="請輸入問題"),
    outputs=gr.Textbox(label="最相關內容片段"),
    title="雲端Embedding文件語意搜尋（超省RAM）"
).launch(server_name="0.0.0.0", server_port=port)
