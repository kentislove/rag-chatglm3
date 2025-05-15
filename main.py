import os
import gradio as gr
import psutil
import faiss
from sentence_transformers import SentenceTransformer

VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

# 改用極小模型
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# MODEL_NAME = "all-MiniLM-L6-v2" # 英文用這個，體積最小

# 全域變數
model = None
index = None
doc_texts = []

def load_model():
    global model
    if model is None:
        print("[MODEL] 載入中...")
        model = SentenceTransformer(MODEL_NAME)
    return model

def build_index():
    global index, doc_texts
    doc_texts = []
    for fname in os.listdir(DOCUMENTS_PATH):
        if fname.endswith(".txt"):
            with open(os.path.join(DOCUMENTS_PATH, fname), "r", encoding="utf-8") as f:
                doc_texts.append(f.read())
    if not doc_texts:
        index = None
        return
    model = load_model()
    emb = model.encode(doc_texts)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)

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
    if index is None or not doc_texts:
        return "沒有可搜尋的文件"
    model = load_model()
    q_emb = model.encode([query])
    D, I = index.search(q_emb, k=2)
    result = []
    for idx, dist in zip(I[0], D[0]):
        if idx < 0 or idx >= len(doc_texts): continue
        result.append(f"Score: {dist:.2f}\n{doc_texts[idx]}")
    return "\n---\n".join(result) if result else "找不到相關內容"

gr.Interface(
    fn=search_answer,
    inputs=gr.Textbox(lines=2, label="請輸入問題"),
    outputs=gr.Textbox(label="最相關內容片段"),
    title="最輕量級文件語意搜尋"
).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)))
