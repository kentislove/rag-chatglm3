import os
from llama_cpp import Llama
import gradio as gr

# ========== 模型路徑 & 自動下載（如需） ==========
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
if not os.path.exists(MODEL_PATH):
    import gdown
    gdown.download(
        "https://drive.google.com/uc?id=1Lm7FI7lpzmN6Rxrii_EQ7Wed4aNjrHIZ",
        MODEL_PATH,
        fuzzy=True
    )

# ========== 初始化本地 LLM（TinyLlama 1.1B） ==========
CONTEXT_SIZE = 1024
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CONTEXT_SIZE,
    n_threads=1,
    n_batch=8,
    verbose=True,
)

# ========== 建立RAG知識檢索組件 ==========
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

def build_vector_store():
    documents = []
    for filename in os.listdir(DOCUMENTS_PATH):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(DOCUMENTS_PATH, filename), encoding="utf-8")
            documents.extend(loader.load())
    # 更小 chunk 以防 context 爆掉
    splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    texts = splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embedding_model)
    db.save_local(VECTOR_STORE_PATH)
    return db

def load_vector_store():
    return FAISS.load_local(VECTOR_STORE_PATH, embedding_model)

if not os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
    vectorstore = build_vector_store()
else:
    vectorstore = load_vector_store()

# ========== 封裝本地 LLM 供 langchain 用 ==========
from langchain.llms import LlamaCpp

llm_chain = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=CONTEXT_SIZE,
    n_threads=1,
    n_batch=8,
    temperature=0.7
)

# 設定 retriever 只返回 1 段 context，避免爆 context window
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

qa = RetrievalQA.from_chain_type(
    llm=llm_chain,
    chain_type="stuff",
    retriever=retriever
)

# ========== 自動摘要（太長就摘要） ==========
def simple_summarize(text, max_len=300):
    # 用最簡單摘要法（選前面 max_len 字），可改成更好的方法
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text

# ========== 自動偵測 context window，分批送出 LLM ==========
def safe_rag_answer(question):
    try:
        result = qa({"query": question}, return_only_outputs=True)
        prompt_and_context = result.get("result", "")
        # 檢查 context 長度，若超過模型最大 n_ctx，則分批摘要或截斷
        if len(prompt_and_context) > CONTEXT_SIZE:
            # 簡單摘要或切片
            prompt_and_context = simple_summarize(prompt_and_context, CONTEXT_SIZE)
            # 用 LLM 再問一次
            out = llm(
                prompt=f"Summarize the following:\n{prompt_and_context}",
                max_tokens=128,
                stop=["</s>"]
            )
            summary = out["choices"][0]["text"]
            return f"內容太長，已自動摘要：\n{summary.strip()}"
        else:
            return prompt_and_context
    except Exception as e:
        return f"錯誤: {e}"

def chat_fn(msg):
    return safe_rag_answer(msg)

gr.Interface(
    fn=chat_fn,
    inputs=gr.Textbox(lines=2, label="請輸入問題"),
    outputs=gr.Textbox(label="AI 回答"),
    title="RAG + Local TinyLlama-1.1B (記憶體防爆/自動摘要/自動分段)"
).launch(server_name="0.0.0.0", server_port=10000)
