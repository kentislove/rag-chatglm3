import os
from llama_cpp import Llama
import gradio as gr

# 1. 本地模型路徑（就是你 wget 下來後放的路徑）
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# 2. 初始化本地 LLM（記憶體省用版）
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1024,     # 小機器推薦 1024
    n_threads=1,    # Starter/Pro instance 設 1（如是 Pro 可以設2）
    n_batch=16,     # 省記憶體
    verbose=True,
)

# 3. 下面是 RAG（檢索式生成）必備元件
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# 嵌入模型(可選 bge-small-zh 或英文字向量模型)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")

VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

def build_vector_store():
    # 預設把 ./docs 下的 txt 檔做知識檢索
    documents = []
    for filename in os.listdir(DOCUMENTS_PATH):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(DOCUMENTS_PATH, filename), encoding="utf-8")
            documents.extend(loader.load())
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
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

# 4. 用 langchain 封裝 llama-cpp（可直接丟給 RetrievalQA）
from langchain.llms import LlamaCpp
llm_chain = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=1024,
    n_threads=1,
    n_batch=16,
    temperature=0.7
)

qa = RetrievalQA.from_chain_type(
    llm=llm_chain,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

def rag_answer(question):
    return qa.run(question)

def chat_fn(msg):
    response = rag_answer(msg)
    return response

gradio_app = gr.Interface(
    fn=chat_fn,
    inputs=gr.Textbox(lines=2, label="請輸入問題"),
    outputs=gr.Textbox(label="AI 回答"),
    title="RAG + Local Mistral-7B (llama-cpp-python)"
)

if __name__ == "__main__":
    gradio_app.launch(server_name="0.0.0.0", server_port=10000)
