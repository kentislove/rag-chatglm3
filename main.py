import os
from llama_cpp import Llama
import gradio as gr

# 1. 本地模型路徑
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
if not os.path.exists(MODEL_PATH):
    import gdown
    gdown.download(
        "https://drive.google.com/uc?id=1Lm7FI7lpzmN6Rxrii_EQ7Wed4aNjrHIZ",
        MODEL_PATH,
        fuzzy=True
    )
# 2. 初始化本地 LLM（TinyLlama 1.1B，超省RAM）
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_threads=1,
    n_batch=8,    # TinyLlama 超省RAM，這樣設一般2GB夠用
    verbose=True,
)

# 3. 建立RAG知識檢索組件
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# 4. 嵌入模型選MiniLM
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
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=60)
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

# 5. 用 langchain 封裝本地llama-cpp-python
from langchain.llms import LlamaCpp

llm_chain = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=1024,
    n_threads=1,
    n_batch=8,
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
    try:
        return rag_answer(msg)
    except Exception as e:
        return f"錯誤: {e}"

gr.Interface(
    fn=chat_fn,
    inputs=gr.Textbox(lines=2, label="請輸入問題"),
    outputs=gr.Textbox(label="AI 回答"),
    title="RAG + Local TinyLlama-1.1B"
).launch(server_name="0.0.0.0", server_port=10000)
