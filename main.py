import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI  # 可換成 HuggingFaceInference if 你有 HF API

# 設定環境
VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

# 嵌入模型（小模型，可本地跑）
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")

# LLM 模型（**必須用雲端！本地6B以上不可能**）
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    print("未偵測到 OpenAI API 金鑰，請申請並設置 OPENAI_API_KEY 環境變數。")
    llm = None
else:
    llm = OpenAI(openai_api_key=OPENAI_KEY, temperature=0.3, max_tokens=1024)

# 向量資料庫自動載入/建立
def build_vector_store():
    # 簡化：只讀目錄下所有 txt 文件
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

# 問答鏈（需 LLM）
if llm:
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
else:
    qa = None

# FastAPI 啟動
app = FastAPI()

@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()
    question = payload.get("message", "")
    if not llm:
        return JSONResponse({"reply": "未設定雲端 LLM 服務，請先申請 API KEY。"})
    if not qa:
        return JSONResponse({"reply": "QA 系統尚未初始化。"})
    reply = qa.run(question)
    return JSONResponse({"reply": reply})

# 若要用 Gradio UI，附上（可選）
if __name__ == "__main__":
    import gradio as gr
    def ask_gr(question):
        if not llm:
            return "未設定雲端 LLM 服務，請先申請 API KEY。"
        return qa.run(question)
    gr.Interface(fn=ask_gr, inputs="text", outputs="text", title="RAG 知識庫問答").launch()
