import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
# LLM 建議用外部 API 或簡單版本
from langchain.llms import OpenAI  # 假設用 API KEY 跑 GPT-3.5

app = FastAPI()

VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.5, max_tokens=1024)

def build_vector_store():
    # 這裡自己寫文件載入
    documents = []  # TODO: 加入實際文件
    splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=50)
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

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

def rag_answer(question):
    return qa.run(question)

@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()
    user_message = payload.get("message", "")
    reply = rag_answer(user_message)
    return JSONResponse(content={"reply": reply})
