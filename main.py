import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import gradio as gr

from utils import sync_google_drive_files, load_documents_from_folder

from langchain.llms.base import LLM

class HuggingFaceInferenceAPI(LLM):
    def __init__(self, api_url, api_token, **kwargs):
        super().__init__(**kwargs)
        self._api_url = api_url   # 用變數，不要寫死
        self._api_token = api_token

    @property
    def _llm_type(self):
        return "custom_hf_api"
    @property
    def api_url(self):
        return self._api_url
    @property
    def api_token(self):
        return self._api_token

    def _call(self, prompt, stop=None):
        import requests
        headers = {"Authorization": f"Bearer {self.api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1024,
                "do_sample": True,
                "temperature": 0.7
            }
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"]
        else:
            return str(result)

# 這裡請改成你想用的 API endpoint，gpt2 一定通，bloomz-560m 也行
api_url = "https://api-inference.huggingface.co/models/bigscience/bloomz-560m"
HUGGINGFACE_API_TOKEN = os.getenv("HF_API_TOKEN")
llm = HuggingFaceInferenceAPI(api_url, HUGGINGFACE_API_TOKEN)

VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")

def build_vector_store():
    sync_google_drive_files(DOCUMENTS_PATH)
    documents = load_documents_from_folder(DOCUMENTS_PATH)
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

qa = RetrievalQA.from_chain_type(
    llm=llm,
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
    title="RAG AI 機器人 (bloomz-560m)"
)

if __name__ == "__main__":
    gradio_app.launch(server_name="0.0.0.0", server_port=10000)
