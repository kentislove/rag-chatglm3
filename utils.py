import os
import glob

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh"
)

creds_content = os.getenv("GOOGLE_CREDENTIALS_JSON")
if creds_content:
    with open("credentials.json", "w") as f:
        f.write(creds_content)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"

os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

def sync_google_drive_files(folder_path):
    """
    同步 Google Drive 指定資料夾內所有檔案到本地 folder_path
    只下載真正的檔案（略過 Google Docs/Sheets/Slides 原生格式）
    """
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    import io

    FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    if not FOLDER_ID:
        print("請設定 GOOGLE_DRIVE_FOLDER_ID 環境變數")
        return

    creds = service_account.Credentials.from_service_account_file(
        "credentials.json",
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )

    service = build("drive", "v3", credentials=creds)
    query = f"'{FOLDER_ID}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    items = results.get("files", [])

    if not items:
        print("Google Drive 資料夾內無檔案")
        return

    for item in items:
        file_id = item["id"]
        file_name = item["name"]
        mime_type = item.get("mimeType", "")
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            print(f"{file_name} 已存在，跳過下載")
            continue

        # 跳過 Google Docs/Sheets/Slides 等原生格式
        if mime_type.startswith("application/vnd.google-apps."):
            print(f"{file_name} 是 Google 文件格式（{mime_type}），略過下載")
            continue

        request = service.files().get_media(fileId=file_id)
        with open(file_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            print(f"已下載 {file_name}")

def load_documents_from_folder(folder_path):
    """
    讀取指定本地資料夾下所有 .txt 檔（可擴充副檔名）
    """
    documents = []
    for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())
    return documents

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
