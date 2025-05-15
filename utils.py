import os
import io
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader,
    UnstructuredExcelLoader
)

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# 載入 Google Service Account
SERVICE_ACCOUNT_FILE = "credentials.json"
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

def sync_google_drive_files(local_folder: str):
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    service = build('drive', 'v3', credentials=creds)
    query = f"'{FOLDER_ID}' in parents and trashed = false"
    results = service.files().list(q=query, pageSize=50).execute()
    files = results.get('files', [])

    os.makedirs(local_folder, exist_ok=True)

    for file in files:
        file_id = file['id']
        file_name = file['name']
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        file_path = os.path.join(local_folder, file_name)
        with open(file_path, 'wb') as f:
            f.write(fh.getvalue())
        print(f"下載完成：{file_name}")

def load_documents_from_folder(folder_path: str) -> List[Document]:
    docs = []
    for file in os.listdir(folder_path):
        filepath = os.path.join(folder_path, file)
        if file.endswith(".txt"):
            loader = TextLoader(filepath, autodetect_encoding=True)
        elif file.endswith(".pdf"):
            loader = UnstructuredPDFLoader(filepath)
        elif file.endswith(".xlsx") or file.endswith(".xls"):
            loader = UnstructuredExcelLoader(filepath)
        else:
            print(f"不支援的格式：{file}")
            continue
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"讀取失敗 {file}: {e}")
    return docs
