import os
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    WebBaseLoader
)

def load_documents_from_folder(folder_path: str) -> List[Document]:
    docs = []
    for file in os.listdir(folder_path):
        filepath = os.path.join(folder_path, file)
        if file.endswith(".txt"):
            loader = TextLoader(filepath, autodetect_encoding=True)
        elif file.endswith(".pdf"):
            loader = UnstructuredPDFLoader(filepath)
        elif file.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(filepath)
        elif file.endswith(".xlsx") or file.endswith(".xls"):
            loader = UnstructuredExcelLoader(filepath)
        elif file.endswith(".csv"):
            docs.extend(parse_csv_file(filepath))
            continue
        elif file.endswith(".url"):
            with open(filepath, "r", encoding="utf-8") as f:
                urls = [line.strip() for line in f if line.strip()]
                for url in urls:
                    try:
                        web_loader = WebBaseLoader(url)
                        docs.extend(web_loader.load())
                    except Exception as e:
                        print(f"讀取網址失敗 {url}: {e}")
            continue
        else:
            print(f"不支援的格式：{file}")
            continue
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"讀取失敗 {file}: {e}")
    return docs

def parse_csv_file(filepath: str) -> List[Document]:
    import csv
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        for row in reader:
            combined = "\n".join(f"{h}: {r}" for h, r in zip(header, row))
            rows.append(Document(page_content=combined, metadata={"source": filepath}))
    return rows

# ★★★ 新增功能：自動爬網站並產生 .url ★★★

def crawl_links_from_homepage(start_url: str, max_pages=100) -> List[str]:
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin, urlparse

    visited = set()
    to_visit = [start_url]
    domain = urlparse(start_url).netloc
    urls = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            urls.append(url)
            visited.add(url)
            for a in soup.find_all('a', href=True):
                link = urljoin(url, a['href'])
                if urlparse(link).netloc == domain and link not in visited and link not in to_visit:
                    if link.startswith('http'):
                        to_visit.append(link)
        except Exception as e:
            print(f"讀取失敗 {url}: {e}")
    return urls

def fetch_urls_from_sitemap(sitemap_url: str) -> List[str]:
    import requests
    from bs4 import BeautifulSoup

    res = requests.get(sitemap_url)
    res.raise_for_status()
    soup = BeautifulSoup(res.content, "xml")
    urls = [loc.text for loc in soup.find_all("loc")]
    return urls

def save_url_list(urls: List[str], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")
    print(f"共寫入 {len(urls)} 筆網址到 {file_path}")

# 範例調用（若要自動爬）
# urls = crawl_links_from_homepage("https://www.example.com", max_pages=100)
# save_url_list(urls, "./docs/example_homepage.url")
# 或
# urls = fetch_urls_from_sitemap("https://www.example.com/sitemap.xml")
# save_url_list(urls, "./docs/example_sitemap.url")
