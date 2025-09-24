# 套件
import os 
import pandas as pd
from openai import OpenAI # openai api 客戶端
from dotenv import load_dotenv, find_dotenv # dotenv 是專門用來讀取.env套件的套件，並接上環境
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI


def setup_enviroment():

    # 作業系統相關功能（讀取環境變數）
    _ = load_dotenv(find_dotenv()) # 讀取.env檔案
    client = OpenAI( api_key=os.environ['OPENAI_API_KEY'])
    print("done")

    # 清除舊的環境變數
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
    # 重新載入
    load_dotenv()
    api_key = os.environ.get('OPENAI_API_KEY')

    if api_key:
        print("✅ Success! API key loaded")
        print(f"Key starts with: {api_key[:15]}...")
        print(f"Key length: {len(api_key)} characters")
    else:
        print("❌ Still not working")



def load_metadata_mapping(csv_path):

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip() # df.colums是物件，加上'.str()'轉換成陣列才可以使用.strip()
    # 將filename作為key，其他資訊作為value
    metadata_map = {}
    for _, row in df.iterrows():
        metadata_map[row['filename']] = {
            'title': row['title'],
            'year': row['year'],
            'authors': row['authors'],
            'topic': row['topic']
        }
    return metadata_map

def filter_first_page_only(documents):
    """只保留第一頁（Abstract）"""
    first_page_docs = [doc for doc in documents if doc.metadata['page'] == 0]
    print(f"原始頁數：{len(documents)} → 只保留第一頁：{len(first_page_docs)} 頁")
    return first_page_docs


def load_all_first_pages_with_csv_metadata(folder_path, csv_path):
    """載入PDF第一頁並從CSV對應metadata
    接收兩個參數：folder_path, csv_path
    一、load_metadata_mapping：讀取 csv 裡面的 metadata 對應表
    二、讀取 folder_path：讀取資料夾內的pdf
    三、回圈處理每一份pdf：使用 PyPDFLoader ，並透過 def filter_first_page_only 只保留第一頁
    四、從 csv 對應 metadata
    五、返回「只有第一頁且一定包含指定欄位、以轉成document的pdf」
    """
    
    # 1. 先讀取metadata對應表
    metadata_map = load_metadata_mapping(csv_path)
    print(f"載入metadata對應表，共 {len(metadata_map)} 筆資料")
    
    all_first_pages = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        print(f"處理：{pdf_file}")
        
        file_path = os.path.join(folder_path, pdf_file)
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # 只取第一頁
        first_page_docs = filter_first_page_only(documents)
        
        # 從CSV對應metadata
        for doc in first_page_docs:
            if pdf_file in metadata_map:
                # 找到對應的metadata
                doc.metadata.update(metadata_map[pdf_file])
                print(f"  ✅ 已更新metadata: {metadata_map[pdf_file]['title']}")
            else:
                # 找不到對應資料
                print(f"  ⚠️  警告：{pdf_file} 在CSV中找不到對應資料")
                doc.metadata.update({
                    'title': pdf_file.replace('.pdf', ''),
                    'year': None,
                    'authors': 'Unknown',
                    'topic': 'Unknown'
                })
        all_first_pages.extend(first_page_docs)
    print(f"總共載入 {len(all_first_pages)} 個第一頁，metadata已更新")
    return all_first_pages




def clean_metadata(document): # 只接收單一doc.metadata
    """清理並標準化metadata
    一、接收 def load_all_first_pages_with_csv_metadata 處理完的一頁文檔
    二、指定保留的keys
    三、每份
    """
    keep_keys = ['source', 'total_pages', 'title', 'page', 'year', 'authors', 'topic']
    
    clean_meta = {}
    for key in keep_keys:
        if key in document:
            clean_meta[key] = document[key]
    
    return clean_meta



# all_abstracts_with_metadata會經過資料擷取與欄位清理


def split_document():
    # 分割檔案
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=50,
        separators=[ "\n\n", ". ", "\n", "(?<=\. )", " ", ""]
        )    
    docs = text_splitter.split_documents(all_abstracts_with_metadata)
    print(len(docs)) 
    print(len(all_abstracts_with_metadata)) 

def create_vectordb():
    # 建立資料夾
    os.makedirs('./chroma_db', exist_ok=True)
    print("finish！")

    # 檢查是否成功
    print(f"資料夾存在嗎？{os.path.exists('./chroma_db')}")

    # define embedding
    embeddings = OpenAIEmbeddings()

    # 注意，需要先在自己的環境中建立資料庫路徑
    persist_directory = './chroma_db' # 指定資料庫路徑
    !rm -rf ./chroma_db  # remove old database files if any

    # 建立新的向量資料庫，並將文件放進去
    from langchain.vectorstores import Chroma
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    vectordb.persist() # 手動儲存剛剛建立的資料庫（現在不用手動了，自動儲存）


# 使用 def mian() 整合數據流
    def main():
        """執行主要流程"""
        print("start buliding RAG database...")

        # step1 - setting enviorment
        if not setup_enviroment():
            return False

        # step2 - setting path
        folder_path = "/Users/mangtinglee/Desktop/2025_gap_careerpath/RAG_LLM/pdfs" #這兩塊可能連到github之後要改
        csv_path = "/Users/mangtinglee/Desktop/2025_gap_careerpath/RAG_LLM/meta_data_correction.csv"

            # checking path
        if not os.path.exists(folder_path):
            print(f"not found pdf folder: {folder_path}")
            return False
        
        if not os.path.exists(csv_path):
            print(f"not found pdf folder: {csv_path}")
            return False

        try:
            # step3: loading pdf (data strem starting...)
            print("\n step 1: loading documents")
            documents = load_all_first_pages_with_csv_metadata(folder_path, csv_path)

            # step4: clean matadata
            print('\n step 2: clean metadata')
            clean_documents = clean_metadata(documents)


    all_abstracts_with_metadata = load_all_first_pages_with_csv_metadata(folder_path, csv_path)
    for doc in all_abstracts_with_metadata:
        doc.metadata = clean_metadata(doc.metadata)