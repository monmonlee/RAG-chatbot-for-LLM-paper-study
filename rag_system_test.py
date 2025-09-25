import os # 作業系統相關功能（讀取環境變數）
import json
from openai import OpenAI # openai api 客戶端
from dotenv import load_dotenv, find_dotenv # dotenv 是專門用來讀取.env套件的套件，並接上環境
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import BaseRetriever, Document
from typing import List, Any
from langchain_openai import ChatOpenAI
from pydantic import Field



def setup_enviorment():
    # 啟動openai api
    # 呼叫data_processing.py已經處理好的資料庫 vectordb
    # 啟動資料庫
    _ = load_dotenv(find_dotenv()) # 讀取.env檔案
    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY']
    )
    
    # activate llm
    llm = ChatOpenAI(model="gpt-3.5-turbo")  

    # activate vectirdb
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings
    )


    if client:
        print("activate all")
        return llm, vectordb
    else:
        print("not workin")
        # return False
        raise RuntimeError("API key activation failed!")


class SmartHybridRetriever(BaseRetriever):  

    # 明確定義fields
    vectordb: Any = Field(default=None)
    llm: Any = Field(default=None)
    
    def __init__(self, vectordb, llm):
        super().__init__()  # 重要：調用父類初始化
        self.vectordb = vectordb
        self.llm = llm
    
    def llm_extract_metadata(self, query):  # 加入self參數
        prompt = f"""
you are a metadata extraction expert, extract year and topic information from user queries.

Available topics:
    - "transformer"
    - "scaling laws"
    - "fine tuning"
    - "LLM limitation"
    - "LLM application"
    - "prompt engineering"
    - "RAG"
    - "LLM agent"  # 修正為"LLM agent"
    - null (if no specific topic)

Available years: 2017-2025 or null

Query = "{query}"

Extract and return ONLY a JSON object:
{{"year":<number or null>, "topic":"<topic or null>"}}

Examples:
Query:"2023年關於transformer的研究"
{{"year":2023, "topic":"transformer"}}

Query:"LLM的限制有哪些"
{{"year":null, "topic":"LLM limitation"}}

Query: "最新的RAG論文"
{{"year": 2025, "topic": "RAG"}}

Now extract from the query above:
"""
        try:
            response = self.llm.invoke(prompt)  # 使用self.llm
            response_text = response.content
            metadata = json.loads(response_text.strip())
            return metadata
        except:
            return {"year": None, "topic": None}

    def smart_hybrid_search(self, query, k=5):  # 加入self參數
        metadata = self.llm_extract_metadata(query)  # 使用self
        year = metadata.get('year')
        topic = metadata.get('topic')

        if year and topic:
            candidates = self.vectordb.max_marginal_relevance_search(query, k=k*4)
            filtered = [doc for doc in candidates
                       if doc.metadata.get('year') == year
                       and doc.metadata.get('topic') == topic]
        elif year:
            candidates = self.vectordb.max_marginal_relevance_search(query, k=k*3)
            filtered = [doc for doc in candidates
                       if doc.metadata.get('year') == year]
        elif topic:
            candidates = self.vectordb.max_marginal_relevance_search(query, k=k*3)
            filtered = [doc for doc in candidates
                       if doc.metadata.get('topic') == topic]
        else:
            filtered = self.vectordb.max_marginal_relevance_search(query, k=k)

        # 修正extend的使用
        if len(filtered) < k:
            additional = self.vectordb.max_marginal_relevance_search(query, k=k)
            filtered.extend([doc for doc in additional if doc not in filtered])  # 修正
        
        return filtered[:k]
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """BaseRetriever要求的抽象方法"""
        return self.smart_hybrid_search(query, k=5)
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """異步版本"""
        return self._get_relevant_documents(query)



def main():
    print("start answering")
    
    try:
        # step1 - setting enviorment
        llm, vectordb =  setup_enviorment()
        print("enviorment setup already")

    except Exception as e:
        print("Setup failed:", e)
        return False
    
    # step2 - activate hybridHybridRetriever
    custom_retriever = SmartHybridRetriever(vectordb=vectordb, llm=llm)
    print(custom_retriever)
if __name__ == "__main__":
    cool = main()