# 測試基線的項目不需要包含在這裡

# 問題：兩個class or one class?



import os # 作業系統相關功能（讀取環境變數）
from openai import OpenAI # openai api 客戶端
from dotenv import load_dotenv, find_dotenv # dotenv 是專門用來讀取.env套件的套件，並接上環境
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings




def setup_enviorment():# 還是這個應該要變成init?
    # 啟動openai api
    # 呼叫data_processing.py已經處理好的資料庫 vectordb
    # 啟動資料庫
    _ = load_dotenv(find_dotenv()) # 讀取.env檔案
    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY']
    )
    print("done")
    # activate vectirdb
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)



class # 混合查詢（因為）

    def init

    def llm_extract_metadata(): # 建立解析器

    def smart_hybrid_search(): #混合查詢

    def _get_relevant_documents():

    async def _aget_relevant_documents():





def llm_answer():

# 歷史問答buffer + 自定義prompt + ConversationalRetrievalChain


def main()：


'''
接下來我有想做的事，那就是繼續整理rag的筆記，將剩餘的轉換成rag_system.py
其中包含「啟動環境＆剛剛建立的資料」、「混合檢索（已建立class）」、「llm問答」。

有以下問題：
一、「環境設置」、「llm問答」這兩塊目前有部分沒有變成def，需要轉換之外，他們應該要是「其他的class」還是「合併在混合檢索的class」裡？
二、是否合併在同一個def、class裡有什麼判斷標準？
    目前知道class會是一個物件，具有她自己的「行動」與「屬性」，def是每一個物件會執行的「行動」init是屬性，
    如果今天大家需要共用「屬性」（資料庫、api key）但行動不一樣，這樣是否要分兩個class？
'''


'''
回答：
一、如果有共享的屬性，應該定義在class外面，成為獨立def
二、llm問答應該和class retrival放在同一個class，class retriva是子物件
三、架構應該如下：
    rag_system.py
        def enviorment
        class rag_system
            def init
            class hybrid_retrival
                def init()
                def llm_extract_metadata()
                def smart_hybrid_search()
                def _get_relevant_documents
                async def _aget_relevant_documents
            def query() # 對外接口
            def llm_answer # 私有方法
        def main()

'''

'''
延伸問題：
一、按照以上框架，因為class hybrid_retrival似乎有繼承baseritriever的父類初始化，且有 super().__init__() ，這樣我還可以同步讓他繼承class rag_system的共享資源嗎？而且這個物件似乎規定要有指定以下項目：
    # 明確定義fields
    vectordb: Any = Field(default=None)
    llm: Any = Field(default=None)
二、對外接口是指使用者從ui介面輸入的問題對嗎？因為我目前的研究指導自己打一個question然後請他回答，如下，所以我應該要見一個def架構來傳出使用者發出內容作為回答的參數對嗎？
question = "transformer架構的核心概念是什麼？"
result = qa({"question": question})
print(result['answer'])
三、私有方法指的是我目前開發的「# 歷史問答buffer + 自定義prompt + ConversationalRetrievalChain」對嗎？是否私有的意思就是不會讓外界看到的意思？那為蛇麼混合查詢的物件你沒有說他是「私有方法」？

'''




