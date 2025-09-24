## 🎯 專案目的
這個專案主要目的是讓學習LLM相關知識時，可以透過這個聊天機器人來協助學習經典論文，提升研讀效率和理解深度。

## 📖 資料來源
- **學習路線參考**：倢愷 Oscar 於「[【LLM 新手入門】2025 年如何自學 LLM](https://medium.com/@axk51013/llm-%E6%96%B0%E6%89%8B%E5%85%A5%E9%96%80-2025-%E5%B9%B4%E5%A6%82%E4%BD%95%E8%87%AA%E5%AD%B8-llm-a0de380d78eb)」提及的重要論文
- **核心論文集**：DeepLearning.AI 課程「Generative AI with Large Language Models」涵蓋的經典論文
  - Attention Is All You Need (Transformer)
  - BERT, GPT系列論文
  - InstructGPT, ChatGPT相關研究
  - RLHF相關論文等

## 🚀 功能特色
- 🔍 **智能檢索**：基於向量相似度的語義搜尋
- 💬 **互動問答**：針對論文內容的自然語言對話
- 📊 **多論文整合**：跨論文的概念比較和關聯分析
- 🎓 **學習助手**：幫助理解複雜的AI概念和技術細節

## 🛠️ 技術架構
- **Framework**: LangChain
- **LLM**: OpenAI GPT models  
- **Vector DB**: ChromaDB
- **Embedding**: OpenAI Embeddings
- **Processing**: PyPDF, tiktoken
- **Frontend**: Streamlit (開發中)


## 專案結構
your_rag_project/
├── data_processing.py          # 你的草稿程式碼（清理後）
├── rag_system.py              # 你現有的檢索+QA邏輯
├── streamlit_ui.py            # UI介面
├── config.py                  # 設定檔
├── requirements.txt           # 套件清單
├── .env                       # API金鑰
├── chroma_db/                 # 向量資料庫（執行後生成）
├── pdfs/                      # 原始PDF檔案
└── meta_data_correction.csv   # metadata檔案
