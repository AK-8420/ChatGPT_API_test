from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma # Local database

load_dotenv()
user_input = "あなたの苦手なことは？"

llm = ChatOpenAI(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    model="gpt-3.5-turbo",
)

#  テキストを抽出
loader = TextLoader("./data/profile_butler_jp_2p.txt", encoding="utf-8")
documents = loader.load()
# 30 文字以下で文字を分割 (尚、オーバーラップ 0 文字まで許容) 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# VertexAIEmbeddings を用いて分割したテキストをそれぞれエンべディングして VectorStores に保存
db = Chroma.from_documents(texts, OpenAIEmbeddings())

# user_input に対し類似したテキストをベクター検索して上位 3 つのテキストを取得
docs = db.similarity_search(user_input, 3)
# 確認
for index, doc in enumerate(docs):
    print(f"No{index+1}. {doc.page_content}")
# 上位 3 つの回答を結合
facts = "\n".join([doc.page_content for doc in docs])

# プロンプトのテンプレートを作成
system_template = "Act as a {character} and respond to the chat based on the following facts within 100 words."
human_template = """chat:{text}
facts:{facts}
"""
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", human_template),
])

# 処理を連結
chain = chat_prompt | ChatOpenAI()

response = chain.invoke({
    "character":"proffesional butler", 
    "text": user_input,
    "facts": facts
})
print(response)