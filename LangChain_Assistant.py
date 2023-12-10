from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma # Local database

load_dotenv()
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.environ['OPENAI_API_KEY']
)

user_input = "あなたは力持ちですか？"
character="proffesional butler"
user_role="your master"

#  テキストを抽出
loader = TextLoader("./data/profile_butler_jp_2p.txt", encoding="utf-8")
documents = loader.load()
# 30 文字以下で文字を分割 (尚、オーバーラップ 0 文字まで許容) 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# VertexAIEmbeddings を用いて分割したテキストをそれぞれエンべディングして VectorStores に保存
db = Chroma.from_documents(texts, OpenAIEmbeddings())

# user_input に対し類似したテキストをベクター検索して上位 2 つのテキストを取得
# docs = db.similarity_search(user_input, 3)
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8})
docs = retriever.get_relevant_documents(user_input)
# 確認
for index, doc in enumerate(docs):
    print(f"No{index+1}. {doc.page_content}")
# 上位 3 つの回答を結合
facts = "\n".join([doc.page_content for doc in docs])

# プロンプトを作成
system_template = f"Act as a {character}. You are talking with a little girl named イヴェット who is {user_role}. Respond to her chat based on the following facts within 100 words."
human_template = f"""chat:{user_input}
facts:{facts}
"""

assistant = client.beta.assistants.create(
    name="Butler",
    instructions=system_template,
    model="gpt-3.5-turbo"
)
thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=human_template
)

run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id
)

while True:
    run_result = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
    )
    print(run_result.status)
    if run_result.status != "in_progress":
       break

thread_messages = client.beta.threads.messages.list(thread.id)
print(thread_messages)

for m in thread_messages.data:
  print(m.content[0].text.value)