# スレッドを使う変則的な仕様なのにデフォルトのChainを使おうとしているのは良くないプログラム
from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma # Local database
from fastapi import FastAPI
from langserve import add_routes

load_dotenv()
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.environ['OPENAI_API_KEY']
)

#  テキストを抽出
loader = TextLoader("./data/profile_butler_jp_2p.txt", encoding="utf-8")
documents = loader.load()
# 30 文字以下で文字を分割 (尚、オーバーラップ 0 文字まで許容) 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# VertexAIEmbeddings を用いて分割したテキストをそれぞれエンべディングして VectorStores に保存
db = Chroma.from_documents(texts, OpenAIEmbeddings())

#========================
# 1. Chain definition
#========================
def OutputParser(responce):
    return responce[0].data.content[0].text.value

def InputParser(args):
    # user_input に対し類似したテキストをベクター検索して上位 2 つのテキストを取得
    # docs = db.similarity_search(user_input, 3)
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8})
    docs = retriever.get_relevant_documents(args["user_input"])
    # 確認
    for index, doc in enumerate(docs):
        print(f"No{index+1}. {doc.page_content}")
    # 上位 3 つの回答を結合
    facts = "\n".join([doc.page_content for doc in docs])

    # プロンプトを作成
    system_template = f"Act as a {args['character']}. You are talking with a little girl named イヴェット who is {args['user_role']}. Respond to her chat based on the following facts within 100 words."
    human_template = f"""chat:{args["user_input"]}
    facts:{facts}
    """
    return [system_template, human_template]

def ChatThread(prompts):
    assistant = client.beta.assistants.create(
        name="Butler",
        instructions=prompts[0],
        model="gpt-3.5-turbo"
    )
    thread = client.beta.threads.create()

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompts[1]
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
    return thread_messages

category_chain = InputParser() | ChatThread() | OutputParser()

#========================
# 2. App definition
#========================
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

#========================
# 3. Adding chain route
#========================
add_routes(
    app,
    category_chain,
    path="/category_chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)