from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    model="gpt-3.5-turbo",
)

# プロンプトのテンプレートを作成
system_template = "Act as a {character} and responce to the user's request."
human_template = "{text}"
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", human_template),
])

# 処理を連結
chain = chat_prompt | ChatOpenAI()

response = chain.invoke({
    "character":"proffesional butler", 
    "text": "美味しい紅茶の淹れ方は？"
})
print(response)