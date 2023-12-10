from llama_index.agent import OpenAIAssistantAgent
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores import SupabaseVectorStore
from llama_index.tools import QueryEngineTool, ToolMetadata
from dotenv import load_dotenv

load_dotenv()

# load data
reader = SimpleDirectoryReader(input_files=["./data/profile_butler_jp.txt"])
docs = reader.load_data()
for doc in docs:
    doc.id_ = "profile_docs"

vector_store = SupabaseVectorStore(
    postgres_connection_string=(
        "postgresql://<user>:<password>@<host>:<port>/<db_name>" # 自分のに変える必要あり
    ),
    collection_name="base_demo",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

# sanity check that the docs are in the vector store
num_docs = vector_store.get_by_id("profile_docs", limit=1000)
print(len(num_docs))

query_engine_tools = [
    QueryEngineTool(
        query_engine=index.as_query_engine(similarity_top_k=1),
        metadata=ToolMetadata(
            name="profile",
            description=(
                "Provides information about the butler's profile."
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]

agent = OpenAIAssistantAgent.from_new(
    name="Butler",
    model="gpt-3.5-turbo",
    instructions="You are a helpful butler. You treat the user as a girl and educate her as a good lady.",
    tools=query_engine_tools,
    verbose=True,
    run_retrieve_sleep_time=1.0,
)

response = agent.chat("あなたの名前は？")
print(response)