from llama_index.agent import OpenAIAssistantAgent
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.tools import QueryEngineTool, ToolMetadata
from dotenv import load_dotenv

load_dotenv()

try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./data"
    )
    profile_index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False

if not index_loaded:
    # load data
    profile_docs = SimpleDirectoryReader(
        input_files=["./data/profile_butler_jp.txt"]
    ).load_data()

    # build index
    profile_index = VectorStoreIndex.from_documents(profile_docs)

    # persist index
    profile_index.storage_context.persist(persist_dir="./data")

profile_engine = profile_index.as_query_engine(similarity_top_k=1)

query_engine_tools = [
    QueryEngineTool(
        query_engine=profile_engine,
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
    instructions_prefix="Please address the user as イヴェット.",
    verbose=True,
    run_retrieve_sleep_time=1.0,
)

response = agent.chat("あなたの名前は？")
print(response)