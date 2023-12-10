from llama_index import StorageContext
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
from llama_index import ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import LLMPredictor
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.logger.base import LlamaLogger
from llama_index.callbacks.base import CallbackManager
from llama_index.indices.list.base import ListRetrieverMode
from llama_index import SimpleDirectoryReader
from llama_index import ListIndex
import os
from dotenv import load_dotenv

load_dotenv()
documents = SimpleDirectoryReader(input_dir="./data").load_data()

# Storage Contextの作成
storage_context = StorageContext.from_defaults(
    docstore=SimpleDocumentStore()
    , vector_store=SimpleVectorStore()
    , index_store=SimpleIndexStore()
)

# Service Contextの作成
llm_predictor = LLMPredictor()
service_context = ServiceContext.from_defaults(
    node_parser=SimpleNodeParser()
    , embed_model=OpenAIEmbedding()
    , llm_predictor=llm_predictor
    , prompt_helper=PromptHelper.from_llm_metadata(llm_metadata=llm_predictor.metadata)
    , llama_logger=LlamaLogger()
    , callback_manager=CallbackManager([])
)

# Index作成時にContextを入れる
list_index = ListIndex.from_documents(
    documents
    , storage_context=storage_context
    , service_context=service_context
)

# Query Engineをインスタンス化
query_engine = list_index.as_query_engine(
    retriever_mode=ListRetrieverMode.DEFAULT,
    node_postprocessors=[]
)

response = query_engine.query("あなたの名前を教えてください。")
print(response.response)
"""
for i in response.response.split("。"):
    print(i + "。")
"""