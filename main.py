from typing import Union
from contextlib import asynccontextmanager

from fastapi import FastAPI


from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from pymilvus import connections
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.agent import ReActAgent
from llama_index.llms.huggingface import HuggingFaceLLM
import os

os.environ["OPENAI_API_KEY"]= "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"
data = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    connections.connect(
        uri="https://in03-d25b1342170ffed.api.gcp-us-west1.zillizcloud.com/",
        token="6d9471b699afe11d8cd92a79c0bd431edda1f6a6fac849dae4415014d0b22e54ff4ab16c7d829cabe6c93ec85d39987f70e30799"
        )
    index_loaded=False
    if not index_loaded:
        # load data
        dating_profile = SimpleDirectoryReader(
            input_files=["./data/seattle_events.pdf"]
        ).load_data()
    
        vector_store_dp = MilvusVectorStore(dim=1536, collection_name="datingProfiles", overwrite=True)
        storage_context_dp = StorageContext.from_defaults(vector_store=vector_store_dp)
        dp_index = VectorStoreIndex.from_documents(dating_profile, storage_context=storage_context_dp)
        dp_index.storage_context.persist(persist_dir="./storage/datingProfiles")
    
    dp_engine = dp_index.as_query_engine(similarity_top_k=5)

    query_engine_tool = [
    QueryEngineTool(
        query_engine=dp_engine,
        metadata=ToolMetadata(
            name="dp",
            description=(
                "Provides information about events in the are for dating idea"
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    )
]
    
    llm = HuggingFaceLLM(
        context_window=2048,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.25, "do_sample": True},
        # query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="llmware/bling-sheared-llama-2.7b-0.1",
        model_name="llmware/bling-sheared-llama-2.7b-0.1",
        device_map="cpu",
        tokenizer_kwargs={"max_length": 2048},
        # uncomment this if using CUDA to reduce memory usage
        # model_kwargs={"torch_dtype": torch.float16}
        )
    print("i am in startup")
    
    agent = ReActAgent.from_tools(
        query_engine_tool,
        llm=llm,
        verbose=True,
        # context=context
    )
    data["llm"]=agent
    print("i am after agent")

    # response = agent.chat("What sports event can we go to?")
    # print(str(response))





    
    
    yield
    print("Shutdown")
    #Clean up the ML models and release the resources



app = FastAPI(lifespan=lifespan)
# app = FastAPI()


# @app.on_event("startup")
# async def startup_event():
    
#     connections.connect(
#         uri="https://in03-d25b1342170ffed.api.gcp-us-west1.zillizcloud.com/",
#         token="6d9471b699afe11d8cd92a79c0bd431edda1f6a6fac849dae4415014d0b22e54ff4ab16c7d829cabe6c93ec85d39987f70e30799"
#         )
#     index_loaded=False
#     if not index_loaded:
#         # load data
#         dating_profile = SimpleDirectoryReader(
#             input_files=["./data/seattle_events.pdf"]
#         ).load_data()
    
#         vector_store_dp = MilvusVectorStore(dim=1536, collection_name="datingProfiles", overwrite=True)
#         storage_context_dp = StorageContext.from_defaults(vector_store=vector_store_dp)
#         dp_index = VectorStoreIndex.from_documents(dating_profile, storage_context=storage_context_dp)
#         dp_index.storage_context.persist(persist_dir="./storage/datingProfiles")
    
#     dp_engine = dp_index.as_query_engine(similarity_top_k=5)

#     query_engine_tool = [
#     QueryEngineTool(
#         query_engine=dp_engine,
#         metadata=ToolMetadata(
#             name="dp",
#             description=(
#                 "Provides information about events in the are for dating idea"
#                 "Use a detailed plain text question as input to the tool."
#             ),
#         ),
#     )
# ]
    
#     llm = HuggingFaceLLM(
#         context_window=2048,
#         max_new_tokens=256,
#         generate_kwargs={"temperature": 0.25, "do_sample": False},
#         # query_wrapper_prompt=query_wrapper_prompt,
#         tokenizer_name="llmware/bling-sheared-llama-2.7b-0.1",
#         model_name="llmware/bling-sheared-llama-2.7b-0.1",
#         device_map="cpu",
#         tokenizer_kwargs={"max_length": 2048},
#         # uncomment this if using CUDA to reduce memory usage
#         # model_kwargs={"torch_dtype": torch.float16}
#         )
    
#     agent = ReActAgent.from_tools(
#         query_engine_tool,
#         llm=llm,
#         verbose=True,
#         # context=context
#     )




@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/plandate/{date_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    response = data["llm"].chat("What sports event can we go to?")
    print(str(response))
    return response
    #return {"item_id": item_id, "q": q}