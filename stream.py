import openai
from dotenv import load_dotenv
import os
import json
import streamlit as st
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
import weaviate

load_dotenv()

api_key = os.environ.get('OPENAI_API_KEY')
openai.api_key = api_key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)

client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=auth_config
)

def query_weaviate(ask):
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name="NABH")
    loaded_index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = loaded_index.as_query_engine()
    response = query_engine.query(ask)
    return response

def contract_analysis_w_fact_checking(text):
    if not text:
        st.error("Text field is required in the input data.")
        return

    # Perform contract analysis using query_weaviate (assuming it's a function)
    quert_instance = query_weaviate(text)

    llmresponse = quert_instance.response
    page = quert_instance.source_nodes[0].node.metadata.get('page_label', '')
    file_name = quert_instance.source_nodes[0].node.metadata.get('file_name', '')
    text = quert_instance.source_nodes[0].node.text
    start_char = quert_instance.source_nodes[0].node.start_char_idx
    end_char = quert_instance.source_nodes[0].node.end_char_idx
    score = quert_instance.source_nodes[0].score

    return llmresponse, page, file_name, text, start_char, end_char, score

def main():
    st.title("Easework chat")

    user_message = st.text_input("Enter your text:")
    if st.button("Analyze"):
        llmresponse, page, file_name, text, start_char, end_char, score = contract_analysis_w_fact_checking(user_message)
        st.write(f"LLM Response: {llmresponse}")
        st.write(f"Text: {text}")
        st.write(f"Document Name: {file_name}")
        st.write(f"Page Number: {page}")
        st.write(f"Start Coordination: {start_char}, End Coordination: {end_char}, Score: {score}")

if __name__ == "__main__":
    main()