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
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name="NABSH")
    loaded_index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = loaded_index.as_query_engine(include_text=True,
                                                response_mode="tree_summarize",
                                                embedding_mode="hybrid",
                                                similarity_top_k=5,)
    message_template = f"""<|system|>Please check if the following pieces of context has any 
    mention of the keywords provided in the Question and please try to give correct answer 
    if user ask simple answer then give simple answer otherwise he asked summary then give '
    summary and produce accurate and clean source node which is readable from starting to end</s>
    <|user|>
    Question: {ask}
    Helpful Answer:
    </s>"""
    print("Done 2")
    response = query_engine.query(message_template)
    print("Done 4")
    return response


def contract_analysis_w_fact_checking(text):
    if not text:
        st.error("Text field is required in the input data.")
        return

    # Perform contract analysis using query_weaviate (assuming it's a function)
    quert_instance = query_weaviate(text)
    llmresponse = quert_instance.response
    #node 1
    page1 = quert_instance.source_nodes[0].node.metadata.get('page_label', '')
    file_name1 = quert_instance.source_nodes[0].node.metadata.get('file_name', '')
    text1 = quert_instance.source_nodes[0].node.text
    start_char1 = quert_instance.source_nodes[0].node.start_char_idx
    end_char1 = quert_instance.source_nodes[0].node.end_char_idx
    score1 = quert_instance.source_nodes[0].score
    #node 2
    page2 = quert_instance.source_nodes[1].node.metadata.get('page_label', '')
    file_name2 = quert_instance.source_nodes[1].node.metadata.get('file_name', '')
    text2 = quert_instance.source_nodes[1].node.text
    start_char2 = quert_instance.source_nodes[1].node.start_char_idx
    end_char2 = quert_instance.source_nodes[1].node.end_char_idx
    score2 = quert_instance.source_nodes[1].score

    return llmresponse, page1, file_name1, text1, start_char1, end_char1, score1, page2, file_name2, text2, start_char2, end_char2, score2

def main():
    st.title("Easework chat")

    user_message = st.text_input("Enter your text:")
    if st.button("Analyze"):
        llmresponse, page1, file_name1, text1, start_char1, end_char1, score1, page2, file_name2, text2, start_char2, end_char2, score2 = contract_analysis_w_fact_checking(user_message)
        st.write(f"LLM Response: {llmresponse}")
        st.subheader("Source_Node 1")
        st.write(f"Text: {text1}")
        st.write(f"Document Name: {file_name1}")
        st.write(f"Page Number: {page1}")
        st.write(f"Start Coordination: {start_char1}, End Coordination: {end_char1}, Score: {score1}")
        st.subheader("Source Node 2")
        st.write(f"Text: {text2}")
        st.write(f"Document Name: {file_name2}")
        st.write(f"Page Number: {page2}")
        st.write(f"Start Coordination: {start_char2}, End Coordination: {end_char2}, Score: {score2}")

if __name__ == "__main__":
    main()