import nest_asyncio
nest_asyncio.apply()


import logging
import sys
import os
import io
import asyncio
import streamlit as st
import qdrant_client
import base64
import gc
import tempfile
import uuid
import time


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.llms.ollama import Ollama

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.azure_openai import AzureOpenAI



from llama_index.core import Settings
from workflow import CorrectiveRAGWorkflow
from contextlib import redirect_stdout

from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title="Intellifind", layout="wide")



if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    
if "workflow" not in st.session_state:
    st.session_state.workflow = None
    
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "workflow_logs" not in st.session_state:
    st.session_state.workflow_logs = []



session_id = st.session_state.id


@st.cache_resource
def load_llm():
    
    llm = AzureOpenAI(azure_endpoint= os.environ["AZURE_OPENAI_ENDPOINT"],
                    api_key= os.environ["AZURE_OPENAI_API_KEY"],
                    api_version= os.environ["OPENAI_API_VERSION"],
                    engine= os.environ["AZURE_DEPLOYMENT_NAME"] , 
                    model= os.environ["AZURE_MODEL"]

                )
    
    return llm


    



def reset_chat():
    st.session_state.messages = []
    gc.collect()


def display_pdf(file):
    st.markdown("PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""
    
    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)



def initialize_workflow(file_path):

    with st.spinner("Loading documents and initializing the workflow..."):
        documents = SimpleDirectoryReader(file_path).load_data()

        client = qdrant_client.QdrantClient(
            host="localhost",
            port=6333
        )


        vector_store = QdrantVectorStore(client=client, collection_name="test")
        embed_model = FastEmbedEmbedding(model_name="BAAI/bge-large-en-v1.5")
        Settings.embed_model = embed_model

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )


        workflow = CorrectiveRAGWorkflow(
                index=index,
                linkup_api_key=os.environ["LINKUP_API_KEY"],
                verbose=True, 
                timeout=300,
                llm= load_llm()
        )

        st.session_state.workflow = workflow
        return workflow



async def run_workflow(query):

    f = io.StringIO()
    with redirect_stdout(f):
        result = await st.session_state.workflow.run(query_str=query)

    # Get the captured logs and store them
    logs = f.getvalue()
    if logs:
        st.session_state.workflow_logs.append(logs)
    
    return result



# Sidebar for document upload
with st.sidebar:
    
    col1, col2 = st.columns([1, 3])

    with col1:
        
        st.write("")
        st.image("./assets/linkup.png", width=65)
    
    with col2:
        st.header("Linkup Configuration")
        st.write("Deep Web Search")



# Sidebar for document upload
with st.sidebar:
    
    col1, col2 = st.columns([1, 3])

    with col1:
        
        st.write("")
        st.image("./assets/linkup.png", width=65)
    
    with col2:
        st.header("Linkup Configuration")
        st.write("Deep Web Search")
    


    st.markdown("[Get your API key](https://app.linkup.so/sign-up)", unsafe_allow_html=True)


    linkup_api_key = st.text_input("Enter your Linkup API Key", type="password")



    if linkup_api_key:
        os.environ["LINKUP_API_KEY"] = linkup_api_key
        st.success("API Key stored successfully!")
    

    st.header("Add your documents!")

    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")


    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):

                    # Initialize workflow with the uploaded document
                    workflow = initialize_workflow(temp_dir)

                    st.session_state.file_cache[file_key] = workflow
                else:
                    st.session_state.workflow = st.session_state.file_cache[file_key]
                

                # File Processed! 
                st.success("Ready to Chat!")
                display_pdf(uploaded_file)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()
    


col1, col2 = st.columns([6, 1])


with col1:
    st.markdown("<h2 style='color: #0066cc;'>⚙️ Intellifind (A Corrective RAG agentic workflow)</h2>", unsafe_allow_html=True)

    st.markdown("<div style='display: flex; align-items: center; gap: 10px;'><span style='font-size: 28px; color: #666;'>Powered by LlamaIndex</span><img src='data:image/png;base64,{}' width='50'></div>".format(
        base64.b64encode(open("./assets/llamaindex.jpeg", "rb").read()).decode()
    ), unsafe_allow_html=True)


with col2:
    st.button("Clear ↺", on_click=reset_chat)



# Displaying chat messages from history on app rerun
for i, message in enumerate(st.session_state.messages):

    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
    # Displaying the logs AFTER the user message but BEFORE the next assistant message. 
    if message["role"] == "user" and "log_index" in message and i < len(st.session_state.messages) - 1:
        
        log_index = message["log_index"]
        if log_index < len(st.session_state.workflow_logs):
            with st.expander("View Workflow Execution Logs", expanded=False):
                st.code(st.session_state.workflow_logs[log_index], language="text")




if prompt := st.chat_input("Ask a question about your documents..."):

   
    log_index = len(st.session_state.workflow_logs)
    st.session_state.messages.append({"role": "user", "content": prompt, "log_index": log_index})


    with st.chat_message("user"):
        st.markdown(prompt)
    

    if st.session_state.workflow:
        result = asyncio.run(run_workflow(prompt))
        
    

    if log_index < len(st.session_state.workflow_logs):
        with st.expander("View Workflow Execution Logs", expanded=False):
            st.code(st.session_state.workflow_logs[log_index], language="text")
    



    with st.chat_message("assistant"):
        if st.session_state.workflow:
            message_placeholder = st.empty()
            full_response = ""

            logging.info(f"Result attributes: {dir(result)}")
            logging.info(result.response)

            response_text = result.response


            # Stream the response word by word
            words = response_text.split()
            for i, word in enumerate(words):
                full_response += word + " "
                message_placeholder.markdown(full_response + "▌")

                if i < len(words) - 1:  
                    time.sleep(0.1)
            

           
            message_placeholder.markdown(full_response)
        else:
            full_response = "Please upload a document first to initialize the workflow."
            st.markdown(full_response)
    

    st.session_state.messages.append({"role": "assistant", "content": full_response})



















