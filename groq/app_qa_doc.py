# Misc Libs
import streamlit as st
import os
import time
# import llm from groq
from langchain_groq import ChatGroq
# get embeddings from openAI
from langchain_openai import OpenAIEmbeddings
# Data Ingestion
from langchain_community.document_loaders import PyPDFDirectoryLoader
# Data Transformation
from langchain.text_splitter import RecursiveCharacterTextSplitter
# get chat prompt
from langchain.prompts import ChatPromptTemplate
# Get vector store
from langchain_community.vectorstores import FAISS
# get chain
from langchain.chains import create_retrieval_chain
# import document chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# Load Env Variables
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.environ["GROQ_API_KEY"]

#---------------------------------------------------------------------------#
st.title("ChatGroq with LLAMA3 Demo")
#---------------------------------------------------------------------------#
#-------- Create LLM -------------------------------------------------------#
llm = ChatGroq(groq_api_key = groq_api_key,
               model_name = "Llama3-8b-8192")
#---------------------------------------------------------------------------#
#-------- Create CHAT Prompt -----------------------------------------------#
prompt = ChatPromptTemplate.from_template('''
answer the question based on provided context only.
<context>{context}</context>
question:{input}
'''
)
prompt_input = st.text_input("Ask your question from documents?")
#---------------------------------------------------------------------------#
#-------- Function to load documents in vectors-----------------------------#
def vector_embeddings():
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
#---------------------------------------------------------------------------#

#-------- Create a button for embedding-------------------------------------#
# if st.button("Documents Embedding"):
#     vector_embeddings()
#     st.write("Vector Store DB is ready")
#---------------------------------------------------------------------------#

#---------- Create retrieval chain -----------------------------------------#
if "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
else:
    vector_embeddings()
    st.write("Vector Store DB is ready")     
#---------------------------------------------------------------------------#

#-------- Check for prompt input--------------------------------------------#
if prompt_input:
    start = time.time()
    response = retrieval_chain.invoke({"input":prompt_input})
    resopnse_time = time.time()-start
    st.write("Time taken for the query to run {:.2f} Seconds.".format(resopnse_time))
    st.write(response["answer"])
    # with a Streamlit Expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
            break    

