# misc
import streamlit as st
import os
import time
# Get the Groq library from Langchain frame work
from langchain_groq import ChatGroq
# Get the libraries to load document and transform (text splitter)
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# create a vector store
from langchain_community.vectorstores import FAISS
# get the embeddings vector
from langchain_community.embeddings import OllamaEmbeddings
# get the library to create chains
from langchain.chains.combine_documents import create_stuff_documents_chain
# get chat prompt template
from langchain_core.prompts import ChatPromptTemplate
# get retrieval chain
from langchain.chains import create_retrieval_chain

# Load Env Variables
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]

if "vectors" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://www.geeksforgeeks.org/agents-artificial-intelligence/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# create LLM Model
st.title("ChatGroq Demo")
llm = ChatGroq(groq_api_key = groq_api_key,
               model_name = "mixtral-8x7b-32768")

# create chat prompt template
prompt = ChatPromptTemplate.from_template(
    '''
answer the question based on provided context only.
<context>
{context}
</context>
question:{input}
'''
)

# create a document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# create retriever
retriever = st.session_state.vectors.as_retriever()

# create retriever chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Ask your question ?")

if prompt:
    start  = time.time()
    response = retrieval_chain.invoke({"input":prompt})
    print("Resopnse Time ", time.time()-start)
    st.write(response["answer"])
    
    # with a Streamlit Expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")






