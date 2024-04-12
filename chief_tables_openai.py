from langchain_google_vertexai import VertexAIEmbeddings
import streamlit as st
from streamlit_gsheets import GSheetsConnection
from langchain import hub
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import BigQueryVectorSearch
from langchain.vectorstores.utils import DistanceStrategy
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
import time
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel

PROJECT_ID = "skooldio-vertex-ai-demo"
REGION = "asia-southeast1"
# @title Dataset and Table { display-mode: "form" }
DATASET = "my_langchain_dataset"  # @param {type: "string"}
TABLE = "chief_table_dataset"  # @param {type: "string"}

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
# Create a connection object.
def loadData():
    conn = st.connection("gsheets", type=GSheetsConnection)



    df = conn.read(usecols=[0,1,2,3],nrows=9)

    loader = DataFrameLoader(df,page_content_column="transcript")
    docs = loader.load()
    # st.write(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    splits = text_splitter.split_documents(docs)
    splits




def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def LLM_init():
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings(api_key=openai_api_key))


    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=openai_api_key)
    llmprompt = PromptTemplate(
        template="""
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise and answer in Thai.
        Question: {question} 
        Context: {context} 
        Answer:""", 
        input_variables=["question","context"], 
    )
    # memory = ConversationBufferMemory(memory_key="chat_history")
    
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | llmprompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    
    return rag_chain_with_source

def get_chief(docs):
    return " \n - ".join(set(",".join([doc.metadata['episode'], doc.metadata['position'], doc.metadata['url']]) for doc in docs))

st.title("Chief's Table Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
# Streamed response emulator
def response_generator(prompt):
    llm_chain = LLM_init()
    msg = llm_chain.invoke(prompt)
    ref = get_chief(msg["context"])
    response_msg = msg["answer"]+" \n \n **ใช้วัตถุดิบในการปรุงแต่งจาก**: \n - "+ref
    for line in (response_msg.split("\n")):
        for word in line.split():
            yield word + " "
            time.sleep(0.05)
        yield " \n "
        time.sleep(0.05)
    

if prompt is not None:
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Display assistant response in chat message container
    st.session_state.messages.append({"role": "assistant", "content": response})