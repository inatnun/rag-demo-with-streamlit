import streamlit as st
from langchain_community.retrievers import GoogleVertexAISearchRetriever
from google.oauth2 import service_account
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import time

PROJECT_ID = "skooldio-vertex-ai-demo"  # Set to your Project ID
LOCATION_ID = "global"  # Set to your data store location
REGION = "asia-southeast1"
SEARCH_ENGINE_ID = "skooldio-search_1711534755149"  # Set to your search app ID
DATA_STORE_ID = "skooldio-courses_1711534793759"  # Set to your data store ID

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

retriever = GoogleVertexAISearchRetriever(
    project_id=PROJECT_ID,
    location_id=LOCATION_ID,
    data_store_id=DATA_STORE_ID,
    max_documents=10,
    max_extractive_answer_count=10,
    get_extractive_answers=True,
    engine_data_type=2,
    credentials=credentials
)

llm = GoogleGenerativeAI(model="gemini-pro", credentials=credentials, max_output_tokens=8192, temperature=1, top_p=0.95)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def LLM_init():
    llmprompt = PromptTemplate(
        template="""
    You are a career changing designer who design learning path for career for Workforce of the future. Use the following pieces of retrieved context to design learning path for given role. 
    Keep the answer concise and put the reference source from context metadata name "source".
        Role: {role} 
        Context: {context} 
        Answer:""", 
        input_variables=["role","context"], 
    )
    # memory = ConversationBufferMemory(memory_key="chat_history")
    
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | llmprompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "role": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    
    return rag_chain_with_source


st.title("Skooldio Learning Path")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What role do you want to be?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
# Streamed response emulator
def response_generator(prompt):
    llm_chain = LLM_init()
    msg = llm_chain.invoke(prompt)
    st.write(msg["context"])
    response_msg = msg["answer"]
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