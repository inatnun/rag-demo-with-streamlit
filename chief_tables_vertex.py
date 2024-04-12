from langchain_google_vertexai import VertexAIEmbeddings
import streamlit as st
from streamlit_gsheets import GSheetsConnection
from langchain import hub
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

PROJECT_ID = "skooldio-vertex-ai-demo"
REGION = "asia-southeast1"
# @title Dataset and Table { display-mode: "form" }
DATASET = "my_langchain_dataset"  # @param {type: "string"}
TABLE = "chief_table_dataset"  # @param {type: "string"}

os.environ["GOOGLE_API_KEY"] = st.secrets['gemini_api_key']
# Create a connection object.
conn = st.connection("gsheets", type=GSheetsConnection)
vertexai.init(project=PROJECT_ID, location=REGION)

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

llm = GenerativeModel("gemini-1.5-pro-preview-0409")

embedding = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest", project="skooldio-vertex-ai-demo"
)

vectorstore = BigQueryVectorSearch(
    project_id=PROJECT_ID,
    dataset_name=DATASET,
    table_name=TABLE,
    location=REGION,
    embedding=embedding,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_chief(docs):
    return "\n\n ".join(set(",".join([doc.metadata['episode'], doc.metadata['position'], doc.metadata['url']]) for doc in docs))

def predictOutput(query):
    query_vector = embedding.embed_query(query)
    docs = vectorstore.similarity_search_by_vector(query_vector, k=10)

    promptllm = PromptTemplate(
        template="""
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise and answer in Thai.
    \nQuestion: {question} 
    \nContext: {context} 
    \nAnswer:""", 
        input_variables=["question","context"], 
    )

    # st.write(prompt.format(question=query, context=format_docs(docs)))
    formatPrompt = promptllm.format(question=query, context=format_docs(docs))
    st.write(formatPrompt)
    return llm.generate_content(
        [formatPrompt],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False,
    ).text + "\n\n**ใช้วัตถุดิบในการปรุงแต่งจาก**:\n\n " + get_chief(docs)

    # return llm.predict(promptllm.format(question=query, context=format_docs(docs)),**parameters).text + "\n\n**ใช้วัตถุดิบในการปรุงแต่งจาก**:\n " + get_chief(docs)





st.title("Chief's Table Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("อยากถาม Chiefs ว่า?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
# Streamed response emulator
def response_generator(prompt):
    msg = predictOutput(query=prompt)
    return msg

if prompt is not None:
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write(response_generator(prompt))
    # Display assistant response in chat message container
    st.session_state.messages.append({"role": "assistant", "content": response})