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
import vertexai
from vertexai.language_models import TextGenerationModel
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
import time

PROJECT_ID = "skooldio-vertex-ai-demo"
REGION = "asia-southeast1"
# @title Dataset and Table { display-mode: "form" }
DATASET = "my_langchain_dataset"  # @param {type: "string"}
TABLE = "chief_table_dataset"  # @param {type: "string"}

os.environ["GOOGLE_API_KEY"] = "AIzaSyCGjhOruoVxcmKe_OdAgolfHJRhTupzoG4"
# Create a connection object.
conn = st.connection("gsheets", type=GSheetsConnection)
vertexai.init(project=PROJECT_ID, location=REGION)
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 2048,
    "temperature": 0.9,
    "top_p": 1
}
llm = TextGenerationModel.from_pretrained("text-bison")

# df = conn.read(usecols=[0,1,2,3],nrows=9)

# # Print results.
# # for row in df.itertuples():
# #     st.write(f"{row.episode} has a :{row.url}:")
    
# loader = DataFrameLoader(df,page_content_column="transcript")
# docs = loader.load()
# # st.write(docs)

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# splits = text_splitter.split_documents(docs)
# st.write(splits)

# embedding = VertexAIEmbeddings(
#     model_name="textembedding-gecko@latest", project="skooldio-vertex-ai-demo"
# )

# vectorstore = BigQueryVectorSearch(
#     project_id=PROJECT_ID,
#     dataset_name=DATASET,
#     table_name=TABLE,
#     location=REGION,
#     embedding=embedding,
#     distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
# )

# all_texts = [d.page_content for d in splits]
# # st.write(all_texts)
# metadatas = [t.metadata for t in splits]
# # # 
# vectorstore.add_texts(all_texts, metadatas=metadatas)

# query = "การเป็นผู้นำที่ดีควรจะเป็นอย่างไร"

# st.write(docs[0].page_content)





# retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={'k': 3})



# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# st.write(rag_chain.invoke("การเป็นผู้นำที่ดีควรจะเป็นอย่างไร"))

# st.write (prompt)

# prompt = "
# You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
# Question: {question} 
# Context: {context} 
# Answer:
# "

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
    return ", ".join(set(doc.metadata['episode'] for doc in docs))

def predictOutput(query):
    query_vector = embedding.embed_query(query)
    docs = vectorstore.similarity_search_by_vector(query_vector, k=3)

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

    return llm.predict(promptllm.format(question=query, context=format_docs(docs)),**parameters).text + "\n\n--- ใช้วัตถุดิบในการปรุงแต่งจาก Chief: " + get_chief(docs)





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
    for word in msg.split():
        yield word + " "
        time.sleep(0.05)

if prompt is not None:
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Display assistant response in chat message container
    st.session_state.messages.append({"role": "assistant", "content": response})