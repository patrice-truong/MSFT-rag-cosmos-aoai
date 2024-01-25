import streamlit as st
import os, uuid
from urllib.parse import quote
from datetime import datetime

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory, CosmosDBChatMessageHistory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import AzureCosmosDBVectorSearch

from dotenv import load_dotenv

# Can you list all types of bikes?
# Can you provide more information on mountain bikes?
# List all types of mountain bikes
# Can you provide more details on the Mountain-100?

def init_env():
    load_dotenv()

    st.set_page_config(page_title="CosmicWorks Chatbot", page_icon="🛒")
    st.title("🛒 CosmicWorks Chatbot")

    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")
    os.environ["azure_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
    os.environ["OPENAI_EMBEDDINGS_MODEL_NAME"] = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")

    st.write(f"Azure OpenAI url = {os.getenv('AZURE_OPENAI_ENDPOINT')}")
def main():    
    st.write("Product embeddings are stored in Azure Cosmos DB for MongoDB vCore")
    st.write("Conversations are stored in Azure Cosmos DB for NoSQL")

    # Set up the LLM
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_MODEL"), 
        temperature=0, 
        max_tokens=1000
    )

    # Set up the LLMChain
    template = """You are an AI chatbot having a conversation with a human.

    Human: {human_input}
    AI: """
    prompt = PromptTemplate(input_variables=["human_input"], template=template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Set up the conversation
    if prompt := st.chat_input():
        st.chat_message("human").write(prompt)

        with st.spinner("Please wait.."):
            response = llm_chain.run(prompt)
            
            st.chat_message("ai").write(response)            

def get_cosmosdb_mongodb_connection_string():

    host = os.getenv('COSMOSDB_MONGODB_HOST')
    username = os.getenv('COSMOSDB_MONGODB_USERNAME')
    password = os.getenv('COSMOSDB_MONGODB_PASSWORD')
    encoded_password = quote(password, safe='')

    connection_string = f'mongodb+srv://{username}:{encoded_password}@{host}/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000'

    return connection_string

def calculate_embeddings(query):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        openai_api_version=os.getenv("OPENAI_API_VERSION")
    )
    query_vector = embeddings.embed_query(query)
    return query_vector

def configure_retriever():

    connection_string = get_cosmosdb_mongodb_connection_string()

    embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
            openai_api_version=os.getenv("OPENAI_API_VERSION")
    )

    database_name = os.getenv('COSMOSDB_MONGODB_DATABASE')
    products_collection_name = os.getenv("COSMOSDB_MONGODB_PRODUCTS")    
    namespace = f"{database_name}.{products_collection_name}"
    index_name = products_collection_name + "_vectorindex"
    
    vector_store = AzureCosmosDBVectorSearch.from_connection_string(
        connection_string, 
        namespace, 
        embeddings, 
        index_name=index_name        
    )

    return vector_store

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None
        self.complete = False  # Added flag to track completion

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)

    def on_llm_end(self, response, **kwargs):
        self.complete = True  # Mark completion

class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for doc in documents:
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

def rag():
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )
    # Setup vector store, LLM and QA chain
    vector_store = configure_retriever()

    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_MODEL"), 
        temperature=0, 
        max_tokens=1000
    )

    # Setup the QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, 
        retriever=vector_store.as_retriever(), 
        memory=memory, 
        verbose=True
    )

    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")

    view_messages = st.expander("View the message contents in session state")

    # Render current messages
    for msg in msgs.messages:
        st.chat_message(msg.type).markdown(msg.content, unsafe_allow_html=True)

    # If user inputs a new prompt, generate and draw a new response
    if prompt := st.chat_input():
        st.chat_message("human").markdown(prompt, unsafe_allow_html=True)
        msgs.add_user_message(prompt)

        with st.spinner("Please wait.."):
            retrieval_handler = PrintRetrievalHandler(st.container())
            stream_handler = StreamHandler(st.empty())           

            response = qa_chain.run(
                prompt, 
                callbacks=[retrieval_handler, stream_handler]
            )
            
            st.chat_message("ai").markdown(response, unsafe_allow_html=True)
            msgs.add_ai_message(response)

    # Draw the messages at the end, so newly generated ones show up immediately
    with view_messages:
        view_messages.json(st.session_state.langchain_messages)

def init_cosmos_nosql_history():
    cosmos_endpoint = f"https://{os.getenv('COSMOSDB_NOSQL_ACCOUNT')}.documents.azure.com:443/"
    cosmos_key = os.getenv('COSMOSDB_NOSQL_KEY')
    cosmos_database = os.getenv('COSMOSDB_NOSQL_DATABASE_NAME')
    cosmos_container = os.getenv('COSMOSDB_NOSQL_CONTAINER_NAME')
    cosmos_connection_string = f"AccountEndpoint={cosmos_endpoint};AccountKey={cosmos_key}"

    current_dt = str(datetime.now().strftime("%Y%m%d_%H%M%S"))

    # get user_id from session_state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # get user_id from session_state (in a real app, we would read from authenticated user)
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())

    cosmos_nosql = CosmosDBChatMessageHistory(
        cosmos_endpoint=cosmos_endpoint,
        cosmos_database=cosmos_database,
        cosmos_container=cosmos_container,
        connection_string=cosmos_connection_string,
        session_id=current_dt,
        user_id=st.session_state.user_id
    )
    # prepare the cosmosdb instance
    cosmos_nosql.prepare_cosmos()
    return cosmos_nosql

def rag_with_cosmos_history():

    cosmos_nosql = init_cosmos_nosql_history()

    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        chat_memory=cosmos_nosql,
        return_messages=True
    )
    # Setup vector store, LLM and QA chain
    vector_store = configure_retriever()

    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_MODEL"), 
        temperature=0, 
        max_tokens=1000
    )

    # Setup the QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, 
        retriever=vector_store.as_retriever(), 
        memory=memory, 
        verbose=True
    )

    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")

    view_messages = st.expander("View the message contents in session state")

    # Render current messages
    for msg in msgs.messages:
        st.chat_message(msg.type).markdown(msg.content, unsafe_allow_html=True)

    # If user inputs a new prompt, generate and draw a new response
    if prompt := st.chat_input():
        st.chat_message("human").markdown(prompt, unsafe_allow_html=True)
        msgs.add_user_message(prompt)

        with st.spinner("Please wait.."):
            retrieval_handler = PrintRetrievalHandler(st.container())
            stream_handler = StreamHandler(st.empty())           

            response = qa_chain.run(
                prompt, 
                callbacks=[retrieval_handler, stream_handler]
            )
            
            st.chat_message("ai").markdown(response, unsafe_allow_html=True)
            msgs.add_ai_message(response)

    # Draw the messages at the end, so newly generated ones show up immediately
    with view_messages:
        view_messages.json(st.session_state.langchain_messages)
                

if __name__ == "__main__":
    init_env()
    # main()
    # rag()
    rag_with_cosmos_history()





