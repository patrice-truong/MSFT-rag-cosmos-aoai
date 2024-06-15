import streamlit as st
import os, time
from azure.cosmos import CosmosClient
from tenacity import retry, wait_random_exponential, stop_after_attempt  
from openai import AzureOpenAI
from dotenv import load_dotenv

def init_env():
    st.set_page_config(page_title="CosmicWorks Chatbot", page_icon="🛒")
    st.title("🛒 CosmicWorks Chatbot")

    load_dotenv("..\.env")

    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_EMBEDDING_MODEL"] = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")

def get_completion(openai_client, model, prompt: str):    

    start_time = time.time()
    response = openai_client.chat.completions.create(
        model = model,
        messages = [{"role": "user", "content": prompt}]
    )   
    end_time = time.time()
    elapsed_time = end_time - start_time
    return response.choices[0].message.content, elapsed_time

def init_cosmos():
    COSMOS_DB_ENDPOINT = os.getenv('AZURE_COSMOSDB_NOSQL_ENDPOINT')
    COSMOS_DB_KEY = os.getenv('AZURE_COSMOSDB_NOSQL_KEY')
    DATABASE_NAME = os.getenv('AZURE_COSMOSDB_NOSQL_DATABASE_NAME')
    CONTAINER_NAME = os.getenv('AZURE_COSMOSDB_NOSQL_CONTAINER_NAME')
    client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
    database = client.get_database_client(DATABASE_NAME) 
    products_container = database.get_container_client(CONTAINER_NAME)
    return client, database, products_container

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def generate_embeddings(openai_client, text):
    """
    Generates embeddings for a given text using the OpenAI API v1.x
    """
    
    return openai_client.embeddings.create(
        input = text,
        model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
    ).data[0].embedding

def get_completion(openai_client, model, prompt: str):    
    start_time = time.time()
    response = openai_client.chat.completions.create(
        model = model,
        messages = [{"role": "user", "content": prompt}]
    )   
    end_time = time.time()
    elapsed_time = end_time - start_time
    return response.choices[0].message.content, elapsed_time

def get_similar_docs(openai_client, container, query_text, limit=5):
    """ 
        Get similar documents from Cosmos DB for NoSQL 

        input: 
            container: name of the container
            query_text: user question
            limit: max number of documents to return
        output:
            documents: json documents similar to the user question
            elapsed_time
    """
    # vectorize the question
    query_vector = generate_embeddings(openai_client, query_text)

    # find product keys of products that match the question
    query = f"""
        SELECT TOP {limit} 
            VALUE c.productId
        FROM c 
        WHERE c.type = 'vector'
        ORDER BY VectorDistance(c.embedding, {query_vector})
    """
    start_time = time.time()          

    results = container.query_items(
        query=query,
        parameters=None,
        enable_cross_partition_query=True
    )   

    # get products from list of id
    id_list = [id for id in results]
    
    id_list_str = ', '.join([f"'{id}'" for id in id_list])
    query = f"""
        SELECT * FROM c 
        WHERE c.type = 'product' AND c.productId IN ({id_list_str})
    """
    results = container.query_items(
        query=query,
        enable_cross_partition_query=True
    )

    products = []
    for product in results:
        products.append(product)    

    end_time = time.time()
    elapsed_time = end_time - start_time

    return products, elapsed_time

def main():    
    """ Use vector search in Cosmos DB for NoSQL to answer the question """
    top_k = 10

    questions = [
        "List the categories of bikes that you have",        
        "Can you list all types of mountain bikes?",
        "Can you provide more details on the Mountain-100?",
        "What is your most expensive bike?",
        "Do you have any helmets?"
    ]
    questions
    question = st.text_input("How can I help you?")

    # init Cosmos DB and Azure OpenAI
    client, database, container = init_cosmos()
    openai_client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version = os.getenv("AZURE_OPENAI_API_VERSION"),  
        azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
    )

    if st.button("Submit"):
        with st.spinner("Please wait.."):

            # get similar docs from Cosmos DB
            docs, elapsed_time = get_similar_docs(openai_client, container, question, top_k)

            # pass docs in the context and generate response using Azure OpenAI
            user_prompt = f"""
    Using the following CONTEXT, answer the user's question as best as possible. 
    - Answer in English
    - Answer in markdown format

    CONTEXT:
    {docs}

    USER QUESTION:
    {question}

    ANSWER:
    """            
            response, completion_time = get_completion(openai_client, os.getenv("AZURE_OPENAI_CHAT_MODEL"), user_prompt)
            st.write(response, unsafe_allow_html = True)
            st.write("Elapsed time:", completion_time)

if __name__ == "__main__":
    init_env()
    main()