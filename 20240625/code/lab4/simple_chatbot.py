import streamlit as st
import os, time
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

def simple_question():    
    """ Ask a simple question """
    openai_client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version = os.getenv("AZURE_OPENAI_API_VERSION"),  
        azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
    )

    question = st.text_input("How can I help you?")
    if st.button("Submit"):
        user_prompt = f"""
You are a chatbot, having a friendly conversation with a human
- Answer in markdown format

USER QUESTION:
{question}

ANSWER:
"""            
        response, completion_time = get_completion(
            openai_client,
            os.getenv("AZURE_OPENAI_CHAT_MODEL"), 
            user_prompt)
        
        response, completion_time

if __name__ == "__main__":
    init_env()
    simple_question()
