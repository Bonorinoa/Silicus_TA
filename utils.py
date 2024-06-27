from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import TokenTextSplitter

from langchain_groq import ChatGroq

from operator import itemgetter
from typing import Tuple, List
from langchain.docstore.document import Document

import pandas as pd 
import os 

import streamlit as st

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["HUGGINGFACEHUB_API_KEY"] = st.secrets["HUGGINGFACEHUB_API_KEY"]
os.environ['GROQ_API_KEY'] = st.secrets["GROQ_API_KEY"]

#--

# TODO: Add function to store chat history, user info, session info, and feedback in a database (e.g., MongoDB). Let's test with a local json file first.

# --------- Local Utils --------- # 

@st.cache_data
def load_docs():
    loader = DirectoryLoader("ECON_Files", glob = "**/*.txt")
    econ_docs = loader.load()
    return econ_docs

def compute_cost(tokens, engine):
    """Computes a proxy for the cost of a response based on the number of tokens generated (i.e, cos of output) and the engine used"""
    model_prices = {"gpt-3.5-turbo": 1, 
                    "gpt-4": 50,
                    "cohere-free": 0,
                    "llama3-8b-8192": 0}
    model_price = model_prices[engine]
    
    cost = (tokens / 1_000_000) * model_price

    return cost


def load_model(provider):
    if provider == "GPT-3.5":
        model = ChatOpenAI(model='gpt-3.5-turbo', 
                           temperature=0.8, max_tokens=500)
        
    elif provider == "Llama3":
        model = ChatGroq(model_name="llama3-8b-8192",
                         temperature=0, max_tokens=500)
        
    elif provider == "GPT-4":
        model = ChatOpenAI(model='gpt-4', 
                           temperature=0.8, max_tokens=500)
        
    #elif provider == "HuggingFace":
    #    model = HuggingFaceHub(repo_id="google/flan-t5-base", 
    #                                     model_kwargs={"temperature": 0.6, "max_length": 200})
        
    return model
    
@st.cache_resource
def build_chat_chain(provider="Llama3"):
    # Load documents
    econ_docs = load_docs()

    # Split documents into chunks and index them
    vectorstore = split_and_index_docs(econ_docs)
    
    # load the LLM
    llm = load_model(provider=provider)
    
    # design prompt
    system_template = SystemMessagePromptTemplate.from_template("{system_prompt}")
    human_template = HumanMessagePromptTemplate.from_template("{chat_history}")

    # create the list of messages
    chat_prompt = ChatPromptTemplate.from_messages([
        system_template,
        human_template
    ])

    # Build chain
    chain = LLMChain(llm=llm, 
                     prompt=chat_prompt)
    
    return chain, vectorstore

def run_hf_chain(message_history, 
                 user_query,
                 provider):
    
    chat_chain, vectorstore = build_chat_chain(provider)
    
    # Retrieve context
    context = vectorstore.similarity_search(user_query, k=2, 
                                            return_documents=False)
    
    # Create the complete prompt with conversation history
    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in message_history])
    
    SYS_PROMPT = f"""Use the following pieces of context to answer the users question. 
    Maintain a conversational tone and try to be as helpful as possible. Keep the chat history into account
    
    Chat History:
    {chat_history}
    
    Retrieved_Dcouments:
    {context}
    
    User Query:
    {user_query}
    
    Answer:
    
    """
    
    prompt = {
        'system_prompt': SYS_PROMPT,
        'chat_history': chat_history  
    }
    
    # Generate response
    answer = chat_chain.invoke(prompt)['text']
    
    # Compute cost
    cost = compute_cost(len(answer.split()), 
                        "gpt-3.5-turbo")
    
    return answer, cost
    

def split_and_index_docs(documents: List[Document]):
    '''
    Function to index documents in the vectorstore.
    params:
        documents: list
        vectorstore: FAISS
        embeddings: OpenAIEmbeddings
        text_splitter: TokenTextSplitter
    return:
        None
    '''
    embeddings = OpenAIEmbeddings()

    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=70)
    
    # Split document into chunks
    docs = text_splitter.split_documents(documents)
    
    # Embed chunks
    vectorstore = FAISS.from_documents(docs, 
                                       embeddings)

    
    return vectorstore


