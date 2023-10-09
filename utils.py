from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceHubEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.schema import format_document
from langchain.schema.runnable import RunnableMap
from operator import itemgetter
from typing import Tuple, List
from langchain.docstore.document import Document

import pandas as pd 

#--
# Might be useful for turning sentences into questions
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
#--

# TODO: Add function to store chat history, user info, session info, and feedback in a database (e.g., MongoDB). Let's test with a local json file first.

# --------- Local Utils --------- # 

def load_docs():
    loader = DirectoryLoader("ECON_Files", glob = "**/*.txt")
    econ_docs = loader.load()
    return econ_docs

def load_model(provider):
    if provider == "OpenAI":
        model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.8, max_tokens=300)
    elif provider == "HuggingFace":
        model = HuggingFaceHub(repo_id="google/flan-t5-base", 
                                         model_kwargs={"temperature": 0.6, "max_length": 200})
        
    return model
    
    
def build_chat_chain(provider="OpenAI"):
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
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    
    return chain, vectorstore

def run_hf_chain(message_history, 
                 user_query, chain, vectorstore):
    # Retrieve context
    context = vectorstore.similarity_search(user_query, k=2, return_documents=False)
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
    answer = chain(prompt)
    
    # Compute cost
    cost = compute_cost(len(answer["text"]), "gpt-3.5-turbo")
    
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
    vectorstore = FAISS.from_documents(docs, embeddings)

    
    return vectorstore

    
def compute_cost(tokens, engine):
    """Computes a proxy for the cost of a response based on the number of tokens generated (i.e, cos of output) and the engine used"""
    model_prices = {"text-davinci-003": 0.02, 
                    "gpt-3.5-turbo": 0.002, 
                    "gpt-4": 0.03,
                    "cohere-free": 0}
    model_price = model_prices[engine]
    
    cost = (tokens / 1000) * model_price

    return cost

# Function for generating LLM response
def generate_response(user_query, message_history, index):
    '''
    Function to set smart goal conversationally with LLM.
    params:
        system_prompt: dict
        message_history: str
    return:
        smart_goal: str
    '''
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    
    # design prompt
    system_template = """Use the following pieces of context to answer the users question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer.
    The "SOURCES" part should be a reference to the source of the document from which you got your answer and in the same language.

    And if the user greets with greetings like Hi, hello, How are you, etc reply accordingly as well.

    Example of your response should be:

    The answer is foo
    SOURCES: xyz


    Begin!
    ----------------
    {summaries}"""
    
    # build chat llm
    chatgpt = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.8, max_tokens=300,  stop=["\n", "SOURCES:"])
    
    # build retriever
    retriever = index.as_retriever()
    
    # build chat chain
    chat_chain = ConversationalRetrievalChain.from_llm(chatgpt, retriever=retriever, memory=memory)
    
    # parse messages history which is a list of dictionaries into a string
    chat_history = [f"{message['role']}: {message['content']}" for message in message_history]
    #st.write(chat_history)
    
    current_context = index.similarity_search(user_query, k=2, return_documents=True)
    #st.write(current_context)
    
    # generate response
    answer = chat_chain({"question": user_query, "chat_history": f"{str(chat_history)} - Context: {current_context}"})
    #st.write(answer)
    
    # compute cost
    cost = compute_cost(len(answer["answer"]), "gpt-3.5-turbo")

    return answer, cost