# Import necessary libraries
import streamlit as st
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
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

# Set OpenAI API Key
import os
os.environ["OPENAI_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

loader = DirectoryLoader("ECON_Files", glob = "**\*.txt")
econ_docs = loader.load()
#econ_docs = [format_document(doc) for doc in econ_docs]

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

embeddings = HuggingFaceHubEmbeddings()

text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)

# --------- Local Utils --------- #
def _combine_documents(docs, 
                       document_prompt = DEFAULT_DOCUMENT_PROMPT, 
                       document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def _format_chat_history(chat_history: List[Tuple]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

def index_documents(documents: List[Document], 
                    embeddings: HuggingFaceHubEmbeddings, 
                    text_splitter: TokenTextSplitter):
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
    
    # Split document into chunks
    docs = text_splitter.split_documents(documents)
    
    # Embed chunks
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    return vectorstore
    
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    
def compute_cost(tokens, engine):
    
    model_prices = {"text-davinci-003": 0.02, 
                    "gpt-3.5-turbo": 0.002, 
                    "gpt-4": 0.03,
                    "cohere-free": 0}
    model_price = model_prices[engine]
    
    cost = (tokens / 1000) * model_price

    return cost

# Function for generating LLM response
def generate_response(system_prompt, message_history, retriever):
    '''
    Function to set smart goal conversationally with LLM.
    params:
        system_prompt: dict
        message_history: str
    return:
        smart_goal: str
    '''
    # build chat llm
    chatgpt = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.9, max_tokens=150, 
                         top_p=1, frequency_penalty=0, presence_penalty=0, stop=["\n", "SOURCES:"])
    
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
    


    memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")
    
    # First we add a step to load memory
    # This needs to be a RunnableMap because its the first input
    loaded_memory = RunnableMap(
        {
            "question": itemgetter("question"),
            "memory": memory.load_memory_variables,
        }
    )
    # Next we add a step to expand memory into the variables
    expanded_memory = {
        "question": itemgetter("question"),
        "chat_history": lambda x: x["memory"]["history"]
    }

    # Now we calculate the standalone question
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: _format_chat_history(x['chat_history'])
        } | CONDENSE_QUESTION_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser(),
    }
    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"]
    }
    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question")
    }
    # And finally, we do the part that returns the answers
    answer = {
        "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(),
        "docs": itemgetter("docs"),
    }
    # And now we put it all together!
    final_chain = loaded_memory | expanded_memory | standalone_question | retrieved_documents | answer
    
    result = final_chain.invoke({"question": message_history})
    
    answer = result.get("answer")
    sources = result.get("docs")
    
    cost = compute_cost(len(answer.content), "gpt-3.5-turbo")
    
    return answer, cost
# -------------------------------- #

# Index documents with FAISS
faiss_index = index_documents(econ_docs, embeddings, text_splitter)

# faiss as retriever
retriever = faiss_index.as_retriever()

st.set_page_config(page_title="🤗💬 Silicus TA")


with st.sidebar:
    st.title('🤗💬 Silicus TA')
    st.subheader('Powered by 🤗 Language Models')
    system_prompt = st.text_area("Enter your system prompt here. This will tune the style of output (kind of like the persona of the model).", height=150)
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
            
# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# User-provided prompt
# We isntantiate a new prompt with each chat input
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        st.session_state.current_query = prompt

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, cost = generate_response(system_prompt, st.session_state.current_query, retriever) 
            st.write(answer.content)
            st.session_state.total_cost += cost
            st.sidebar.write(f"Cost of interaction: {cost}")
            st.sidebar.write(f"Total cost: {st.session_state.total_cost}")
    message = {"role": "assistant", "content": answer.content}
    st.session_state.messages.append(message)
    