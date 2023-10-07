# Import necessary libraries
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub
from langchain.schema.output_parser import StrOutputParser
from utils import format_chat_history, build_chat_chain, run_hf_chain
import time

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    
# Store cumulative cost of interaction
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0

# Store current query (i.e., last user input)
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
    
# Store time spent in session
if "session_time" not in st.session_state:
    st.session_state.session_time = 0
    
# Store feedback
if "feedback" not in st.session_state:
    st.session_state.feedback = ""        
    
session_start_time = time.time()

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# -------------------------------- #

def run_silicus_ta():
    st.set_page_config(page_title="🤗💬 Silicus TA")
    
    chain, vectorstore = build_chat_chain()
    
    with st.sidebar:
        st.title('🤗💬 Silicus TA')
        st.subheader('Powered by 🤗 Language Models')
        #system_prompt = st.text_area("Enter your system prompt here. This will tune the style of output (kind of like the persona of the model).", height=150)
        
        st.sidebar.subheader("Feedback")
        st.sidebar.write("Was the conversation helpful? Your honest feedback will help me improve the system.")
        feedback = st.sidebar.text_area("Feedback", height=150)
        if st.sidebar.button("Submit Feedback"):
            st.session_state.feedback = feedback
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
                

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


    # User-provided prompt
    # We instantiate a new prompt with each chat input
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            st.session_state.current_query = prompt

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                
                answer, cost = run_hf_chain(st.session_state.messages,
                                            st.session_state.current_query, 
                                            chain, vectorstore)
            
                
                st.write(answer["text"])
                st.session_state.total_cost += cost
                st.sidebar.write(f"Cost of interaction: {cost}")
                st.sidebar.write(f"Total cost: {st.session_state.total_cost}")
        message = {"role": "assistant", "content": answer["text"]}
        st.session_state.messages.append(message)
    
if __name__ == "__main__":
    response_start_time = time.time()
    run_silicus_ta()
    response_end_time = time.time()

    print(f"Response time: {response_end_time - response_start_time}")
    print(f"Session time: {response_end_time - session_start_time}")
    
    