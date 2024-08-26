import streamlit as st

st.title('🤗💬 Silicus TA')
st.subheader('Sponsored by Blais Foundation')

st.markdown("**Silicus TA** is your personal assistant for ECON 57 and ECON 101. It is a simple web application that leverages the Retrieval Augmented Generation (RAG) prompting technique to help you generate or distil information based on all the documents related to the course. I developed with the hope that it will 1) save you time, 2) clarify concepts covered in class, 3) generate complementary resources (e.g., exercises, explanations, etc...), and 4) provide a conversational interface to interact with the course material that hopefully is fun to play with. Also, an underlying motivation is to help you get used to working with language models. It will prove a valuable skill in the labor market of the next 5 years. I hope you enjoy using it! 🤖📚")

st.markdown("### How to use Silicus TA?")

st.markdown("Use it as you would use any chatbot or search engine. Type in your question or prompt in the chat box and Silicus TA will generate a response based on the documents it has indexed in its knowledge base (i.e., syllabus, my lecture notes, slide decks, previous assigments, etc...).")

st.markdown("The chat feature is designed to be conversational. You can ask follow-up questions, provide feedback, or even ask for clarification. Silicus TA will do its best to provide you with the information you need. Ask it to \n - translate some explanation,\n - generate more exercises related to some specific assignment/concept/lecture,\n - explain learning outcomes,\n - break down the schedule, or\n - about other general inquiries about the post.")

st.markdown("The Data Source Finder uses only one part of the model powering modern LLMs, the embeddings model. I have compiled a list of data sources relevant to economics, wiht accompanying descriptions, that you can use to find data for your projects. Given your query, research question, or topic, the program will search for the top 10 most semantically similar data sources in the database. Finding data is always the most time-consuming part of any project, so I hope this tool will help you find the data you need faster.")