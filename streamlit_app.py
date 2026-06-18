import streamlit as st

st.title("RAG Persona Chatbot")

user_query = st.text_input("Ask a question")

if user_query:
    # Call your existing RAG pipeline here
    response = "Your chatbot response"
    st.write(response)