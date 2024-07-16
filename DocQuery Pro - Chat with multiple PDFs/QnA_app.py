import streamlit as st
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import uuid

# ChromaDB configuration
database_path = "Database"
chat_history = "History"

PROMPT_TEMPLATE = """
Answer the question based only on the following context and chat history:

Context:
{context}
--------------------------------------------------------------------------------
Chat History:
{chat_history}

--------------------------------------------------------------------------------

Answer the question based on the above context and chat history: {question}
"""

# Initialize ChromaDB
embedding_function = get_embedding_function()
history_database = Chroma(persist_directory=chat_history, embedding_function=embedding_function)
pdf_database = Chroma(persist_directory=database_path, embedding_function=embedding_function)

def save_chat_history(session_id, user_query, response):
    history_database.add_texts(
        texts=[f"User: {user_query}\nAssistant: {response}"],
        metadatas=[{"session_id": session_id}],
        embedding=embedding_function,
        collection_name=session_id
    )

def get_chat_history(session_id):
    results = history_database.similarity_search_with_score(session_id, k=50)
    history = []
    for res, _score in results:
        doc_metadata = res.metadata
        if "session_id" in doc_metadata and doc_metadata["session_id"] == session_id:
            history.append(res.page_content)
    return history

def query_rag(query_text: str, session_id: str):
    results = pdf_database.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Get complete chat history for the session
    chat_history_list = get_chat_history(session_id)
    chat_history = "\n".join(chat_history_list)
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, chat_history=chat_history, question=query_text)
    
    llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key="Your API key here")
    response = llm.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    # Assuming response is an AIMessage object, extract content appropriately
    formatted_response = {
        "response": response.content,
        "sources": sources
    }
    # print(prompt)
    
    # Save chat history
    save_chat_history(session_id, query_text, response.content)
    
    return formatted_response

st.title("Query System")

# Get session ID from the user
session_id = st.text_input("Enter your session ID:")

if session_id:
    query_text = st.text_input("Enter your query:")

    if query_text and st.button("Query Database"):
        response_data = query_rag(query_text, session_id)
        response_text = response_data["response"]
        sources = response_data["sources"]
        
        st.markdown("### Response")
        st.markdown(response_text)
        
        st.markdown("### Sources")
        st.markdown("\n".join(f"- {source}" for source in sources))
