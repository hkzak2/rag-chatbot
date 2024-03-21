import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
import tempfile


# retrival
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()

# set page config
st.set_page_config(page_title="Chatbot", page_icon=":ghost:", layout="wide")

st.title('Chatbot :penguin:')

st.session_state.rag = False

# load documents
uploaded_file = st.sidebar.file_uploader("Upload File", type="txt")
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    documents = TextLoader(
        tmp_file_path,
    ).load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    print(f"split  {len(documents)} documents into {len(texts)} chunks")

    st.session_state.rag = True

# initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input('Ask your question here....')

llm_provideer = st.sidebar.selectbox("Select the model", ["OpenAI", "Ollama"])

# ai response
def get_ai_response(user_input, chat_history):
    template = """
    You are a helpful assistant, your task is to answer user questions based on the provided context and chat history.

    context: {context}

    chat history: {chat_history}

    user question: {user_input}
    """

    prompt = ChatPromptTemplate.from_template(template)

    if llm_provideer == "OpenAI":
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    else:
        llm = ChatOllama(model="llama2:7b-chat")
        embeddings = OllamaEmbeddings(model="llama2:7b-chat")

    context = None

    if st.session_state.rag:
        retriever = FAISS.from_documents(texts, embeddings)
        # similarity search
        context = retriever.similarity_search(user_input, top_k=1)
        print(context)

    chain = prompt | llm | StrOutputParser()


    return chain.stream({
        "context": context,
        "chat_history": chat_history,
        "user_input": user_input  
    })


# chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
       with st.chat_message("Human"):
           st.markdown(message.content)
    else :
        with st.chat_message("AI"):
            st.markdown(message.content)

# user input
if user_input is not None and user_input != "":
    st.session_state.chat_history.append(HumanMessage(user_input))

    with st.chat_message("Human"):
        st.markdown(user_input)

    with st.chat_message("AI"):
        # st.markdown("I am a bot, I am still learning. I will get back to you soon..."
        ai_response = st.write_stream(get_ai_response(user_input, st.session_state.chat_history))
        
   
    st.session_state.chat_history.append(AIMessage(ai_response))