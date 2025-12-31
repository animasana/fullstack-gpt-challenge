import os
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage.file_system import LocalFileStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.callbacks.base import BaseCallbackHandler
import tempfile
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache


st.set_page_config(
    page_title="Fullsack GPT Challenge Assignment 06",
    page_icon="🤖",
)


history = StreamlitChatMessageHistory()


set_llm_cache(SQLiteCache(database_path="database/documentgpt.db"))


class ChatCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, *args, **kwargs):
        self.message = ""
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        history.add_ai_message(self.message)

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


with st.sidebar:
    OPENAI_API_KEY = st.text_input(
        label="OpenAI API Key",
        type="default",
    )
    file = st.file_uploader(
        "Upload a text file(.txt only)",
        type=["txt"],
    )
    st.write("https://github.com/animasana/fullstack-gpt-challenge/tree/main/assignment06/app.py")


llm = ChatOpenAI(    
    model="gpt-4o-mini",
    streaming=True,    
    callbacks=[
        ChatCallbackHandler(),
    ],
    api_key=OPENAI_API_KEY,
)


@st.cache_resource(show_spinner="Embedding document...")
def embed_file(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name
    
    
    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(    
            chunk_size=5000,
            chunk_overlap=1000,
        )
        loader = TextLoader(tmp_file_path)
        docs = loader.load_and_split(text_splitter=splitter)    
        
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            api_key=OPENAI_API_KEY
        )
        
        vectorstore = FAISS.from_documents(
            documents=docs, 
            embedding=embeddings,
        )
        retriever = vectorstore.as_retriever()
        return retriever
    
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


def send_human_message(message):
    st.chat_message("human").markdown(message)
    history.add_user_message(message)


def paint_history():
    for msg in history.messages:
        st.chat_message(msg.type).markdown(msg.content)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_memory(_):
    return history.messages


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """
            You are a helpful assistant. 
        
            You may also use conversation history to remember user preferences or personal details.
        
            When answering knowledge questions about the document, use ONLY the following context.
        
            If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}
            """
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


st.title("Fullstack GPT Challenge Assignment 06")


if file:
    retriever = embed_file(file)

    st.chat_message("ai").write("I'm ready! Ask away!")
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_human_message(message)
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "history": load_memory
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):        
            chain.invoke(message)        

else:
    history.clear()
