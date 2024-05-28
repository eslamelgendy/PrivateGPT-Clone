import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory   
from langchain.chains import ConversationalRetrievalChain
from htmlTemp import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import os

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_csv_text(csv_docs):
    text = ""
    for csv in csv_docs:
        df = pd.read_csv(csv)
        text += df.to_string(index=False)
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Define available LLM models and their corresponding embedding models
model_pairs = {
    "Default": {
        "llm": HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":0.5, "max_length":512}),
        "embedding": HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    },
    "Turbo": {
        "llm": ChatOpenAI(),
        "embedding": OpenAIEmbeddings()
    }
}


def get_vectorstore(text_chunks,embedding_model):
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embedding_model)
    return vectorstore


def get_conversation_chain(vectorstore,llm_model):
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
    


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            

# Hide Streamlit menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with your Documents",
                       page_icon=":books:", layout="wide", initial_sidebar_state="expanded")
    st.write(css, unsafe_allow_html=True)
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your Documents :books:")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation is None:
            st.warning("Please upload and process your documents first.")
        else:
            handle_userinput(user_question)

    with st.sidebar:

        # Dropdown menu for model pair selection
        model_pair_name = st.selectbox("Select a model", list(model_pairs.keys()))
        selected_pair = model_pairs[model_pair_name]
        st.write('Turbo costs addtional 0.25$ per question')

        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type="pdf")
        
        csv_docs = st.file_uploader(
            "Upload your CSVs here and click on 'Process'", accept_multiple_files=True, type="csv")
        

        if st.button("Process"):
            if pdf_docs or csv_docs:
                with st.spinner("Processing"):
                    raw_text = ""
                    # get pdf text
                    if pdf_docs:  
                        raw_text += get_pdf_text(pdf_docs)

                    # get csv text
                    if csv_docs:
                        raw_text += get_csv_text(csv_docs)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks, selected_pair['embedding'])

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore, selected_pair['llm'])
                    
                    st.success("Processing completed")
            else:
                st.warning("Please upload at least one PDF or CSV document.")


if __name__ == '__main__':
    main()