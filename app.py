import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate, MessagesPlaceholder,ChatPromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()

    model_id = "sentence-transformers/all-mpnet-base-v2"
    # model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceInstructEmbeddings(model_name=model_id)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore




def get_conversation_chain(vectorstore):

    llm = HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", temperature=0.1, max_new_tokens=512,task="text-generation")
    
    retreival_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an helpful assistant named Akai. Answer the user's questions in his language based on the below context:"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer directly and concisely the user's questions in his language based on the below context:{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])


    retriever_chain=create_history_aware_retriever(llm= llm , retriever=vectorstore.as_retriever(), prompt=retreival_prompt)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt )
    conversation_chain=create_retrieval_chain(retriever=retriever_chain, combine_docs_chain=combine_docs_chain)
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation.invoke({
        'input': user_question,
        'chat_history' : st.session_state.chat_history
        })
    # Append the user's question and the bot's response to the chat history
    st.session_state.chat_history.append(HumanMessage(content= user_question))
    st.session_state.chat_history.append(AIMessage(content= response['answer']))

    print(f" \n\n\n\n {response}\n\n\n\n")
    # st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:

            answer = message.content
            st.write(bot_template.replace(
                "{{MSG}}", answer), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="MistralDoc",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("M7b Doc 🌍​")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
    
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()

