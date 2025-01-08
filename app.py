import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import fitz
import tempfile
import base64

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_relevant_context(response):
    if 'source_documents' in response:
        return [doc.page_content for doc in response['source_documents']]
    return []

def highlight_pdf(pdf_path, contexts):
    doc = fitz.open(pdf_path)
    for page in doc:
        for context in contexts:
            clean_context = ' '.join(context.split())
            page_text = ' '.join(page.get_text().split())
            
            if clean_context.lower() in page_text.lower():
                text_inst = page.search_for(clean_context)
                for inst in text_inst:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_opacity(0.3)
                    highlight.update()

    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
        doc.save(tmp.name)
        doc.close()
        return tmp.name

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
        model_kwargs={
            "max_new_tokens": 512,
            "temperature": 0.7,
        }
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        get_chat_history=lambda h: h,
        max_tokens_limit=512
    )
    return conversation_chain

def display_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.header("💬 Chat with your PDF")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_paths" not in st.session_state:
        st.session_state.pdf_paths = []

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Document Management")
        pdf_docs = st.file_uploader(
            "Upload your PDFs",
            type="pdf",
            accept_multiple_files=True
        )
        
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    st.session_state.pdf_paths = []
                    for pdf in pdf_docs:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(pdf.read())
                            st.session_state.pdf_paths.append(tmp.name)
                    
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Ready to chat!")
            else:
                st.error("Please upload PDF documents")

    with col2:
        st.subheader("Chat Interface")
        if st.session_state.conversation:
            user_question = st.text_input("Ask about your documents:")
            if user_question:
                with st.spinner("Processing..."):
                    response = st.session_state.conversation({
                        'question': user_question
                    })
                    
                    st.markdown("### Answer:")
                    st.write(response['answer'])
                    
                    contexts = get_relevant_context(response)
                    st.session_state.chat_history.append((user_question, response['answer']))
                    
                    if st.session_state.pdf_paths:
                        st.markdown("### Relevant Document Sections:")
                        for pdf_path in st.session_state.pdf_paths:
                            highlighted_pdf = highlight_pdf(pdf_path, contexts)
                            display_pdf(highlighted_pdf)
            
            if st.session_state.chat_history:
                st.markdown("### Chat History")
                for q, a in st.session_state.chat_history:
                    st.markdown(f"**Q:** {q}")
                    st.markdown(f"**A:** {a}")
                    st.markdown("---")

if __name__ == '__main__':
    main()