import os
import json
import streamlit as st
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

from utils.actors_list import ACTORS
from process_ocr import group_by_keywords

from dotenv import load_dotenv
load_dotenv()

# ---- Configuration ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4"
OCR_MODEL = PaddleOCR(use_textline_orientation=True, lang='en')
KNOWLEDGE_DIR = "knowledge"

@st.cache_data
def ocr_extract_text(image_bytes):
    # OCR to extract raw text fragments
    result = OCR_MODEL.predict(image_bytes)[0]
    steps = group_by_keywords(result)
    # Convert flow list into numbered string for chatbot input
    ans = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
    return ans

@st.cache_data
def load_knowledge_docs():
    docs = []
    for fname in os.listdir(KNOWLEDGE_DIR):
        path = os.path.join(KNOWLEDGE_DIR, fname)
        if fname.endswith('.txt') or fname.endswith('.json'):
            with open(path, 'r') as f:
                content = f.read()
            metadata = {"source": fname}
            docs.append(Document(page_content=content, metadata=metadata))
    return docs

@st.cache_resource
def build_vector_store(_documents):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(_documents)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

# ---- Streamlit UI ----
st.set_page_config(page_title="Dynamic RAG Swimlane Chatbot", layout="wide")
st.title("📊 Dynamic RAG Swimlane Diagram Assistant")

# Layout
left_col, right_col = st.columns([1, 2])

# display the image in the left
with left_col:
    uploaded = st.file_uploader("Upload Swimlane Diagram", type=["png","jpg","jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        st.image(image, caption="Swimlane Diagram", use_container_width=True)
        # Extract text from image
        image_np = np.array(image)
        raw_text = ocr_extract_text(image_np)
        # st.markdown("**Extracted Text:**")
        # st.caption(raw_text[:200] + '...')

with right_col:
    if uploaded:
        # Load and index knowledge docs
        knowledge_docs = load_knowledge_docs()
        vectorstore = build_vector_store(tuple(knowledge_docs))
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k":3} # k is number of top similar docs to return
        )
        
        # Retrieve relevant docs based on image text
        relevant_chunks = retriever.get_relevant_documents(raw_text)

        st.markdown("**Matched Knowledge Sources:**")
        for doc in relevant_chunks:
            st.markdown(f"- {doc.metadata['source']}")
        
        # Setup RAG chain on relevant chunks
        embeddings = OpenAIEmbeddings()
        
        # Create a temporary FAISS store for chunks
        temp_store = FAISS.from_documents(relevant_chunks, embeddings)
        partial_retriever = temp_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k":3}
        )
        llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=OPENAI_API_KEY, 
            temperature=0.2 # lower for more precision and less creative
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=partial_retriever, 
            return_source_documents=False
        )

        # Chat interface
        if "history" not in st.session_state:
            st.session_state.history = []
        
        query = st.text_input("Ask a question about the flow:")

        if query:
            st.session_state.history.append(("user", query))
            answer = qa_chain.run(query)
            st.session_state.history.append(("assistant", answer))

        # Display chat history above input
        for role, msg in st.session_state.history:
            if role == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**Assistant:** {msg}")
        
    else:
        st.info("Upload a swimlane diagram image to begin dynamic retrieval and Q&A.")
