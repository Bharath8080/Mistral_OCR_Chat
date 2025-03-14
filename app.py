import streamlit as st
import base64
import tempfile
import os
import io
import time
from PIL import Image
from mistralai import Mistral
from langchain_mistralai import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Mistral OCR Chat",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "client" not in st.session_state:
    st.session_state.client = None
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = None

# Get API keys from environment variables
mistral_api_key = os.getenv("MISTRAL_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize clients if API keys are available in environment variables
if mistral_api_key and pinecone_api_key:
    try:
        st.session_state.client = Mistral(api_key=mistral_api_key)
        
        # Test Pinecone connection
        pc = Pinecone(api_key=pinecone_api_key)
        pc.list_indexes()  # This will throw an error if the key is invalid
    except Exception as e:
        st.error(f"Error initializing clients from environment variables: {str(e)}")

def upload_pdf(client, content, filename):
    """
    Uploads a PDF to Mistral's API and retrieves a signed URL for processing.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, filename)
        
        with open(temp_path, "wb") as tmp:
            tmp.write(content)
        
        try:
            with open(temp_path, "rb") as file_obj:
                file_upload = client.files.upload(
                    file={"file_name": filename, "content": file_obj},
                    purpose="ocr"
                )
            
            signed_url = client.files.get_signed_url(file_id=file_upload.id)
            return signed_url.url
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

def process_ocr(client, document_source):
    """
    Processes a document using Mistral's OCR API.
    """
    return client.ocr.process(
        model="mistral-ocr-latest",
        document=document_source,
        include_image_base64=True
    )

def get_vectorstore_from_text(text, index_name="mistral-ocr-index"):
    """
    Creates a Pinecone vector store from text content.
    """
    # Initialize MistralAI embedding model
    embedding_model = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=mistral_api_key
    )
    
    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Check if index exists, create if not
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        with st.spinner("Creating Pinecone index (this may take a moment)..."):
            pc.create_index(
                name=index_name,
                dimension=1024,  # Dimension for Mistral embeddings
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait for index to be ready
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
    
    # Connect to the index
    index = pc.Index(index_name)
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    from langchain_core.documents import Document
    from uuid import uuid4
    
    # Split text into chunks and create Documents
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    # Generate unique IDs for documents
    ids = [str(uuid4()) for _ in range(len(documents))]
    
    # Create vector store
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
    
    # Add documents to vector store
    vector_store.add_documents(documents=documents, ids=ids)
    
    return vector_store

def get_vectorstore_from_url(url, index_name="mistral-ocr-index"):
    """
    Creates a Pinecone vector store from a website URL.
    """
    with st.spinner("Loading and processing website content..."):
        # Initialize MistralAI embedding model
        embedding_model = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=mistral_api_key
        )
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if index exists, create if not
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=1024,  # Dimension for Mistral embeddings
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait for index to be ready
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
        
        # Connect to the index
        index = pc.Index(index_name)
        
        # Get the text in document form
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        document_chunks = text_splitter.split_documents(documents)
        
        # Generate unique IDs for documents
        from uuid import uuid4
        ids = [str(uuid4()) for _ in range(len(document_chunks))]
        
        # Create vector store
        vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
        
        # Add documents to vector store
        vector_store.add_documents(documents=document_chunks, ids=ids)
        
        return vector_store

def get_context_retriever_chain(vector_store):
    """
    Creates a context retriever chain.
    """
    # Initialize MistralAI model for LLM
    llm = ChatMistralAI(
        model="mistral-large-latest",
        mistral_api_key=mistral_api_key
    )
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(retriever_chain):
    """
    Creates a conversational RAG chain.
    """
    # Initialize MistralAI model for LLM
    llm = ChatMistralAI(
        model="mistral-large-latest",
        mistral_api_key=mistral_api_key,
        streaming=True
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_streaming_response(user_input):
    """
    Gets a streaming response from the RAG chain.
    """
    # Format chat history for the LLM
    formatted_history = []
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            formatted_history.append(HumanMessage(content=message["content"]))
        else:
            formatted_history.append(AIMessage(content=message["content"]))

    # Create retriever chain
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    # Initialize message placeholder
    message_placeholder = st.empty()
    full_response = ""
    
    # Get streaming response
    for chunk in conversation_rag_chain.stream({
        "chat_history": formatted_history,
        "input": user_input
    }):
        if "answer" in chunk:
            full_response += chunk["answer"]
            message_placeholder.markdown(full_response + "▌")
    
    message_placeholder.markdown(full_response)
    return full_response

def display_pdf(content):
    """
    Displays a PDF in Streamlit using an iframe.
    """
    base64_pdf = base64.b64encode(content).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.markdown(
    f'<h1><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Mistral_AI_logo_%282025%E2%80%93%29.svg/1200px-Mistral_AI_logo_%282025%E2%80%93%29.svg.png" width="60"/> Chat with Pinecone <img src="https://raw.githubusercontent.com/deepset-ai/haystack-integrations/main/logos/pinecone.png" width="40"/></h1>',
    unsafe_allow_html=True
    )

    
    # Sidebar for document source selection
    with st.sidebar:
        st.sidebar.image("https://home-wordpress.deeplearning.ai/wp-content/uploads/2024/01/pinecone-logo-white.png", use_container_width=True)
        st.sidebar.header("📤 Document Upload")
        # Pinecone index name input
        index_name = st.text_input("Pinecone Index Name", value="mistral-ocr-index")
        
        # Document source selector
        source_type = st.radio("Select source type:", ["URL", "PDF", "Image"])
        
        # URL input
        if source_type == "URL":
            url = st.text_input("Enter website URL:")
            if url and st.button("Process URL"):
                if mistral_api_key and pinecone_api_key:
                    try:
                        with st.spinner("Processing URL..."):
                            # Process URL with OCR
                            document_source = {
                                "type": "document_url",
                                "document_url": url
                            }
                            
                            # Process OCR
                            ocr_response = process_ocr(st.session_state.client, document_source)
                            
                            if ocr_response and ocr_response.pages:
                                # Combine extracted text from all pages
                                extracted_text = "\n\n".join([page.markdown for page in ocr_response.pages])
                                st.session_state.extracted_text = extracted_text
                                
                                # Create vector store
                                st.session_state.vector_store = get_vectorstore_from_text(
                                    extracted_text, 
                                    index_name
                                )
                                
                                st.success("URL processed successfully!")
                            else:
                                st.warning("No content extracted from URL.")
                    except Exception as e:
                        st.error(f"Error processing URL: {str(e)}")
                else:
                    st.warning("API keys not found in environment variables. Please check your .env file.")
        
        # PDF upload
        elif source_type == "PDF":
            uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
            if uploaded_file and st.button("Process PDF"):
                if mistral_api_key and pinecone_api_key:
                    try:
                        with st.spinner("Processing PDF..."):
                            # Read file content
                            content = uploaded_file.getvalue()
                            
                            # Upload PDF and get signed URL
                            signed_url = upload_pdf(st.session_state.client, content, uploaded_file.name)
                            
                            # Process PDF with OCR
                            document_source = {
                                "type": "document_url",
                                "document_url": signed_url
                            }
                            
                            # Process OCR
                            ocr_response = process_ocr(st.session_state.client, document_source)
                            
                            if ocr_response and ocr_response.pages:
                                # Combine extracted text from all pages
                                extracted_text = "\n\n".join([page.markdown for page in ocr_response.pages])
                                st.session_state.extracted_text = extracted_text
                                
                                # Create vector store
                                st.session_state.vector_store = get_vectorstore_from_text(
                                    extracted_text, 
                                    index_name
                                )
                                
                                st.success("PDF processed successfully!")
                            else:
                                st.warning("No content extracted from PDF.")
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
                else:
                    st.warning("API keys not found in environment variables. Please check your .env file.")
        
        # Image upload
        else:
            uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
            if uploaded_file and st.button("Process Image"):
                if mistral_api_key and pinecone_api_key:
                    try:
                        with st.spinner("Processing image..."):
                            # Read file content
                            content = uploaded_file.getvalue()
                            
                            # Convert image to base64
                            image = Image.open(io.BytesIO(content))
                            buffered = io.BytesIO()
                            image.save(buffered, format="PNG")
                            img_str = base64.b64encode(buffered.getvalue()).decode()
                            
                            # Process image with OCR
                            document_source = {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{img_str}"
                            }
                            
                            # Process OCR
                            ocr_response = process_ocr(st.session_state.client, document_source)
                            
                            if ocr_response and ocr_response.pages:
                                # Combine extracted text from all pages
                                extracted_text = "\n\n".join([page.markdown for page in ocr_response.pages])
                                st.session_state.extracted_text = extracted_text
                                
                                # Create vector store
                                st.session_state.vector_store = get_vectorstore_from_text(
                                    extracted_text, 
                                    index_name
                                )
                                
                                st.success("Image processed successfully!")
                            else:
                                st.warning("No content extracted from image.")
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
                else:
                    st.warning("API keys not found in environment variables. Please check your .env file.")
        
        # Add option to clear vector store
        if st.session_state.vector_store is not None:
            if st.button("Clear Vector Store"):
                try:
                    # Initialize Pinecone
                    pc = Pinecone(api_key=pinecone_api_key)
                    
                    # Delete the index
                    try:
                        pc.delete_index(index_name)
                        st.session_state.vector_store = None
                        st.success(f"Pinecone index '{index_name}' deleted successfully!")
                    except Exception as e:
                        st.error(f"Error deleting index: {str(e)}")
                except Exception as e:
                    st.error(f"Error connecting to Pinecone: {str(e)}")
        
    # Create container for chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
    
    # Check if vector store exists
    if st.session_state.vector_store is None:
        st.info("Please process a document first to start chatting.")
    
    # Chat input
    if st.session_state.vector_store is not None:
        user_question = st.chat_input("Ask a question about your document")
        if user_question:
            # Add user message to chat
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Display user message
            st.chat_message("user").write(user_question)
            
            # Get streaming response
            with st.chat_message("assistant"):
                response = get_streaming_response(user_question)
            
            # Add assistant message to chat
            st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
