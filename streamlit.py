__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import tempfile
import shutil
import uuid
import chromadb
import os
import tempfile

# Patch ChromaDB environment before importing
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["PERSIST_DIRECTORY"] = tempfile.mkdtemp()

import chromadb
from chromadb.config import Settings




# --- APPLY THE ASYNCIO PATCH ---
import nest_asyncio
nest_asyncio.apply()
# -------------------------------

# --- Constants ---
CHROMA_PATH = "chroma_db"  # Directory to store the Chroma database
PREDEFINED_COLLECTION = "pdf_documents"  # Predefined collection name

# Set your Google API Key here (hardcoded)
GOOGLE_API_KEY = "AIzaSyDNy7UW2rTV9WRCJwG5ailDCu3mQOS3LUE"  # Replace with your actual API key
# GOOGLE_API_KEY= st.secrets["AIzaSyDNy7UW2rTV9WRCJwG5ailDCu3mQOS3LUE"]  # Replace with your actual API key
# --- UI Configuration ---
st.set_page_config(page_title="PDF Query with ChromaDB + Gemini RAG", layout="wide")
st.title("📄 PDF Query System with ChromaDB + Gemini RAG")
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #1a73e8; /* Google Blue */
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        border: none;
        font-size: 16px;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #1a73e8;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1a73e8;
        margin: 10px 0;
    }
    .answer-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 15px 0;
    }
    .source-box {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Components ---
@st.cache_resource
def initialize_components():
    """Initialize ChromaDB client, embedding model, and Gemini"""
    try:
        # Create a temporary directory for ChromaDB persistence
        temp_dir = tempfile.mkdtemp()
        client = chromadb.Client(Settings(
            persist_directory=temp_dir,
            anonymized_telemetry=False
        ))

        # Load embedding model
        model = SentenceTransformer('BAAI/bge-small-en')

        # Initialize Gemini
        if GOOGLE_API_KEY != "your_google_api_key_here":
            genai.configure(api_key=GOOGLE_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            gemini_model = None
            st.warning("⚠️ Please set your Google API key in the code to enable Gemini RAG")

        return client, model, gemini_model

    except Exception as e:
        st.error(f"Failed to initialize components: {e}")
        return None, None, None


client, embedding_model, gemini_model = initialize_components()

# --- Sidebar Information ---
with st.sidebar:
    st.header("🔧 System Configuration")
    st.markdown(f"""
    <div class="info-box">
    <strong>Collection:</strong> {PREDEFINED_COLLECTION}<br>
    <strong>Database Path:</strong> {CHROMA_PATH}<br>
    <strong>Embedding Model:</strong> BAAI/bge-small-en<br>
    <strong>RAG Model:</strong> Gemini Pro<br>
    <strong>Status:</strong> {'🟢 Ready' if gemini_model else '🔴 API Key Needed'}
    </div>
    """, unsafe_allow_html=True)
    
    # Check if database exists and get collection info
    if client:
        try:
            collection = client.get_collection(PREDEFINED_COLLECTION)
            doc_count = collection.count()
            st.success("✅ ChromaDB Connected")
            st.info(f"📊 Documents in store: {doc_count}")
        except Exception:
            st.warning("❌ No collection found. Upload documents first.")

    if st.button("Clear Chroma Database"):
        if client:
            try:
                # Delete collection if it exists
                try:
                    client.delete_collection(PREDEFINED_COLLECTION)
                    st.sidebar.success("ChromaDB collection cleared.")
                except:
                    pass
                # Also remove the directory
                if os.path.exists(CHROMA_PATH):
                    shutil.rmtree(CHROMA_PATH)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error clearing database: {e}")
        else:
            st.sidebar.info("ChromaDB client not initialized.")

# --- Core Functions ---

def read_and_chunk_pdfs(uploaded_files):
    """
    Reads uploaded PDF files, extracts text, and splits it into chunks.
    """
    all_docs = []
    with st.spinner('Reading and chunking PDFs...'):
        for uploaded_file in uploaded_files:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                    tmpfile.write(uploaded_file.getvalue())
                    tmpfile_path = tmpfile.name

                loader = PyPDFLoader(tmpfile_path)
                documents = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunked_docs = text_splitter.split_documents(documents)
                
                # Add filename to metadata
                for doc in chunked_docs:
                    doc.metadata['filename'] = uploaded_file.name
                
                all_docs.extend(chunked_docs)
                os.remove(tmpfile_path)
                st.success(f"✅ Processed: {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {e}")
    return all_docs

def create_vector_store(docs):
    """
    Creates embeddings and stores them in ChromaDB collection.
    """
    if not client or not embedding_model:
        st.error("❌ ChromaDB client or embedding model not initialized.")
        return False

    with st.spinner(f"Creating embeddings and storing in ChromaDB..."):
        try:
            # Try to get existing collection or create new one
            try:
                collection = client.get_collection(PREDEFINED_COLLECTION)
                st.info("📝 Using existing collection")
            except:
                collection = client.create_collection(PREDEFINED_COLLECTION)
                st.info("🆕 Created new collection")

            # Prepare data for ChromaDB
            texts = []
            metadatas = []
            ids = []

            for doc in docs:
                text_content = doc.page_content
                texts.append(text_content)
                
                metadata = {
                    'filename': doc.metadata.get('filename', 'Unknown'),
                    'page': str(doc.metadata.get('page', 'Unknown')),
                    'source': doc.metadata.get('source', 'Unknown')
                }
                metadatas.append(metadata)
                
                # Generate unique ID
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)

            # Create embeddings
            embeddings = embedding_model.encode(texts).tolist()

            # Add to collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )

            st.success(f"✅ Successfully stored {len(docs)} document chunks!")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to create vector store: {e}")
            return False

def vector_search(query, top_k=5):
    """
    Search for relevant documents using vector similarity.
    """
    if not client or not embedding_model:
        st.error("❌ ChromaDB client or embedding model not initialized.")
        return None

    try:
        collection = client.get_collection(PREDEFINED_COLLECTION)
        
        # Create query embedding
        query_embedding = embedding_model.encode([query]).tolist()
        
        # Search collection
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        return results
        
    except Exception as e:
        if "does not exist" in str(e):
            st.error("❌ No documents found in ChromaDB. Please upload and process documents first.")
        else:
            st.error(f"❌ Search failed: {e}")
        return None

def generate_gemini_answer(query, search_results):
    """
    Generate answer using Gemini RAG system with retrieved context.
    """
    if not gemini_model:
        return generate_simple_answer(query, search_results)
    
    if not search_results or not search_results['documents'][0]:
        return "No relevant documents found to answer your query."
    
    try:
        # Prepare context from retrieved documents
        relevant_docs = search_results['documents'][0]
        metadatas = search_results['metadatas'][0]
        
        context = ""
        sources = []
        
        for i, (doc, metadata) in enumerate(zip(relevant_docs[:4], metadatas[:4]), 1):
            context += f"Document {i}:\n{doc}\n\n"
            source_info = f"{metadata.get('filename', 'Unknown')} (Page: {metadata.get('page', 'Unknown')})"
            if source_info not in sources:
                sources.append(source_info)
        
        # Create RAG prompt for Gemini
        rag_prompt = f"""
You are an expert assistant helping users understand documents. Based on the provided context, answer the user's question comprehensively and accurately.

Context from documents:
{context}

User Question: {query}

Instructions:
1. Provide a detailed and accurate answer based solely on the context provided
2. If the context doesn't contain enough information to answer fully, mention what's missing
3. Structure your response clearly with proper formatting
4. Be specific and cite relevant information from the context
5. If you find contradictory information, mention it

Answer:
"""
        
        # Generate response using Gemini
        response = gemini_model.generate_content(rag_prompt)
        
        return response.text, sources
        
    except Exception as e:
        st.error(f"❌ Gemini RAG generation failed: {e}")
        return generate_simple_answer(query, search_results)

def generate_simple_answer(query, search_results):
    """
    Fallback simple answer generation when Gemini is not available.
    """
    if not search_results or not search_results['documents'][0]:
        return "No relevant documents found.", []
    
    # Get the most relevant chunks
    relevant_texts = search_results['documents'][0][:3]
    metadatas = search_results['metadatas'][0][:3]
    
    answer = f"Based on the documents, here are the most relevant findings:\n\n"
    sources = []
    
    for i, (text, metadata) in enumerate(zip(relevant_texts, metadatas), 1):
        answer += f"**Finding {i}:**\n{text[:400]}{'...' if len(text) > 400 else ''}\n\n"
        source_info = f"{metadata.get('filename', 'Unknown')} (Page: {metadata.get('page', 'Unknown')})"
        if source_info not in sources:
            sources.append(source_info)
    
    return answer, sources

# --- Streamlit App Logic ---

st.header("1. Upload Your Documents")
st.markdown("Upload PDF files to add them to the ChromaDB store. Documents will be automatically processed and vectorized using sentence transformers.")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("📤 Process and Store PDFs"):
        documents = read_and_chunk_pdfs(uploaded_files)
        if documents:
            st.write(f"📊 Total chunks created: {len(documents)}")
            success = create_vector_store(documents)
            if success:
                st.balloons()

st.header("2. Ask Questions (RAG-Powered)")
st.markdown("Ask questions about your documents. The system will retrieve relevant context and use Gemini to generate comprehensive answers.")

query = st.text_input("🤖 Enter your question:", placeholder="What is the main topic discussed in the documents?")

if query:
    with st.spinner("🔍 Searching documents and generating answer..."):
        # Step 1: Vector search for relevant documents
        search_results = vector_search(query, top_k=5)
        
        if search_results and search_results['documents'][0]:
            # Step 2: Generate answer using Gemini RAG
            result = generate_gemini_answer(query, search_results)
            
            if isinstance(result, tuple):
                answer, sources = result
            else:
                answer, sources = result, []
            
            # Display the answer
            st.subheader("🎯 AI-Generated Answer:")
            st.markdown(f"""
            <div class="answer-box">
            {answer}
            </div>
            """, unsafe_allow_html=True)
            
            # Display sources
            if sources:
                st.subheader("📚 Sources:")
                for source in sources:
                    st.markdown(f"""
                    <div class="source-box">
                    📄 {source}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display retrieved context (optional)
            with st.expander("🔍 View Retrieved Context"):
                documents = search_results['documents'][0]
                metadatas = search_results['metadatas'][0]
                distances = search_results['distances'][0] if 'distances' in search_results else [0] * len(documents)
                
                for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                    similarity_score = 1 - distance
                    st.markdown(f"**Context {i+1}** (Similarity: {similarity_score:.3f})")
                    st.markdown(f"*Source: {metadata.get('filename', 'Unknown')} | Page: {metadata.get('page', 'Unknown')}*")
                    
                    # Display document chunk
                    display_text = doc[:600] + "..." if len(doc) > 600 else doc
                    st.text_area(f"Content {i+1}:", display_text, height=150, disabled=True)
                    st.markdown("---")
        else:
            st.warning("❌ No relevant documents found. Please upload and process some PDFs first.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("🚀 **Advanced RAG System**")
st.sidebar.markdown("**Architecture:**")
st.sidebar.markdown("- 📊 Vector Search: ChromaDB + Sentence Transformers")
st.sidebar.markdown("- 🧠 Answer Generation: Google Gemini Pro") 
st.sidebar.markdown("- 🔍 Semantic Search: BAAI/bge-small-en")
st.sidebar.markdown("- 💾 Persistent Storage: Local ChromaDB")

st.sidebar.markdown("**Advantages:**")
st.sidebar.markdown("- ✅ Reliable vector storage")
st.sidebar.markdown("- 🎯 Accurate AI responses")
st.sidebar.markdown("- 📚 Source attribution")

st.sidebar.markdown("- 🔄 Context-aware answers")
