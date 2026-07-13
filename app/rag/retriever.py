import os
import logging
from typing import List
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from app.utils.retry import RetryingGoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

logger = logging.getLogger("gapnavigator.rag.retriever")

# Configuration paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(BASE_DIR, "rag", "knowledge_base")
INDEX_PATH = os.path.join(os.path.dirname(BASE_DIR), "vectorstore", "faiss_index")

def get_embeddings_model() -> RetryingGoogleGenerativeAIEmbeddings:
    """Returns the Google Generative AI Embeddings model wrapper."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it in your .env file.")
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2")
    return RetryingGoogleGenerativeAIEmbeddings(
        model=embedding_model,
        google_api_key=api_key
    )

def build_vector_store() -> FAISS:
    """
    Parses resource files from knowledge_base, chunks them,
    creates a FAISS vector index, and saves it locally.
    """
    logger.info("Initializing knowledge base indexing...")
    
    if not os.path.exists(KB_DIR):
        os.makedirs(KB_DIR, exist_ok=True)
        logger.warning(f"Knowledge base directory was missing. Created empty folder at {KB_DIR}")
        
    # Read files in KB_DIR
    all_text = ""
    for filename in os.listdir(KB_DIR):
        if filename.endswith(".md") or filename.endswith(".txt"):
            filepath = os.path.join(KB_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                all_text += f.read() + "\n\n"
                
    if not all_text.strip():
        # Fallback raw data if KB directory is empty
        logger.warning("No knowledge base documents found. Creating default index from fallback data.")
        all_text = """
# Fallback Python Programming
- **Title**: Official Python Tutorial
  **URL**: https://docs.python.org/3/tutorial/
  **Type**: Documentation
  **Description**: Learn Python syntax, control flow, data structures, and standard libraries.
  **Skills**: Python, Coding, Basics
        """
        
    # Split by double newlines so each bullet block/resource is a document
    chunks = [c.strip() for c in all_text.replace("\r\n", "\n").split("\n\n") if c.strip()]
    
    # Filter out header elements or very short lines that don't represent resource items
    cleaned_chunks = [c.strip() for c in chunks if c.strip() and "Title" in c]
    
    if not cleaned_chunks:
        # If nothing parsed with "Title", use all non-empty chunks
        cleaned_chunks = [c.strip() for c in chunks if c.strip()]
        
    logger.info(f"Split knowledge base into {len(cleaned_chunks)} resource items.")
    
    embeddings = get_embeddings_model()
    
    db = FAISS.from_texts(cleaned_chunks, embeddings)
    
    # Save the index locally
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    db.save_local(INDEX_PATH)
    logger.info(f"Vector store saved successfully at {INDEX_PATH}")
    return db

def get_vector_store() -> FAISS:
    """
    Retrieves the local FAISS vector store. 
    Loads from disk if available, otherwise builds it.
    """
    embeddings = get_embeddings_model()
    
    if os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        logger.info(f"Loading existing FAISS index from {INDEX_PATH}")
        try:
            return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.error(f"Error loading local index: {e}. Rebuilding index...")
            return build_vector_store()
    else:
        logger.info("Local FAISS index not found. Building a new one.")
        return build_vector_store()

def retrieve_resources(query: str, k: int = 5) -> List[str]:
    """
    Retrieves the top k learning resource documents matching the query.
    
    Args:
        query: Query string (e.g., missing skills like 'React, Redux, Node.js')
        k: Number of resources to retrieve
        
    Returns:
        List[str]: Vetted resource text blocks
    """
    try:
        db = get_vector_store()
        docs = db.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    except Exception as e:
        logger.error(f"Failed to retrieve resources: {e}")
        return []
