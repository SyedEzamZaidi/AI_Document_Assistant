"""
Vector store module for AI Document Assistant.
Handles document storage and retrieval using ChromaDB.
"""

from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config_loader import load_config, get_vector_store_config


def create_vector_store(chunks: List[str], embeddings, config: dict = None):
    """
    Create a new vector store from document chunks.
    
    Args:
        chunks: List of text chunks to store.
        embeddings: Embeddings model instance.
        config: Configuration dictionary. Loads from file if None.
        
    Returns:
        Chroma vector store instance.
        
    Raises:
        Exception: If vector store creation fails.
    """
    try:
        if config is None:
            config = load_config()
        
        # Get vector store configuration
        vs_config = get_vector_store_config(config)
        persist_dir = vs_config['persist_directory']
        
        # Convert string chunks to Document objects
        documents = [Document(page_content=chunk) for chunk in chunks]
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        
        return vectorstore
        
    except Exception as e:
        raise Exception(f"Failed to create vector store: {e}")


def load_vector_store(embeddings, config: dict = None):
    """
    Load an existing vector store from disk.
    
    Args:
        embeddings: Embeddings model instance.
        config: Configuration dictionary. Loads from file if None.
        
    Returns:
        Chroma vector store instance.
        
    Raises:
        Exception: If vector store cannot be loaded.
    """
    try:
        if config is None:
            config = load_config()
        
        # Get persist directory
        vs_config = get_vector_store_config(config)
        persist_dir = vs_config['persist_directory']
        
        # Load existing vector store
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        
        return vectorstore
        
    except Exception as e:
        raise Exception(f"Failed to load vector store: {e}")


def search_similar(query: str, vector_store, k: int = 3) -> List[Document]:
    """
    Search for similar documents in the vector store.
    
    Args:
        query: Search query text.
        vector_store: Chroma vector store instance.
        k: Number of results to return.
        
    Returns:
        List of similar Document objects.
    """
    results = vector_store.similarity_search(query, k=k)
    return results


if __name__ == "__main__":
    print("="*70)
    print("VECTOR STORE MODULE TEST")
    print("="*70)
    
    try:
        # Import required modules
        from document_processor import load_and_chunk_text
        from embeddings import create_embedding_model
        
        # Load configuration
        config = load_config()
        print("\nLoading document and creating chunks...")
        
        # Load and chunk document
        filepath = "data/sample/test.txt"
        chunks = load_and_chunk_text(filepath, config)
        print(f"Created {len(chunks)} chunks")
        
        # Create embeddings model
        print("\nCreating embeddings model...")
        embeddings = create_embedding_model(config)
        print("Embeddings model ready")
        
        # Create vector store
        print("\nCreating vector store...")
        vectorstore = create_vector_store(chunks, embeddings, config)
        print("Vector store created successfully")
        
        # Test search
        print("\n" + "="*70)
        print("TESTING SEARCH FUNCTIONALITY")
        print("="*70)
        
        query = "library"
        print(f"\nQuery: '{query}'")
        results = search_similar(query, vectorstore, k=2)
        
        print(f"\nFound {len(results)} similar chunks:")
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"{doc.page_content[:200]}...")
        
        # Test loading existing vector store
        print("\n" + "="*70)
        print("TESTING LOAD FUNCTIONALITY")
        print("="*70)
        
        print("\nLoading existing vector store...")
        loaded_vectorstore = load_vector_store(embeddings, config)
        print("Vector store loaded successfully")
        
        # Verify loaded store works
        results = search_similar(query, loaded_vectorstore, k=1)
        print(f"Search on loaded store: Found {len(results)} result")
        
        print("\n" + "="*70)
        print("VECTOR STORE MODULE WORKING")
        print("="*70)
        
    except Exception as e:
        print(f"\nERROR: {e}")