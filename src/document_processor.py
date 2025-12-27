"""
Document processor module for AI Document Assistant.
Loads and chunks documents using configurable strategies.
"""

from typing import List
from pathlib import Path

# Import ALL possible text splitters (professional pattern)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document

from config_loader import (
    load_config,
    get_chunking_method,
    get_chunking_params,
    get_documents_directory,
    is_recursive_search
)


def get_text_splitter(chunk_size: int, chunk_overlap: int, method: str):
    """
    Create a text splitter based on method name.
    
    Args:
        chunk_size (int): Target size for each chunk
        chunk_overlap (int): Overlap between chunks
        method (str): Splitter class name
        
    Returns:
        TextSplitter: Configured text splitter instance
        
    Raises:
        ValueError: If method is not recognized
    """
    # Map config class name → imported class
    splitter_map = {
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
        "CharacterTextSplitter": CharacterTextSplitter,
        "TokenTextSplitter": TokenTextSplitter
    }
    
    if method not in splitter_map:
        raise ValueError(
            f"Unknown splitter method: {method}. "
            f"Available: {list(splitter_map.keys())}"
        )
    
    splitter_class = splitter_map[method]
    
    return splitter_class(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


def load_and_chunk_text(filepath: str, config: dict = None) -> List[str]:
    """
    Load a text file and split it into chunks using config.
    
    Args:
        filepath (str): Path to the text file
        config (dict): Configuration dictionary (loads from file if None)
        
    Returns:
        List[str]: List of text chunks (strings)
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    # Load config if not provided
    if config is None:
        config = load_config()
    
    # Verify file exists
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Get chunking configuration from YAML
    method = get_chunking_method(config)
    chunk_size, chunk_overlap = get_chunking_params(config)
    
    # Create splitter
    splitter = get_text_splitter(chunk_size, chunk_overlap, method)
    
    # Read file
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split and return chunks
    chunks = splitter.split_text(text)
    
    return chunks



def load_documents_from_directory(config: dict = None) -> List[Document]:
    """
    Load all documents from configured directory and chunk them.
    Supports multiple file types: PDF, TXT, DOCX.
    
    Args:
        config: Configuration dictionary. Loads from file if None.
        
    Returns:
        List of Document objects with chunks.
        
    Raises:
        Exception: If directory doesn't exist or loading fails.
    """
    try:
        if config is None:
            config = load_config()
        
        # Get configuration
        directory_path = get_documents_directory(config)
        file_types = config['document_sources']['file_types']
        recursive = is_recursive_search(config)
        
        # Check if directory exists
        if not Path(directory_path).exists():
            raise FileNotFoundError(f"Documents directory not found: {directory_path}")
        
        print(f"Loading documents from: {directory_path}")
        print(f"File types: {file_types}")
        print(f"Recursive search: {recursive}")
        
        all_documents = []
        
        # Load each file type
        for file_type in file_types:
            glob_pattern = f"**/*.{file_type}" if recursive else f"*.{file_type}"
            
            # Choose appropriate loader
            if file_type == "pdf":
                loader_cls = PyPDFLoader
            elif file_type == "txt":
                from langchain_community.document_loaders import TextLoader
                loader_cls = TextLoader
            elif file_type == "docx":
                from langchain_community.document_loaders import Docx2txtLoader
                loader_cls = Docx2txtLoader
            else:
                print(f"Warning: Unsupported file type '{file_type}', skipping...")
                continue
            
            # Load documents of this type
            loader = DirectoryLoader(
                directory_path,
                glob=glob_pattern,
                loader_cls=loader_cls
            )
            
            docs = loader.load()
            
            if docs:
                print(f"Loaded {len(docs)} pages from {file_type.upper()} files")
                all_documents.extend(docs)
        
        if not all_documents:
            print(f"Warning: No documents found in {directory_path}")
            return []
        
        print(f"\nTotal pages loaded: {len(all_documents)}")
        
        # Get chunking configuration
        method = get_chunking_method(config)
        chunk_size, chunk_overlap = get_chunking_params(config)
        
        # Create text splitter
        splitter = get_text_splitter(chunk_size, chunk_overlap, method)
        
        # Split documents into chunks
        chunks = splitter.split_documents(all_documents)
        
        print(f"Created {len(chunks)} chunks")
        
        return chunks
        
    except Exception as e:
        raise Exception(f"Failed to load documents: {e}")


if __name__ == "__main__":
    print("="*70)
    print("DOCUMENT PROCESSOR TEST")
    print("="*70)
    
    try:
        config = load_config()
        
        # Test 1: Single text file
        print("\nTest 1: Single Text File")
        print("-"*70)
        filepath = "data/sample/test.txt"
        
        try:
            chunks = load_and_chunk_text(filepath, config)
            print(f"File: {filepath}")
            print(f"Chunks created: {len(chunks)}")
            print(f"First chunk: {chunks[0][:150]}...\n")
        except FileNotFoundError:
            print(f"File not found: {filepath}\n")
        
        # Test 2: All documents from directory
        print("Test 2: All Documents from Directory")
        print("-"*70)
        
        try:
            document_chunks = load_documents_from_directory(config)
            
            if document_chunks:
                print(f"Total chunks created: {len(document_chunks)}")
                
                # Show first chunk WITH page number
                first_chunk = document_chunks[0]
                source = first_chunk.metadata.get('source', 'Unknown')
                page = first_chunk.metadata.get('page', 'N/A')  # Page number!
                
                print(f"\nFirst chunk preview:")
                print(f"Content: {first_chunk.page_content[:150]}...")
                print(f"Source: {source}")
                print(f"Page: {page}")  # Show page number
                
            else:
                print("No documents found")
                print("Add files to data/documents/ to test")
                
        except FileNotFoundError:
            print("Directory 'data/documents/' not found")
            print("Create it and add PDF/TXT/DOCX files to test")
        
        print("\n" + "="*70)
        print("TESTS COMPLETE")
        print("="*70)
        
    except Exception as e:
        print(f"ERROR: {e}") 