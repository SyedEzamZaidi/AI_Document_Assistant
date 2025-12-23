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

from config_loader import (
    load_config,
    get_chunking_method,
    get_chunking_params
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


if __name__ == "__main__":
    # Test the document processor with config
    print("="*70)
    print("DOCUMENT PROCESSOR TEST")
    print("="*70)
    
    # Load config
    config = load_config()
    
    # Test file path
    filepath = "data/sample/test.txt"
    
    # Display configuration being used
    print(f"\nFile: {filepath}")
    print(f"Method: {get_chunking_method(config)}")
    chunk_size, chunk_overlap = get_chunking_params(config)
    print(f"Parameters: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}\n")
    
    # Process document
    try:
        chunks = load_and_chunk_text(filepath, config)
        
        # Display results
        print(f"✅ Successfully processed document!")
        print(f"Total Chunks: {len(chunks)}\n")
        
        # Show first 3 chunks
        print("First 3 chunks:")
        print("-" * 70)
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\nChunk {i} ({len(chunk)} chars):")
            print(f"{chunk[:200]}{'...' if len(chunk) > 200 else ''}")
        
        if len(chunks) > 3:
            print(f"\n... and {len(chunks) - 3} more chunks")
        
        print("\n" + "="*70)
        print("✅ DOCUMENT PROCESSOR WORKING!")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Make sure data/sample/test.txt exists!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")