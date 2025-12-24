"""
Embeddings module for AI Document Assistant.
Handles text-to-vector conversion using HuggingFace models.
"""

from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from config_loader import load_config, get_embedding_model_name


def create_embedding_model(config: dict = None):
    """
    Create embedding model from config.
    
    Args:
        config: Configuration dictionary. Loads from file if None.
        
    Returns:
        Configured HuggingFace embeddings model.
        
    Raises:
        Exception: If model cannot be loaded.
    """
    try:
        if config is None:
            config = load_config()

        model_name = get_embedding_model_name(config)
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        return embeddings
        
    except Exception as e:
        raise Exception(f"Failed to create embedding model: {e}")


def embed_text(text: str, embeddings) -> List[float]:
    """
    Convert a single text into a vector embedding.
    
    Args:
        text: Text to embed.
        embeddings: HuggingFace embeddings model instance.
        
    Returns:
        Vector representation of text.
    """
    vector = embeddings.embed_query(text)
    return vector


def embed_texts(texts: List[str], embeddings) -> List[List[float]]:
    """
    Convert multiple texts into vector embeddings.
    
    Args:
        texts: List of texts to embed.
        embeddings: HuggingFace embeddings model instance.
        
    Returns:
        List of vector representations.
    """
    vectors = embeddings.embed_documents(texts)
    return vectors


if __name__ == "__main__":
    print("="*70)
    print("EMBEDDINGS MODULE TEST")
    print("="*70)
    
    try:
        config = load_config()
        print(f"\nLoading model: {get_embedding_model_name(config)}")
        
        embedding_model = create_embedding_model(config)
        print("Model loaded successfully.\n")
        
        # Test single text embedding
        text = "The cat sat on the mat"
        vector = embed_text(text, embedding_model)
        
        print(f"Text: '{text}'")
        print(f"Vector length: {len(vector)}")
        print(f"First 5 numbers: {[round(n, 4) for n in vector[:5]]}\n")
        
        # Test multiple texts
        texts = [
            "The cat sat on the mat",
            "A dog played in the park"
        ]
        vectors = embed_texts(texts, embedding_model)
        
        print(f"Embedded {len(vectors)} texts")
        print(f"Each vector has {len(vectors[0])} dimensions")
        
        print("\n" + "="*70)
        print("EMBEDDINGS MODULE WORKING")
        print("="*70)
        
    except Exception as e:
        print(f"\nERROR: {e}")