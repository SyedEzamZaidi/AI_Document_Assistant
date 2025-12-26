"""
RAG Pipeline module for AI Document Assistant.
Connects all components to enable question-answering from documents.
"""

from typing import Optional
from config_loader import load_config
from embeddings import create_embedding_model
from vector_store import load_vector_store, search_similar
from llm_manager import create_llm, generate_answer


def ask_question(query: str, config: dict = None) -> str:
    """
    Ask a question and get an answer from the RAG system.
    
    This function connects all RAG components:
    1. Loads embeddings model
    2. Loads vector store with document chunks
    3. Searches for relevant chunks
    4. Generates answer using LLM
    
    Args:
        query: User's question.
        config: Configuration dictionary. Loads from file if None.
        
    Returns:
        Generated answer string.
        
    Raises:
        Exception: If any component fails to load or execute.
    """
    try:
        if config is None:
            config = load_config()

        # Load all required components
        embeddings = create_embedding_model(config)
        llm = create_llm(config)
        vector_store = load_vector_store(embeddings, config)
        
        # Search for relevant context
        context_chunks = search_similar(query=query, vector_store=vector_store, k=3)
        
        # Generate answer from context
        answer = generate_answer(query=query, context_chunks=context_chunks, llm=llm)
        
        return answer
        
    except Exception as e:
        raise Exception(f"Failed to process question: {e}")


if __name__ == "__main__":
    print("="*70)
    print("RAG PIPELINE TEST")
    print("="*70)
    
    try:
        config = load_config()
        
        # Test questions
        questions = [
            "When was the library built?",
            "What are the library hours?",
            "How many books does the library have?",
            "Who is the current librarian?"
        ]
        
        print("\nAsking questions about the library...\n")
        
        for i, question in enumerate(questions, 1):
            print(f"Question {i}: {question}")
            answer = ask_question(query=question, config=config)
            print(f"Answer: {answer}\n")
            print("-"*70 + "\n")
        
        print("="*70)
        print("RAG PIPELINE WORKING")
        print("="*70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nMake sure:")
        print("1. Vector store exists (run vector_store.py first)")
        print("2. Ollama is running with llama3.1:8b model")
        print("3. All dependencies are installed")