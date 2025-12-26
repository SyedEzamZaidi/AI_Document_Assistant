"""
LLM manager module for AI Document Assistant.
Handles answer generation using local Ollama models.
Cloud provider support will be added in Week 2.
"""

from typing import List
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from config_loader import (
    load_config, 
    get_llm_model_name, 
    get_llm_provider,
    get_llm_temperature,
    get_llm_max_tokens
)


def create_llm(config: dict = None):
    """
    Create LLM instance from config.
    Currently supports only Ollama local models.
    
    Args:
        config: Configuration dictionary. Loads from file if None.
        
    Returns:
        ChatOllama instance.
        
    Raises:
        Exception: If provider is not 'local' or model creation fails.
    """
    try:
        if config is None:
            config = load_config()
        
        provider = get_llm_provider(config)
        
        # Currently Only support local Ollama models
        if provider != "local":
            raise Exception(
                f"Currently only 'local' provider is supported. "
                f"Found: '{provider}'. Set llm.current_provider to 'local' in config."
            )
        
        model_name = get_llm_model_name(config)
        temperature = get_llm_temperature(config)
        max_tokens = get_llm_max_tokens(config)
        
        llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=max_tokens
        )
        
        return llm
        
    except Exception as e:
        raise Exception(f"Failed to create LLM: {e}")


def generate_answer(query: str, context_chunks: List[Document], llm) -> str:
    """
    Generate answer using retrieved chunks and LLM.
    
    Args:
        query: User's question.
        context_chunks: List of Document objects from vector search.
        llm: LLM instance.
        
    Returns:
        Generated answer string.
    """
    try:
        # Extract text content from Document objects
        context_texts = [doc.page_content for doc in context_chunks]
        
        # Combine all chunks into single context
        context = "\n\n".join(context_texts)
        
        # Create prompt
        prompt = f"""Based on the following context, answer the question. If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response
        response = llm.invoke(prompt)
        
        # Extract answer text
        answer = response.content
        
        return answer
        
    except Exception as e:
        raise Exception(f"Failed to generate answer: {e}")


if __name__ == "__main__":
    print("="*70)
    print("LLM MANAGER MODULE TEST")
    print("="*70)
    
    try:
        # Load config and create LLM
        config = load_config()
        print(f"\nLoading LLM: {get_llm_model_name(config)}")
        print(f"Temperature: {get_llm_temperature(config)}")
        print(f"Max tokens: {get_llm_max_tokens(config)}")
        
        llm = create_llm(config)
        print("LLM created successfully\n")
        
        # Create sample context chunks
        print("="*70)
        print("TESTING ANSWER GENERATION")
        print("="*70)
        
        test_chunks = [
            Document(page_content="The old library stood at the corner of Main Street and Oak Avenue. It was built in 1892 by the city council."),
            Document(page_content="The library houses over 50,000 books across various genres including fiction, non-fiction, and reference materials."),
            Document(page_content="Library hours are Monday through Friday from 9 AM to 6 PM, and Saturday from 10 AM to 4 PM.")
        ]
        
        # Test question 1
        query1 = "When was the library built?"
        print(f"\nQuestion: {query1}")
        answer1 = generate_answer(query1, test_chunks, llm)
        print(f"Answer: {answer1}\n")
        
        # Test question 2
        query2 = "What are the library hours?"
        print(f"Question: {query2}")
        answer2 = generate_answer(query2, test_chunks, llm)
        print(f"Answer: {answer2}\n")
        
        # Test question 3 (not in context)
        query3 = "Who is the current librarian?"
        print(f"Question: {query3}")
        answer3 = generate_answer(query3, test_chunks, llm)
        print(f"Answer: {answer3}\n")
        
        print("="*70)
        print("LLM MANAGER MODULE WORKING")
        print("="*70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nMake sure:")
        print("1. Ollama is running (ollama serve)")
        print("2. Model is downloaded (ollama pull llama3.2:3b)")
        print("3. Config has correct model name")