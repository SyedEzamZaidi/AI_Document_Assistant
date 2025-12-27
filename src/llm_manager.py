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
    


def generate_answer_with_sources(query: str, context_chunks: List[Document], llm) -> dict:
    """
    Generate answer with source citations including page numbers.
    
    Args:
        query: User's question.
        context_chunks: List of Document objects from vector search.
        llm: LLM instance.
        
    Returns:
        dict: {
            'answer': str,
            'sources': List[dict],
            'num_sources': int
        }
    """
    try:
        # Build context with source citations
        context_texts = []
        sources = []
        
        for i, doc in enumerate(context_chunks, 1):
            # Add source number to context
            context_texts.append(f"[Source {i}]\n{doc.page_content}")
            
            # Extract metadata
            source_file = doc.metadata.get('source', 'Unknown')
            page_num = doc.metadata.get('page', None)
            
            # Format page display
            if page_num is not None:
                page_display = f"Page {page_num + 1}"  # Convert 0-indexed to 1-indexed
            else:
                page_display = "N/A"
            
            # Store source info
            sources.append({
                'source_id': i,
                'file': source_file,
                'page': page_display,
                'content_preview': doc.page_content[:200] + "..."
            })
        
        # Combine context
        context = "\n\n".join(context_texts)
        
        # Create prompt with citation instructions
        prompt = f"""Based on the following context, answer the question. 
IMPORTANT: Mention which source(s) you used (e.g., "According to Source 1..." or "As stated in Source 2...").

Context:
{context}

Question: {query}

Answer with citations:"""
        
        # Generate response
        response = llm.invoke(prompt)
        answer = response.content
        
        return {
            'answer': answer,
            'sources': sources,
            'num_sources': len(sources)
        }
        
    except Exception as e:
        raise Exception(f"Failed to generate answer with sources: {e}")


if __name__ == "__main__":
    print("="*70)
    print("LLM MANAGER MODULE TEST")
    print("="*70)
    
    try:
        config = load_config()
        print(f"\nLoading LLM: {get_llm_model_name(config)}")
        
        llm = create_llm(config)
        print("LLM created successfully\n")
        
        # Create sample context chunks with metadata
        test_chunks = [
            Document(
                page_content="The old library stood at the corner of Main Street. It was built in 1892 by the city council.",
                metadata={'source': 'data/library_history.pdf', 'page': 2}
            ),
            Document(
                page_content="The library houses over 50,000 books across fiction, non-fiction, and reference materials.",
                metadata={'source': 'data/library_info.pdf', 'page': 0}
            ),
            Document(
                page_content="Library hours are Monday through Friday from 9 AM to 6 PM, and Saturday from 10 AM to 4 PM.",
                metadata={'source': 'data/library_info.pdf', 'page': 1}
            )
        ]
        
        print("="*70)
        print("TESTING ANSWER GENERATION WITH CITATIONS")
        print("="*70)
        
        # Test question
        query = "When was the library built and what are its hours?"
        print(f"\nQuestion: {query}\n")
        
        # Generate answer with sources
        result = generate_answer_with_sources(query, test_chunks, llm)
        
        # Display answer
        print("Answer:")
        print(result['answer'])
        
        # Display sources
        print(f"\n{'='*70}")
        print(f"SOURCES USED ({result['num_sources']}):")
        print('='*70)
        
        for source in result['sources']:
            print(f"\n[Source {source['source_id']}]")
            print(f"File: {source['file']}")
            print(f"Page: {source['page']}")
            print(f"Content: {source['content_preview']}")
        
        print("\n" + "="*70)
        print("LLM MANAGER MODULE WORKING WITH CITATIONS")
        print("="*70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
