from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_text(filepath, chunk_size=500, chunk_overlap=50):
    """
    Load a text file and split it into chunks.
    
    Args:
        filepath (str): Path to the text file
        chunk_size (int): Target size for each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of text chunks (strings)
    """
    # Create splitter using the parameters passed to function
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Read file
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split and return chunks
    chunks = splitter.split_text(text)
    return chunks


if __name__ == "__main__":
    # Test the function
    filepath = "data/sample/test.txt"
    chunks = load_and_chunk_text(filepath, chunk_size=500, chunk_overlap=50)
    
    # Your formatting work goes here - for display during testing
    total_chunks = len(chunks)
    list_of_chunks = []
    
    for i, chunk in enumerate(chunks, 1):
        list_of_chunks.append(f"Chunk {i}: {chunk}\n")
    
    output = f"Total Chunks: {total_chunks}\n\n{''.join(list_of_chunks)}"
    print(output)