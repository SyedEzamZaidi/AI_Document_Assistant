try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("Langchain found")
    
except ImportError as e:
    print(f"Import error: {e}")

except Exception as e:
    print (f"Unexpected error: {e}")






