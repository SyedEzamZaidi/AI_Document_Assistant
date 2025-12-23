import sys
sys.path.append('src')

from langchain_huggingface import HuggingFaceEmbeddings
from config_loader import load_config, get_embedding_model_name

# Load config
config = load_config()
model_name = get_embedding_model_name(config)

print(f"Using model: {model_name}\n")

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Test sentences
sentence1 = "The cat sat on the mat"
sentence2 = "A feline rested on the rug"

# Embed both
vector1 = embeddings.embed_query(sentence1)
vector2 = embeddings.embed_query(sentence2)

# Print results
print(f"Vector dimensions: {len(vector1)}\n")

# Sentence 1 
print(f"Sentence 1: '{sentence1}'")
first_10_v1 = []
for num in vector1[:10]:
    first_10_v1.append(round(num, 4))
print(f"First 10 numbers: {first_10_v1}\n")

# Sentence 2 
print(f"Sentence 2: '{sentence2}'")
first_10_v2 = []
for num in vector2[:10]:
    first_10_v2.append(round(num, 4))
print(f"First 10 numbers: {first_10_v2}")