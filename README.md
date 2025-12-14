# AI Document Assistant

Retrieval-Augmented Generation (RAG) system for enterprise document Q&A. Built to demonstrate practical AI integration with existing automation workflows while maintaining data privacy and cost efficiency.

## What It Does

Query natural language questions against internal documents and get accurate, sourced answers. Think of it as ChatGPT for your company's documentation, but running locally so data never leaves your network.

**Current state:** Basic text processing complete, building out the RAG pipeline.

## Why Local Models?

Working in enterprise environments means data privacy is non-negotiable. Using Ollama + HuggingFace models keeps everything on-premises. No API calls = no data leakage = IT Security approval.

Plus it's free to run, which matters when you're trying to get budget approved.

## Architecture

```
Documents → Chunking → Embeddings (local) → ChromaDB
                                                ↓
User Query → Embedding (local) → Vector Search → Context
                                                ↓
Context + Query → LLM (Ollama) → Answer
```

Everything is abstracted so I can swap models via config without changing code. Start with free models, upgrade to Azure OpenAI later if needed.

## Tech Stack

**Current:**
- Python 3.13
- LangChain for RAG orchestration
- Ollama (Llama 3.1) for LLM - runs locally
- sentence-transformers for embeddings - runs locally
- ChromaDB for vector storage
- Gradio for UI (quick demos, has built-in sharing)

**Planned:**
- Docker for deployment
- Redis for caching repeated queries
- Azure OpenAI integration (if company approves budget)
- Power BI for analytics/comparison

## Design Decisions

**Why Ollama instead of OpenAI API?**
- Data stays on company servers (compliance requirement)
- No per-query costs
- Can run in air-gapped environments
- Good enough quality for most use cases

**Why ChromaDB?**
- Simple to set up
- Works locally
- Good performance for <1M documents
- Can switch to FAISS or Pinecone later if needed

**Why Gradio?**
- Fast to build UI
- Built-in public sharing (good for demos)
- Less code than Streamlit for this use case

**Modular design:**
- Embedding layer is abstracted (HuggingFace, OpenAI, Azure)
- LLM layer is abstracted (Ollama, Azure OpenAI, etc)
- Switch providers by changing config.yaml, no code changes

## Project Structure

```
AI_Document_Assistant/
├── src/
│   ├── embeddings/     # embedding model integrations
│   ├── llm/            # LLM provider wrappers
│   ├── rag/            # chunking, retrieval, pipeline
│   ├── evaluation/     # quality scoring (planned)
│   └── utils/          # config, logging
├── configs/
│   ├── local.yaml      # free local models (default)
│   └── azure.yaml      # Azure OpenAI config
├── data/
│   ├── documents/      # source docs
│   └── vectordb/       # ChromaDB storage
├── practice/           # learning exercises (not in git)
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/AI_Document_Assistant.git
cd AI_Document_Assistant
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install Ollama
# Download from https://ollama.ai
ollama pull llama3.1

# Run
python src/main.py
```

## Status

**Completed:**
- Document loading (txt, pdf, docx)
- Text analysis and batch processing
- Project structure and git setup

**In Progress (Dec 2025):**
- Document chunking strategies
- Embedding generation
- Vector database integration
- RAG query pipeline

**Next (Jan 2026):**
- Docker deployment
- Multi-model support (Azure OpenAI option)
- Response quality evaluation
- Cost/performance tracking

**Later (Feb-Mar 2026):**
- Gradio interface
- Power BI dashboard for model comparison
- Production deployment docs

## Deployment Strategy

**Phase 1 - Proof of Concept:**
Run on laptop, share Gradio link with 5-10 people, gather feedback.

**Phase 2 - Pilot:**
Deploy Docker container to company dev VM, test with 50-100 users, measure time savings.

**Phase 3 - Production:**
Either keep local (Ollama on prod VM) or upgrade to Azure OpenAI if budget approved. Depends on cost/quality tradeoffs from pilot data.

## Configuration Example

```yaml
# config.yaml
llm:
  provider: "ollama"              # or "azure_openai"
  model: "llama3.1"
  local: true

embeddings:
  provider: "huggingface"
  model: "all-MiniLM-L6-v2"
  local: true

vector_db:
  type: "chromadb"
  persist_directory: "./data/vectordb"

chunking:
  method: "recursive"             # or "semantic", "fixed"
  chunk_size: 512
  overlap: 50
```

Change the config, system adapts. No code changes needed.

## Security & Compliance

All data processing happens locally. Models are downloaded once to company servers, inference runs offline. No external API calls unless explicitly configured (Azure OpenAI).

Can run in completely air-gapped environments if required.

## Cost Analysis

**Local deployment:**
- Infrastructure: ~₹15k-30k/month (VM with decent specs)
- Models: ₹0 (Ollama, HuggingFace are free)
- Total: ₹15k-30k/month for 500-1000 users

**Azure OpenAI (if upgraded):**
- Infrastructure: ~₹20k/month
- API costs: ~₹2-3L/month at scale
- Total: ₹2.2-3.2L/month

Start local, prove ROI, then decide if premium models are worth it.

## Background

I'm a BI & Automation Lead with 7+ years in RPA and process automation. Building this to combine automation expertise with AI capabilities - specifically around intelligent document processing and knowledge retrieval.

The goal is production deployment in enterprise environments and demonstrating practical AI implementation for automation-heavy industries.

## License

MIT - use it however you want.

## Contact

Ezam Zaidi  
syedezamzaidi@gmail.com  
linkedin.com/in/ezam-zaidi

---

Note: This is an active project. README will evolve as features are implemented.
