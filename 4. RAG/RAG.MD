# Retrieval-Augmented Generation (RAG) — Complete Guide

---

# 1. EMBEDDING MODELS

## 1.1 What They Are and Why They Matter

**Embedding models** convert text into dense numerical vectors (hundreds to thousands of dimensions) that capture semantic meaning. They are the foundation of RAG retrieval — the quality of your retrieval is bounded by the quality of your embedding model . If you choose the wrong embedding model, even a perfect vector database won't help .

**The practical constraint:** Switching embedding models later requires re-indexing your entire vector database, which can be a multi-day operation at scale . This makes embedding choice a **long-term commitment**.

## 1.2 How They Work

**Training objective (contrastive learning):** The model is trained so that semantically similar texts land near each other in a fixed-dimensional vector space. Positive pairs (query and matching document) are pulled together; negative pairs are pushed apart . The result is a function from text to a unit vector where cosine similarity correlates with semantic relatedness .

**Mathematically:** The model maps input text to a vector in ℝᵈ (typically 1024–3072 dimensions). For retrieval, you embed every document chunk at index time and store the vectors. At query time, you embed the query and find the K nearest document vectors using cosine similarity .

## 1.3 Production-Ready Options (2026)

### Best General-Purpose (English)

| Model | Dimensions | MTEB Score | Price / 1M tokens |
|-------|------------|------------|-------------------|
| **Google gemini-embedding-2** | 3072 (Matryoshka 128-3072) | Top of leaderboard | Check vendor |
| **Voyage voyage-4-large** | 2048/1024/512/256 | Top-tier | $0.12 |
| **OpenAI text-embedding-3-large** | 3072 | 64.6% | $0.13 |
| **OpenAI text-embedding-3-small** | 1536 | 62.3% | $0.02 |
| **BAAI bge-large-en-v1.5** | 1024 | Competitive | Free (self-host) |
| **Cohere embed-v4** | 1536 | — | — |
| **BGE-M3** | 1024 | Competitive | Free (self-host) |



**Matryoshka advantage:** Models like `text-embedding-3-large` support Matryoshka Representation Learning — you can truncate from 3072d to 256d and still beat older models, trading some quality for significantly less storage and faster search .

### Domain-Specific

For **clinical/medical domains**, consider PubMedBERT or BioBERT if evaluation shows measurable precision improvement on your actual query workload . General-purpose embeddings often work well for most content; invest in domain-specific only when evaluation justifies the complexity .

## 1.4 Code: Getting Embeddings

```python
import openai

# Option A: OpenAI
openai.api_key = "your-key"

def get_embedding(text: str) -> list[float]:
    response = openai.embeddings.create(
        model="text-embedding-3-small",  # 1536 dims, fast + cheap
        input=text
    )
    return response.data[0].embedding

# Option B: Batch embeddings — ALWAYS batch in ingestion
def embed_batch(texts: list[str]) -> list[list[float]]:
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=texts  # Up to 2048 inputs per request
    )
    return [d.embedding for d in response.data]

# Option C: Open-source (local)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-en-v1.5')
embeddings = model.encode(["Hello world", "Another text"])
```


---

# 2. VECTOR DATABASES

## 2.1 What They Are

Vector databases are purpose-built for storing embeddings and running fast Approximate Nearest Neighbor (ANN) searches at scale. They use index structures like HNSW, IVF, or DiskANN to make similarity search tractable over millions or billions of vectors .

## 2.2 Production Options (2026)

| Vector DB | Best For | Deployment Model | Key Feature |
|-----------|----------|------------------|-------------|
| **Pinecone** | Managed, production scale | Cloud (managed) | Fully managed, no ops |
| **Qdrant** | Open-source self-host | Open-source / Cloud | gRPC support, great performance |
| **pgvector** | When you already use PostgreSQL | Open-source | In your existing Postgres |
| **Chroma** | Local development | Open-source | Dead simple local setup |
| **Weaviate** | Enterprise features | Open-source / Cloud | Built-in hybrid search |
| **Milvus** | Very large scale | Open-source / Cloud | CUDA GPU acceleration |



## 2.3 Architecture Choices

**The frozen-at-ingest problem:** Your chosen vector type and embedding model are long-term commitments. Switching models later requires re-indexing the entire vector database, which is expensive and slows development . Mitigate by adding a versioning layer — create separate indexes for testing new models, then gradually migrate traffic once validated .

## 2.4 Code: Vector Store Setup

```python
# Qdrant (self-hosted)
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="my_docs",
    vectors_config={"size": 1024, "distance": "Cosine"}
)

# Insert vectors
client.upsert(
    collection_name="my_docs",
    points=[
        {"id": 1, "vector": embedding, "payload": {"text": chunk_text}}
    ]
)

# Search
results = client.search(
    collection_name="my_docs",
    query_vector=query_embedding,
    limit=10
)
```


---

# 3. CHUNKING STRATEGIES

## 3.1 Why Chunking Is Critical

Chunking is the **single biggest lever for RAG quality** . Wrong chunking = poor retrieval. Chunks need to be small enough for precise retrieval but large enough to contain meaningful context for the LLM to reason over .

## 3.2 The Production Decision (2026)

**The benchmark evidence is consistent:** recursive 512-token chunking scored **69% end-to-end accuracy** in the FloTorch 2026 RAG benchmark, beating every more expensive alternative. Semantic chunking scored only **54%** because its chunks averaged ~43 tokens — too small for the generator to reason over .

**The contradiction:** Chroma Research separately reports semantic chunking scoring as high as 91.9% on *retrieval recall*, illustrating that strong recall does **not** always translate to strong end-to-end answers . Qu, Tu, and Bao's 2024 study reaches the same conclusion: **fixed-size chunking consistently matches or outperforms semantic chunking once compute cost is factored in** .

## 3.3 Chunking Approaches

| Approach | Description | Best For | When to Use |
|----------|-------------|----------|-------------|
| **Fixed-size** | Split by character/token count (e.g., 512 tokens with 50 overlap) | General purpose, simplicity | Default — start here |
| **Recursive** | Try boundaries in order: paragraph, sentence, word | General prose | The most common production approach |
| **Semantic** | Split where sentence embeddings diverge | Topic boundaries | Only if evaluation shows gain |
| **Document-aware** | Split at Markdown headings, code functions, PDF structure | Structured docs | Formatted documents |
| **Hierarchical (parent-document)** | Index small chunks for retrieval, return larger parent for context | When you need both precision and context | Complex documents |



## 3.4 Chunking Rules of Thumb

| Document Type | Strategy | Chunk Size |
|---------------|----------|------------|
| General prose | Recursive character + paragraph boundary | 800-1200 tokens |
| Markdown docs | Split at ## headings | Full section |
| Code files | Split at function/class | Full function |
| Q&A pairs | Keep Q+A together, never split | Full pair |
| Tables | Keep entire table, add row context | Full table |



## 3.5 Code: Chunking

```python
# Fixed-size (simple)
def fixed_chunk(text: str, size: int = 512, overlap: int = 50) -> list[str]:
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i + size])
    return chunks

# Recursive character splitter (recommended)
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]  # Try boundaries in order
)
chunks = splitter.split_text(document)
```


---

# 4. RETRIEVAL STRATEGIES

## 4.1 Dense Retrieval (Semantic Search)

**How it works:** Embed query and document chunks, find nearest neighbors by vector similarity. Excellent for paraphrasing and synonyms — can match "my package never arrived" with a document about "shipment lost in transit" .

**Strength:** Handles vocabulary mismatch and multi-word concepts well .

**Weakness:** Misses queries with unique identifiers like "SKU-7841-A" because embedding models compress rare tokens .

## 4.2 Sparse Retrieval (BM25)

**How it works:** Matches documents based on term frequency and inverse document frequency (TF-IDF). Fast, exact, and great at proper nouns and rare terms .

**Strength:** If someone searches for the error code "ERR_CONNECTION_REFUSED," BM25 will find documents containing that exact string. Sub-millisecond and explainable .

**Weakness:** Can't recognize that "car" and "automobile" are synonyms — vocabulary mismatch .

## 4.3 Hybrid Retrieval (Production Standard)

**How it works:** Runs both dense and sparse queries in parallel and fuses the results. Nearly everyone runs both in production because BM25 catches rare-term cases and semantic catches paraphrase cases .

**Fusion mechanism — Reciprocal Rank Fusion (RRF):**

$RRF(doc) = \sum_{r \in rankings} \frac{1}{k + rank_r(doc)}$

Where $k=60$ is the constant from the original SIGIR 2009 paper .

No score calibration is needed — RRF uses only ranks. A document that ranks highly in both result lists gets a strong combined score; one that only appears in one list is penalized .

## 4.4 Graph-Based Retrieval (GraphRAG)

**When vector search falls short:** Standard vector search treats every chunk as an isolated unit. It can find individually relevant chunks but has no mechanism for understanding relationships between chunks that span multiple documents .

**GraphRAG approach (Microsoft Research):** During indexing, an LLM extracts entities (people, organizations, concepts), relationships, and claims from the text. These are organized into a hierarchical community structure using graph algorithms. Local search explores the neighborhood of relevant entities; global search answers questions about the corpus as a whole .

**Trade-off:** Significant computational cost during indexing; the LLM processes the entire corpus to build the knowledge graph. Best for focused, high-value knowledge bases where relational reasoning is critical .

## 4.5 Code: Hybrid Retrieval

```python
# Conceptual hybrid with RRF
def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for doc_list in rankings:
        for rank, doc_id in enumerate(doc_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda x: -x[1]))

# Combine dense and sparse results
dense_results = dense_search(query_embedding, top_k=50)
sparse_results = bm25_search(query_terms, top_k=50)
final_ranked = rrf([dense_results, sparse_results])
```


---

# 5. RERANKING

## 5.1 What It Is and Why It's Needed

Initial retrieval uses **bi-encoder** models that embed queries and documents independently. This enables fast search across millions of documents but means the query and document never directly interact during scoring . Standard semantic search often returns high-similarity results that don't match the user's intent .

**Reranking adds a second-pass scoring stage** using **cross-encoder** models, which process the query and each candidate chunk simultaneously . Cross-encoders take the top-N candidates (typically 25-50) and produce much more accurate relevance scores by allowing query and document tokens to attend to each other .

## 5.2 How Many Reranking Stages?

- **Semantic ranking** adds minimal latency and should be enabled by default 
- **Cross-encoder reranking** adds 50-200ms but significantly improves relevance for domain-specific content — justify with offline evaluation showing ≥5 percentage points of nDCG improvement 
- **LLM-as-reranker** is reserved for the final 3-5 candidates when accuracy justifies the cost 

## 5.3 Production Options

| Reranker | Description | Best For |
|----------|-------------|----------|
| **BAAI/bge-reranker-base** | Cross-encoder, open-source | General purpose |
| **Cohere Rerank** | Managed API | When you don't want to self-host |
| **Azure AI Search semantic ranker** | Built-in, add minimal latency | Azure environments |
| **LLM-as-reranker** | Use the generation LLM itself | Maximum accuracy, final candidates |



## 5.4 Code: Reranking

```python
# Using BGE reranker (self-hosted)
import requests

response = requests.post(
    "http://localhost:8000/rerank",
    json={
        "query": "What is Kubernetes?",
        "documents": ["Doc 1 text...", "Doc 2 text..."]
    }
)
scores = response.json()  # [0.95, 0.23, ...]
```


---

# FULL COMPARISON TABLE

| Component | Options | Best For | Trade-off |
|-----------|---------|----------|-----------|
| **Embedding Model** | OpenAI text-embedding-3-large, bge-large-en, Cohere embed-v4 | General English, multilingual, open-source | Quality vs cost; switching requires re-indexing |
| **Vector DB** | Pinecone, Qdrant, pgvector, Chroma | Managed scale, self-host, existing Postgres, local dev | Ops burden vs control |
| **Chunking** | Fixed-size, recursive, semantic, hierarchical | Speed+quality baseline, best accuracy, structured docs | Simple vs complex; recall doesn't equal accuracy |
| **Retrieval** | Dense, Sparse (BM25), Hybrid (RRF) | Semantic match, exact terms, production default | Paraphrasing vs exact term match |
| **Reranking** | Cross-encoder, LLM-as-reranker, Semantic ranker | Domain-specific, maximum accuracy, low-latency baseline | Quality vs latency |

---

# QUICK DECISION RULES

1. **Start with:** Recursive 512-token chunking + text-embedding-3-small + Qdrant/pgvector + hybrid search (dense + BM25) + optional reranking 
2. **Go hybrid always** — pure vector search alone misses exact terms; pure BM25 misses synonyms 
3. **Use reranking** for domain-specific or high-value queries; skip for high-volume, simple tasks 
4. **Embedding is a frozen decision** — pick carefully; switching later means re-indexing everything 
5. **Evaluate** with recall@k, MRR, NDCG on a labeled dataset before production 

---

# 5 MOST-ASKED RAG INTERVIEW QUESTIONS

**1. Why is embedding model choice a "frozen decision" in production RAG?** Because switching embedding models later requires re-embedding the entire document corpus, which at scale is a multi-day operation. You can mitigate this with a versioning layer — separate indexes for testing new models before migrating traffic.

**2. What is hybrid retrieval and when should you use it?** Hybrid retrieval runs both dense (semantic) and sparse (BM25) search in parallel and fuses results using Reciprocal Rank Fusion (RRF). It's the production default because dense alone misses exact terms (like error codes), while sparse alone misses synonyms and paraphrasing.

**3. Why does semantic chunking often underperform fixed-size chunking in end-to-end accuracy?** Semantic chunking creates small chunks (~43 tokens average), which are too small for the generator to reason over. Fixed-size 512-token chunks consistently outperform semantic chunking in end-to-end benchmarks, even though semantic may have better retrieval recall.

**4. What is the difference between a bi-encoder and cross-encoder in retrieval?** Bi-encoders embed queries and documents independently (fast, can pre-compute) — used for initial retrieval. Cross-encoders process query and document simultaneously (slower, but more accurate) — used for reranking the top candidates.

**5. What does RRF solve in hybrid search?** RRF fuses results from multiple retrieval systems (e.g., dense + sparse) using only rank positions, not scores. This avoids the problem of calibrating scores between systems that use different metrics. A document ranked highly in both lists gets a strong combined score.