Got it — here's the Gen AI / LLM-focused hierarchy only, pulled from that broader map:

## Generative AI / LLM Knowledge Hierarchy

### 1. LLM Architecture & Foundations
- Tokenization (BPE, WordPiece, SentencePiece)
- Transformer decoder architectures (GPT-style)
- Attention mechanism, self-attention, multi-head attention
- Positional embeddings (absolute, RoPE, ALiBi)
- Attention variants (multi-query, grouped-query attention)

### 2. Training & Fine-Tuning
- Pretraining objectives (causal LM, masked LM)
- Fine-tuning approaches: full fine-tuning, LoRA, QLoRA, PEFT methods
- Instruction tuning
- RLHF, DPO
- Prompt engineering, few-shot/zero-shot learning

### 3. Inference Optimization
- Model compression: quantization (INT8, INT4, GPTQ, AWQ)
- Pruning, knowledge distillation
- KV-cache optimization
- Batching strategies (continuous/dynamic batching)
- Speculative decoding
- Serving frameworks: vLLM, TensorRT-LLM, llama.cpp, Ollama
- Model parallelism: tensor, pipeline, data parallelism

### 4. Retrieval-Augmented Generation (RAG)
- Embedding models
- Vector databases (FAISS, ChromaDB, Pinecone)
- Chunking strategies
- Retrieval strategies (dense, sparse, hybrid)
- Reranking

### 5. Agentic AI Systems
- Agent architectures
- Tool use / function calling
- Multi-agent orchestration
- Frameworks: LangChain, LangGraph, AutoGen
- Memory systems for agents

### 6. Evaluation of LLMs & Agents
- Benchmark evaluation (MMLU, HellaSwag, etc.)
- RAG evaluation (RAGAS)
- Agent workflow evaluation (DeepEval)
- Hallucination detection, factuality checks

### 7. Generative Model Foundations (underlying theory)
- Autoencoders, Variational Autoencoders (VAE)
- GANs
- Diffusion Models

### 8. Safety & Alignment (Gen AI specific)
- AI alignment considerations
- Jailbreak robustness
- Responsible deployment for generative systems

---

Since this maps closely to your actual stack (you already work with RAGAS and DeepEval for evaluation, and FAISS/ChromaDB for retrieval), want me to go deep on **section 3 (Inference Optimization)** first, given that's your stated core differentiator — Socratic-style, one concept at a time like we did with the LeetCode problems?