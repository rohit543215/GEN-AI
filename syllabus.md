Got it — here's the Gen AI / LLM-focused hierarchy only, pulled from that broader map:

## Generative AI / LLM Knowledge Hierarchy

### 0. The Hardware/Systems Foundation (the missing piece)

Everything else in this document is meaningless without this — quantization, batching, and KV-cache optimization are all just different answers to the same underlying hardware problem.

Concepts to actually understand, not just recognize
Concept	What you need to know, precisely
Memory-bound vs. compute-bound	An operation is memory-bound if the GPU spends more time moving data (from HBM to SRAM) than computing on it. LLM inference's decode phase (generating one token at a time) is famously memory-bound — the GPU is mostly waiting on memory bandwidth, not compute.
Arithmetic intensity	FLOPs performed ÷ bytes moved from memory. Low arithmetic intensity = memory-bound; high = compute-bound. This single number tells you whether a technique like quantization will actually help a given operation.
The Roofline Model	A graph plotting achievable performance (FLOPs/sec) against arithmetic intensity, with a hard ceiling from peak memory bandwidth on one side and peak compute on the other. Every optimization technique you'll study either (a) reduces bytes moved, or (b) reduces FLOPs, or (c) does both — the roofline model tells you which matters for a given operation.
GPU memory hierarchy	HBM (large, slow, where weights/KV-cache live) → SRAM/shared memory (small, fast, on-chip) → registers. Data must move HBM→SRAM before compute can touch it — this movement is the actual bottleneck decode-phase inference fights against.
Prefill vs. decode phase	Prefill (processing the prompt) is compute-bound and highly parallel (all prompt tokens processed at once). Decode (generating each new token) is memory-bound and sequential (one token at a time, re-reading the entire KV-cache each step). Nearly every inference optimization technique targets one phase specifically — know which.
Batch size and its two effects	Increasing batch size raises arithmetic intensity (better hardware utilization, since you reuse loaded weights across more sequences) but also increases memory pressure (more KV-caches to store simultaneously). This tension is why batching strategy is a whole sub-field, not a solved problem.
Self-test

Can you explain, from the roofline model, why quantizing the weights to INT4 speeds up decode-phase generation specifically (hint: it's about bytes moved, not FLOPs) — but gives much less benefit during prefill (hint: prefill is already compute-bound, so reducing memory traffic doesn't help as much)?

Hands-on

Profile a small model's inference (even a small HuggingFace model on CPU/GPU) using torch.profiler or nsight systems if you have access — actually look at the time breakdown between prefill and decode, and between memory operations and compute kernels. Seeing the real numbers, not just the theory, is what makes this stick.


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