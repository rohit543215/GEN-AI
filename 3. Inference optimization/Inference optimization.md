# Inference Optimization — Complete Masterclass

---

# 1. MODEL COMPRESSION: QUANTIZATION (INT8, INT4, GPTQ, AWQ)

## 1.1 Why Quantization Exists

LLM weights are typically stored and computed in 16-bit (FP16/BF16) or even 32-bit (FP32) floating point during training. At inference time, this precision is often more than necessary for acceptable output quality, and it comes at a steep cost: memory footprint and memory-bandwidth (how fast weights can be moved from GPU memory to compute units) directly scale with bit-width. Since LLM inference is very often **memory-bandwidth bound** rather than compute bound (the GPU spends more time waiting to fetch weights than actually multiplying them), reducing the number of bits used to represent each weight can directly speed up inference, not just save memory.

**Quantization** is the process of converting weights (and sometimes activations) from high-precision floating point to lower-precision representations (INT8, INT4, etc.), while trying to preserve model output quality as much as possible.

## 1.2 The Core Math of Quantization

**Basic linear (affine) quantization**, mapping a floating-point range to an integer range:

$$x_q = \text{round}\left(\frac{x}{s}\right) + z$$

Where:
- $x$ = original floating-point value
- $s$ = scale factor (determines the "step size" between representable values)
- $z$ = zero-point (an integer offset, used when the range isn't symmetric around zero)
- $x_q$ = the resulting quantized integer value

**Dequantization** (converting back to approximate floating point for computation):

$$\hat{x} = s \cdot (x_q - z)$$

**Symmetric vs asymmetric quantization:** symmetric quantization sets $z=0$ and assumes the value range is roughly centered around zero (simpler, slightly faster); asymmetric quantization uses a nonzero $z$ to better fit distributions that aren't centered around zero (e.g., post-ReLU activations, which are always ≥ 0).

**Per-tensor vs per-channel quantization:** per-tensor uses a single scale $s$ for an entire weight matrix; per-channel (or per-group) uses a different scale for each output channel (or small group of weights), which better captures the fact that different channels/groups can have very different value ranges — improving accuracy at a small cost of extra scale-factor storage.

## 1.3 INT8 vs INT4 — The Precision/Quality Tradeoff

- **INT8**: 8 bits per weight → 4x memory reduction vs FP32, 2x vs FP16. Generally very safe — INT8 quantization of LLM weights typically causes negligible quality degradation.
- **INT4**: 4 bits per weight → 8x memory reduction vs FP32, 4x vs FP16. More aggressive — naive INT4 quantization can noticeably hurt quality, which is exactly why smarter INT4-specific techniques like GPTQ and AWQ were developed (rather than just naively rounding every weight to 4 bits).

**The fundamental tension:** fewer bits = smaller model, faster loading, less memory bandwidth needed, cheaper serving — but also less precision to represent the learned weight values, risking quality degradation, especially for weights that matter disproportionately to the model's outputs (outlier weights).

## 1.4 GPTQ (Post-Training Quantization via Optimal Brain Quantization)

**Core idea:** rather than naively rounding each weight independently, GPTQ quantizes weights **layer by layer**, and for each layer, it quantizes weights **one at a time**, immediately adjusting the *remaining, not-yet-quantized* weights in that layer to compensate for the error just introduced — using second-order information (an approximation of the Hessian, related to how sensitive the loss is to each weight) to decide the optimal compensation.

**Why this works better than naive rounding:** naive rounding treats every weight independently and ignores how quantizing one weight affects the overall layer output. GPTQ's compensation step means that even though individual weights are quantized to low precision, the *layer's overall output* stays much closer to the original, unquantized layer's output — because errors introduced by early quantized weights are actively corrected for by adjusting later weights before they too get quantized.

**Practical note:** GPTQ requires a small calibration dataset (a few hundred sample sequences) to compute the statistics needed for this error-compensation process — it's a "post-training" technique, applied after the model is already fully trained, not something baked into training itself.

## 1.5 AWQ (Activation-aware Weight Quantization)

**Core insight:** not all weights are equally important to preserve precision for. AWQ observes that a small percentage of weights (roughly 0.1-1%) are disproportionately important for maintaining output quality — specifically, the weights that correspond to **activations with large magnitudes** (since large activations, multiplied by even a slightly-off weight, produce a larger absolute error in the output).

**How it decides which weights matter:** AWQ runs a calibration pass and looks at the activation statistics (not the weights themselves) to identify which input channels tend to produce large-magnitude activations. The weights connected to these "salient" channels get special treatment — instead of literally keeping them at higher precision (which would complicate hardware implementation with mixed precision), AWQ mathematically **rescales** the important channels *before* quantization (scaling them up) and correspondingly rescales the corresponding activations *down*, so that after uniform quantization, the effective precision preserved for these important weights is higher, while the whole computation still uses a single, hardware-friendly low-bit format throughout.

**Why AWQ is popular in production:** it doesn't require backpropagation or expensive optimization during the quantization process (unlike GPTQ's per-weight compensation, which is more computationally involved) — it's based on directly analyzed statistics, making it faster to quantize a model, while achieving comparable or sometimes better quality than GPTQ, especially at very low bit-widths (like INT4).

## 1.6 Quick Comparison

| Method | Approach | Calibration needed | Speed to quantize | Typical use |
|---|---|---|---|---|
| Naive INT8/INT4 | Simple rounding | No (or minimal) | Fastest | INT8 (safe), risky for INT4 |
| GPTQ | Sequential error-compensated quantization | Yes | Slower | High-quality INT4 |
| AWQ | Activation-aware channel rescaling | Yes | Faster than GPTQ | High-quality INT4, popular in production serving |

### 🔧 Activity 1: Quantization
1. By hand, quantize a small set of 5 floating point values (e.g., `[-2.3, 0.5, 1.8, -0.9, 3.1]`) to INT8 using symmetric linear quantization. Compute the scale factor, the quantized integer values, and then dequantize back — calculate the reconstruction error for each value.
2. Using HuggingFace `transformers` + `bitsandbytes`, load a small model in 8-bit and separately in 4-bit. Compare the model's file size and the actual GPU memory used (`torch.cuda.memory_allocated()`) for each.
3. Read the GPTQ paper's abstract/method section and explain in your own words why quantizing weights "one at a time with compensation" produces less accumulated error than quantizing an entire layer's weights all at once independently.
4. Look up benchmark comparisons (e.g. from the AWQ or GPTQ papers, or community benchmarks) showing perplexity/accuracy of a model at FP16 vs GPTQ-INT4 vs AWQ-INT4 — note how close INT4 quantized models get to the FP16 baseline.

---

# 2. PRUNING & KNOWLEDGE DISTILLATION

## 2.1 Pruning

**Core idea:** remove some of a model's weights (or entire structural components) entirely, rather than compressing them — reducing the total parameter count and computation required.

**Unstructured pruning:** individual weights are set to zero based on some importance criterion (most commonly, magnitude — small-magnitude weights are assumed to contribute little to the output and are zeroed out). This can achieve very high sparsity (e.g., 90% of weights zeroed) with relatively little quality loss, **but** the resulting sparse weight matrix has an irregular pattern of zeros that standard GPU hardware isn't naturally efficient at exploiting for speedup — you typically need specialized sparse-matrix kernels/hardware to actually realize a speed benefit, not just a storage benefit.

**Structured pruning:** removes entire structural units — whole neurons, attention heads, or even entire layers — rather than individual scattered weights. This produces a smaller, "dense" model that's immediately compatible with standard hardware/software without needing special sparse kernels, making the speedup much easier to realize in practice, though it's generally harder to achieve the same aggressive compression ratios without hurting quality compared to unstructured pruning, since you're removing entire meaningful units rather than cherry-picking individually unimportant weights.

**Importance criteria for pruning:**
- Magnitude-based: prune weights (or units) with smallest absolute value
- Gradient/Taylor-based: prune based on estimated impact on the loss function if removed
- Activation-based: prune units that produce consistently low-magnitude activations across a calibration set

## 2.2 Knowledge Distillation

**Core idea:** train a smaller "student" model to mimic the behavior of a larger, already-trained "teacher" model — transferring the teacher's learned knowledge into a much more compact architecture that's cheaper to run at inference time.

**How it works — soft targets:** instead of training the student purely on hard ground-truth labels (e.g., one-hot "correct answer"), the student is also trained to match the teacher's full output probability **distribution** (the "soft targets" — the teacher's predicted probability for every possible next token, not just the single correct one).

**Why soft targets carry more information than hard labels:** a hard label just says "the correct answer is X." The teacher's full probability distribution over all tokens also reveals *how confident* the teacher is, and which incorrect answers it considers "almost right" versus "completely wrong" — e.g., for the sentence "The cat sat on the ___", a teacher might assign 70% to "mat", 15% to "rug", 10% to "floor", and tiny probabilities elsewhere. This relative ranking and confidence information (sometimes called "dark knowledge") teaches the student much richer information about the *relationships between classes/tokens* than a single correct label would.

**Distillation loss (typically a combination of two terms):**

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{hard}(y, y_{student}) + (1-\alpha) \cdot \mathcal{L}_{soft}(p_{teacher}^T, p_{student}^T)$$

Where $\mathcal{L}_{hard}$ is standard cross-entropy against the true label, $\mathcal{L}_{soft}$ is typically KL-divergence between the teacher and student's (temperature-softened) probability distributions, and $\alpha$ balances the two.

**Temperature scaling ($T$):** the softmax used to produce the teacher's distribution is "softened" using a temperature parameter:

$$p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

Higher $T$ produces a "softer," more spread-out probability distribution (revealing more of the relative ranking/dark knowledge among incorrect classes, rather than an overconfident near-one-hot distribution), which is more useful signal for the student to learn from.

**Why this matters for LLM inference optimization specifically:** distillation is how many small, efficient "SLMs" (small language models) are created from large frontier models — the small model is trained to mimic a much larger model's behavior, aiming to retain much of its capability at a fraction of the inference cost/latency.

### 🔧 Activity 2: Pruning & Distillation
1. Implement simple magnitude-based unstructured pruning on a small neural network's weight matrix (e.g., zero out the smallest 50% of weights by absolute value in a PyTorch linear layer). Measure how much the model's accuracy on a simple task degrades.
2. Explain in your own words why structured pruning is often preferred in production serving even though it typically achieves less aggressive compression than unstructured pruning.
3. Implement a basic knowledge distillation training loop in PyTorch: train a small "student" network to match both the hard labels AND the softened output distribution of a larger pretrained "teacher" network on a simple classification dataset (e.g., MNIST).
4. Experiment with different temperature values (e.g., T=1, T=5, T=20) in your distillation loss — observe how the student's training behavior/final accuracy changes as the teacher's distribution becomes softer.

---

# 3. KV-CACHE OPTIMIZATION

## 3.1 Recap: What the KV-Cache Is and Why It Matters

As covered in the attention variants topic, the KV-cache stores previously computed Key and Value tensors during autoregressive generation, avoiding redundant recomputation at every new token. But the KV-cache itself becomes a major memory and efficiency bottleneck at scale — this section covers techniques specifically for optimizing *how* that cache is managed, beyond just the MHA/MQA/GQA architectural choice.

## 3.2 The Scale of the Problem

For a model with $L$ layers, $h$ KV-heads, head dimension $d$, sequence length $s$, and batch size $b$, the KV-cache memory (in number of values, ×2 bytes for FP16, ×2 again for both K and V) is:

$$\text{KV-cache size} = 2 \times L \times h \times d \times s \times b$$

For a 70B-scale model serving many concurrent users with long contexts, this can easily exceed the memory used by the model weights themselves — making efficient KV-cache management a first-order concern for serving infrastructure, not a minor detail.

## 3.3 PagedAttention

**The problem it solves:** naive KV-cache implementations pre-allocate a large contiguous block of memory for each sequence's maximum possible length, but sequences vary in actual length and generation can stop early — this leads to substantial memory **fragmentation and waste** (similar to internal fragmentation in classic OS memory management, which you already covered in your OS review).

**The solution (borrowed directly from OS virtual memory/paging, as the name suggests):** instead of one contiguous memory block per sequence, PagedAttention divides the KV-cache into small fixed-size **blocks** (analogous to memory "pages"), and maintains a **block table** per sequence that maps logical token positions to physical blocks, which don't need to be contiguous in memory.

**Benefits:**
- Near-zero memory waste from fragmentation (blocks are only allocated as actually needed, and unused portions of a partially-filled block are the only waste, bounded by the block size)
- Enables efficient **memory sharing** between sequences — e.g., if multiple requests share a common prompt prefix (like a system prompt), the corresponding KV-cache blocks can be shared/reused across those requests rather than duplicated, since the block table simply points multiple sequences to the same physical blocks (with copy-on-write semantics if one sequence's generation diverges)
- This is the core innovation behind **vLLM**, and is a major reason for its throughput advantage over naive serving implementations

## 3.4 Prefix Caching

**The idea:** when multiple requests share a common prefix (a system prompt, a few-shot example set, a long shared document being queried repeatedly), the KV-cache computed for that shared prefix can be **cached and reused** across requests, rather than being recomputed for every single new request from scratch.

**Why this is a significant speedup:** the "prefill" phase (processing the initial prompt before generation begins) can be computationally expensive for long prompts — if a large chunk of that prompt is identical across many requests (e.g., a fixed system prompt used by every user of an application), caching its KV values means new requests only need to compute KV values for the *new, unique* portion of their input, skipping redundant computation entirely for the shared part.

## 3.5 Sliding Window / Cache Eviction Strategies

For very long conversations or documents, keeping the full KV-cache for the entire history becomes impractical. Strategies to manage this include:

- **Sliding window attention:** only keep/attend to the most recent $W$ tokens' KV-cache, discarding older entries — bounds memory usage at the cost of losing access to distant context
- **Attention sink**: research (e.g., StreamingLLM) has found that keeping just the *first few* tokens' KV-cache (even though they're old) alongside a recent sliding window, rather than discarding them, substantially improves quality for very long streams — these initial tokens seem to act as an "attention sink" that the model relies on disproportionately, regardless of their actual semantic content

### 🔧 Activity 3: KV-Cache Optimization
1. Using the KV-cache size formula, calculate the memory required (in GB) for a model with L=80 layers, h=8 KV-heads (GQA), d=128 head dimension, sequence length s=8192, batch size b=16, at FP16 precision.
2. Read the vLLM PagedAttention paper's introduction and diagram how a block table maps logical sequence positions to physical memory blocks — explain how this enables prefix sharing between two requests with an identical system prompt.
3. Explain in your own words why prefix caching is especially valuable for applications with a fixed system prompt used across many user requests (e.g., a customer support chatbot) versus applications where every request has a completely unique, unrelated prompt.
4. Compare sliding window attention vs full KV-cache retention: what quality tradeoff are you making, and why might "attention sink" tokens be worth preserving even in a sliding-window setup?

---

# 4. BATCHING STRATEGIES (CONTINUOUS/DYNAMIC BATCHING)

## 4.1 Why Batching Matters for Inference Throughput

GPUs achieve their best efficiency when processing many requests in parallel (matrix operations across a larger batch dimension use the hardware's parallel compute capacity far more effectively than processing one request at a time). But naively batching LLM requests is complicated by one key fact: different requests generate different numbers of tokens and finish at different times, unlike, say, a fixed-size image classification batch where every input takes the same amount of compute.

## 4.2 Static (Naive) Batching

**How it works:** collect a fixed batch of requests, run the entire batch through generation together, and only return results once **every** request in the batch has finished generating (typically padding shorter sequences to match the longest one in the batch).

**The core inefficiency:** if one request in the batch needs to generate 500 tokens and another only needs 20, the short request's GPU slot sits idle (or wastefully computes on padding tokens) for the remaining 480 steps, waiting for the longest request to finish — this wastes significant compute and increases average latency across the batch.

## 4.3 Continuous (Dynamic) Batching

**How it works:** rather than waiting for an entire fixed batch to complete before processing the next batch, continuous batching operates at the **iteration level** — after every single decoding step (generating one new token for all currently-active sequences), the system checks which sequences have finished (hit an end-of-sequence token or max length) and **immediately evicts them, freeing up their slot for a new incoming request** to join the batch right away, without waiting for the other still-running sequences to finish.

**Why this dramatically improves throughput:** GPU slots are almost never sitting idle waiting for a slow request to finish — as soon as any sequence completes, a new one takes its place, keeping the effective batch size (and thus GPU utilization) consistently high. This is one of the primary reasons serving frameworks like vLLM and TensorRT-LLM achieve significantly higher throughput than naive/static batching implementations.

**Practical implication:** continuous batching requires the serving infrastructure to manage a dynamically-changing set of in-flight sequences at the token-generation level (rather than treating "a batch" as a fixed, static unit that's created once and fully processed) — this is architecturally more complex to implement correctly but is now the standard approach in essentially all modern high-throughput LLM serving systems.

### 🔧 Activity 4: Batching Strategies
1. Diagram static batching vs continuous batching over time (e.g., a timeline showing 4 requests with different generation lengths) — visually show the idle/wasted GPU time in static batching that continuous batching eliminates.
2. If you have access to a serving framework like vLLM, run a small load test sending requests with highly varied expected output lengths (e.g., some asking for 1-sentence answers, others for long essays) and observe throughput compared to what a naive batching implementation would achieve.
3. Explain in your own words why continuous batching is especially valuable for real-world production traffic (where request lengths are highly variable and requests arrive continuously over time) versus a benchmark setting with uniform, simultaneous requests.
4. Research question: how does continuous batching interact with the KV-cache/PagedAttention concepts from the previous section — why do these two techniques (continuous batching + PagedAttention) complement each other particularly well in a system like vLLM?

---

# 5. SPECULATIVE DECODING

## 5.1 The Core Bottleneck It Addresses

As established earlier, autoregressive generation is inherently sequential — one token at a time, each depending on the last. For a large model, generating each individual token requires a full forward pass through all its layers, which is relatively slow (even though it's a small amount of new computation per step, the memory-bandwidth cost of loading all the model's weights for that single step is often the dominant cost). This makes total generation latency scale linearly with the number of tokens generated, each one paying this "full model forward pass" tax.

## 5.2 The Core Idea

**Speculative decoding uses a small, fast "draft" model to propose several candidate tokens at once, then uses the large "target" model to verify all of them in a single parallel forward pass — accepting the ones that match what the large model would have generated anyway, and only falling back to slower one-at-a-time generation when a mismatch occurs.**

**Step by step:**
1. The small draft model generates $k$ candidate tokens autoregressively (fast, since it's a small model) — e.g., $k=4$ speculative tokens
2. The large target model processes all $k$ draft tokens **in a single forward pass** (not one at a time) — since verifying $k$ tokens in parallel is roughly the same memory-bandwidth cost as generating just 1 token normally (recall: the model's weights need to be loaded either way; processing a few extra tokens per load is comparatively cheap)
3. The target model's actual predicted probability distribution at each position is compared against the draft model's proposed token
4. Using a specific acceptance/rejection sampling rule (designed to guarantee the final output distribution is mathematically identical to what the target model would have generated on its own, i.e., **no quality loss**), each draft token is either accepted or rejected
5. All tokens up to and including the first rejection are kept; generation then resumes normally from that point (the target model's own correct prediction is used to replace the rejected token, and the process repeats with a new batch of draft tokens)

## 5.3 Why This Speeds Up Generation Without Sacrificing Quality

The key insight is that verifying $k$ tokens with the large model in one parallel forward pass costs roughly the same (in wall-clock time, due to the memory-bandwidth-bound nature of LLM inference) as generating just 1 token normally with the large model. So if the draft model's guesses are frequently correct (which they usually are for a reasonably well-matched draft model — many tokens in natural language are fairly predictable, like completing "The capital of France is ___" with "Paris"), you effectively get multiple tokens' worth of output for close to the price of one large-model forward pass.

**Why there's no quality tradeoff (this is the elegant part):** the acceptance/rejection sampling procedure is mathematically constructed so that the final sequence of accepted tokens has *exactly* the same probability distribution as if you had just run the large target model alone, token by token, the normal slow way. Speculative decoding is a pure speedup technique — not an approximation — as long as it's implemented correctly.

**Choosing a draft model:** the draft model needs to be small/fast (so proposing candidates is cheap) but well-aligned with the target model's behavior (so its guesses are accepted often enough to be worthwhile) — often a smaller model from the same family (e.g., a distilled or smaller-parameter-count version of the target model) works well.

### 🔧 Activity 5: Speculative Decoding
1. Diagram the speculative decoding process step by step for a toy example: draft model proposes 4 tokens, target model accepts the first 2 and rejects the 3rd — show what happens next in the pipeline.
2. Explain in your own words why verifying k tokens in one parallel forward pass costs roughly the same wall-clock time as generating just 1 token normally, tying this back to the "memory-bandwidth bound" nature of LLM inference discussed in the quantization section.
3. Research question: what factors would make a draft model a poor choice for speculative decoding with a given target model, even if the draft model is very fast? (Hint: think about acceptance rate.)
4. If you have access to a framework supporting speculative decoding (e.g., vLLM with a draft model configured), measure the actual speedup on a generation task and record the "acceptance rate" (percentage of draft tokens accepted) if the framework exposes that metric.

---

# 6. SERVING FRAMEWORKS: vLLM, TensorRT-LLM, llama.cpp, Ollama

## 6.1 vLLM

**Core innovation:** PagedAttention (covered in the KV-cache section) plus continuous batching, combined into a high-throughput serving engine designed primarily for GPU-based serving of models via a Python-friendly API (OpenAI-compatible API server included).

**Best suited for:** production-grade GPU serving where maximizing throughput (requests/second, tokens/second across many concurrent users) is the priority — widely used as a backend for serving open-weight LLMs at scale.

**Key features:** continuous batching, PagedAttention, support for many quantization formats (AWQ, GPTQ), tensor parallelism support, prefix caching.

## 6.2 TensorRT-LLM

**Core innovation:** built by NVIDIA specifically to squeeze maximum performance out of NVIDIA GPU hardware, using deep, hardware-specific kernel optimizations, custom CUDA kernels, and aggressive compilation/graph-optimization techniques (leveraging NVIDIA's broader TensorRT inference-optimization toolkit, extended for transformer-specific patterns).

**Best suited for:** scenarios where you're committed to NVIDIA hardware and want to extract the absolute maximum performance, and are willing to invest more engineering effort into build/compilation steps (TensorRT-LLM typically requires compiling model-specific optimized "engines" ahead of time, which is a more involved setup process than vLLM's more dynamic, Python-native approach).

**Tradeoff vs vLLM:** often achieves the highest raw performance on supported NVIDIA hardware, but with a steeper setup/compilation complexity and less flexibility for rapidly swapping between different models compared to vLLM's more dynamic loading.

## 6.3 llama.cpp

**Core innovation:** a C/C++ implementation designed to run LLM inference efficiently on **CPUs** (and also supports GPU acceleration), using the GGUF model format and heavy quantization support, with minimal dependencies — making it possible to run LLMs on consumer laptops, and even phones/edge devices, without requiring a dedicated GPU at all.

**Best suited for:** local/edge/on-device inference, resource-constrained environments, or situations where you specifically need CPU-based inference (no GPU available) — directly relevant to your own SLM Inference Engine project's focus on CPU/mobile deployment.

**Key features:** GGUF quantization formats (a wide range of bit-widths and quantization schemes optimized for this specific runtime), extremely portable (runs on many platforms/architectures with minimal dependencies), active community producing pre-quantized GGUF versions of popular models.

## 6.4 Ollama

**What it actually is:** a user-friendly wrapper/packaging layer built **on top of llama.cpp**, designed to make running LLMs locally as simple as a single command (`ollama run llama3`), handling model downloading, quantization format selection, and providing a simple local API server — prioritizing ease of use and developer experience over exposing low-level performance tuning knobs.

**Best suited for:** quick local development, prototyping, personal/hobbyist use, or situations where ease of setup matters more than squeezing out maximum possible performance — trades some of llama.cpp's raw configurability for a vastly simpler user experience.

## 6.5 Choosing Between Them — Decision Framework

| Framework | Best for | Hardware | Setup complexity |
|---|---|---|---|
| **vLLM** | High-throughput production GPU serving | GPU (NVIDIA/AMD) | Moderate |
| **TensorRT-LLM** | Maximum raw performance on NVIDIA | NVIDIA GPU only | High (compilation step) |
| **llama.cpp** | CPU/edge/on-device inference | CPU (+ optional GPU) | Low-moderate |
| **Ollama** | Easy local development/prototyping | CPU/GPU (via llama.cpp) | Very low |

### 🔧 Activity 6: Serving Frameworks
1. Install Ollama and run a small quantized model locally (`ollama run` a small model). Time how long it takes to generate a fixed-length response, and note your CPU/GPU usage during generation.
2. If you have GPU access, set up vLLM and serve the same (or similarly-sized) model, comparing throughput/latency against your Ollama/llama.cpp result.
3. Read llama.cpp's GGUF format documentation and identify at least 3 different quantization scheme variants it supports (e.g., Q4_0, Q4_K_M, Q5_K_M) — note the general tradeoff pattern between the naming/bit-width and expected quality/size.
4. Write a short comparison (3-4 sentences) of which serving framework you'd choose for: (a) a high-traffic production API serving thousands of requests/second on GPU servers, (b) a privacy-focused on-device mobile app, (c) quick local experimentation while developing a new agent workflow.

---

# 7. MODEL PARALLELISM: TENSOR, PIPELINE, DATA PARALLELISM

## 7.1 Why Parallelism Is Needed At All

Large models (tens to hundreds of billions of parameters) frequently don't fit in the memory of a single GPU, and even when they technically fit, serving/training them efficiently at scale benefits from splitting work across multiple GPUs. Different parallelism strategies split *different things* across devices, and are often combined together in large-scale systems.

## 7.2 Data Parallelism

**How it works:** the **full model** is replicated identically on every GPU, but each GPU processes a different **subset (shard) of the input data/batch** simultaneously. After each device computes results (during training, gradients) on its data shard, the results are aggregated (e.g., averaging gradients) across all devices.

**Best suited for:** situations where the model fits comfortably on a single GPU's memory, but you want to process more data in parallel to speed up training or increase serving throughput — this is the simplest and most common form of parallelism when the model itself isn't the bottleneck.

**Limitation:** doesn't help at all if the model itself is too large to fit on a single GPU — every GPU still needs a full copy of the entire model.

## 7.3 Tensor Parallelism

**How it works:** individual weight matrices/tensors *within a single layer* are split across multiple GPUs — e.g., a large matrix multiplication is divided so that each GPU computes a portion of the output, and the partial results are combined (via communication between GPUs) to produce the final layer output.

**Concretely, for a transformer's attention/FFN layers:** the large weight matrices can be split column-wise or row-wise across GPUs, with each GPU holding only a slice of the full weight matrix — meaning no single GPU needs to store the entire layer's weights, directly solving the "model too big for one GPU" problem at the layer level.

**Key requirement:** because computing a single layer's output now requires combining partial results from multiple GPUs, tensor parallelism requires **very fast interconnect** between GPUs (like NVLink) since communication happens frequently, within every layer's forward pass — this makes it best suited for GPUs within the same physical server/node, where interconnect bandwidth is high, rather than across separate machines connected by slower networking.

## 7.4 Pipeline Parallelism

**How it works:** different **layers** of the model are assigned to different GPUs — e.g., GPU 1 holds layers 1-20, GPU 2 holds layers 21-40, and so on. Data flows through the GPUs sequentially, like an assembly line: GPU 1 processes its layers and passes the intermediate result to GPU 2, which processes its layers and passes onward, etc.

**The "pipeline bubble" problem:** naively, this creates idle time — while GPU 2 is waiting for GPU 1 to finish processing the first piece of data, GPU 2 sits idle, and once data reaches GPU 2, GPU 1 might be idle waiting for the next batch. This idle time is called a "pipeline bubble." Solutions like **micro-batching** (splitting a batch into smaller pieces and feeding them through the pipeline in a staggered fashion, so multiple micro-batches are "in flight" across different pipeline stages simultaneously) reduce this idle time significantly, keeping more GPUs busy more of the time.

**Best suited for:** splitting a model across GPUs (or even across separate nodes/machines, since pipeline parallelism requires much less frequent inter-GPU communication than tensor parallelism — only at the boundary between pipeline stages, not within every layer) when the model is too large for tensor parallelism alone or when interconnect bandwidth between devices is more limited.

## 7.5 Combining Parallelism Strategies

In practice, very large-scale training/serving systems combine **all three** simultaneously (sometimes called "3D parallelism"):
- **Tensor parallelism** *within* a node (across GPUs connected by fast NVLink)
- **Pipeline parallelism** *across* nodes (where slower inter-node networking is less of a bottleneck since communication only happens at stage boundaries)
- **Data parallelism** *across* multiple full replicas of this tensor+pipeline-parallel setup, to scale out to even more total throughput

**Why this layered combination makes sense:** each strategy is applied at the granularity where its specific communication pattern and bandwidth requirements are best matched to the actual hardware topology available — fast, frequent communication (tensor parallelism) stays within a node; slower, infrequent communication (pipeline parallelism) crosses nodes; data parallelism scales out horizontally with minimal communication needed (just periodic gradient aggregation during training, or simply independent request routing during inference serving).

### 🔧 Activity 7: Model Parallelism
1. Diagram all three parallelism strategies for a toy 4-layer model being split across 2 GPUs — show clearly what's different about which piece of the model/data lives on which GPU for each strategy.
2. Explain in your own words why tensor parallelism requires much more frequent inter-GPU communication than pipeline parallelism, and why this makes tensor parallelism better suited to GPUs within the same physical server.
3. Research question: if you have a model too large to fit on a single GPU even in FP16, and you have 8 GPUs available all within one server (fast interconnect), would you reach for tensor parallelism, pipeline parallelism, or both? Justify your reasoning.
4. Explain the "pipeline bubble" problem and how micro-batching helps address it — draw a timeline diagram showing GPU utilization with and without micro-batching in a 3-stage pipeline.

---

# 20 DEEP INTERVIEW QUESTIONS — INFERENCE OPTIMIZATION

**1. Why is LLM inference typically memory-bandwidth bound rather than compute bound, and why does this fact justify quantization as a speedup technique (not just a memory-saving one)?**
During autoregressive generation, the model must load its entire set of weights from GPU memory for every single token generated, but the actual amount of computation (multiply-accumulate operations) per token is relatively small compared to the time spent moving those weights from memory to compute units. Since the GPU's compute units often sit idle waiting for weights to arrive, reducing the bit-width of those weights (via quantization) directly reduces the amount of data that must be moved per token, cutting the dominant bottleneck — meaning quantization speeds up wall-clock inference time, not just reduces storage footprint.

**2. Explain the basic linear quantization formula and what the scale factor and zero-point represent.**
Quantization maps a floating-point value x to an integer via x_q = round(x/s) + z, where s (scale) determines the step size between representable quantized values (essentially, how much real-valued range each integer step covers), and z (zero-point) is an integer offset used to correctly represent value ranges that aren't symmetric around zero. Dequantization reverses this: x̂ = s·(x_q − z), recovering an approximation of the original floating-point value.

**3. What's the difference between per-tensor and per-channel quantization, and why does per-channel typically improve accuracy?**
Per-tensor quantization uses a single scale factor for an entire weight matrix, assuming a roughly uniform value distribution across it. Per-channel (or per-group) quantization uses a separate scale factor for each output channel or small group of weights, better capturing the fact that different channels can have meaningfully different value ranges/distributions — this finer granularity reduces quantization error compared to forcing one scale to fit the entire tensor's full range, at the modest cost of storing more scale factors.

**4. Explain how GPTQ quantizes a layer's weights and why its approach reduces accumulated error compared to naive independent rounding.**
GPTQ quantizes weights within a layer sequentially, one at a time, and after quantizing each weight, it adjusts the remaining not-yet-quantized weights to compensate for the error just introduced, using second-order (Hessian-based) information about how sensitive the layer's output is to each weight. This means errors from early-quantized weights are actively corrected for in later weights before those too get quantized, keeping the overall layer output much closer to the original unquantized output than if every weight were rounded independently with no compensation.

**5. What is AWQ's core insight about which weights matter most, and how does it preserve their effective precision without using mixed-precision hardware?**
AWQ observes that a small percentage of weights — specifically those connected to input channels that produce large-magnitude activations — disproportionately affect output quality, since a quantization error multiplied by a large activation produces a larger absolute output error. Rather than literally storing these weights at higher precision (which would require complex mixed-precision hardware support), AWQ mathematically rescales the salient channels' weights up and their corresponding activations down before quantization, so uniform low-bit quantization preserves more effective precision for the weights that matter most, while the entire computation still uses one hardware-friendly bit-width throughout.

**6. Compare unstructured and structured pruning in terms of achievable compression and practical speedup.**
Unstructured pruning zeros out individually unimportant weights (typically by magnitude), achieving very high sparsity ratios with relatively small quality loss, but the resulting irregular sparsity pattern isn't naturally exploited by standard GPU hardware without specialized sparse-matrix kernels, so storage savings don't automatically translate to inference speedup. Structured pruning removes entire structural units (neurons, attention heads, layers), producing a smaller but still fully dense model compatible with standard hardware/software, making speedup straightforward to realize — at the cost of generally being harder to push to the same aggressive compression ratios without hurting quality, since whole meaningful units are removed rather than cherry-picked individual weights.

**7. Explain knowledge distillation's "soft target" concept and why it transfers more information than training on hard labels alone.**
Instead of training the student model only on one-hot correct-answer labels, distillation also trains it to match the teacher model's full output probability distribution across all classes/tokens. This distribution reveals not just the correct answer but the teacher's relative confidence and which incorrect answers it considers plausible versus implausible ("dark knowledge") — for example, revealing that "rug" is a much more reasonable near-miss than a completely unrelated word — giving the student richer training signal about the relationships between outputs than a single hard label could convey.

**8. What role does temperature play in the distillation loss, and why is a higher temperature often used when computing the teacher's soft targets?**
Temperature scaling divides the logits by T before the softmax; higher T produces a softer, more spread-out probability distribution across all classes rather than a sharp, nearly one-hot distribution. A softer distribution reveals more of the relative ranking and "dark knowledge" information among the non-top classes, which is exactly the extra signal distillation is trying to transfer to the student — an extremely peaked (low-temperature) teacher distribution would look almost like a hard label anyway, defeating the purpose of using soft targets in the first place.

**9. Derive/explain the KV-cache memory formula and identify which architectural choice (from the attention variants topic) most directly reduces it.**
KV-cache size = 2 × L × h × d × s × b (layers × KV-heads × head-dim × sequence-length × batch-size, times 2 for storing both K and V). The number of KV-heads (h) is the term most directly reduced by the attention variant choice — Grouped-Query Attention or Multi-Query Attention reduce the number of distinct KV-head projections that must be cached, directly shrinking this term (and thus the total cache size) without needing to touch sequence length, batch size, or layer count.

**10. Explain PagedAttention and the specific memory problem it solves, drawing the analogy to OS-level memory management.**
Naive KV-cache implementations pre-allocate a large contiguous memory block per sequence sized for its maximum possible length, wasting significant memory when actual sequences are shorter or finish early (fragmentation, similar to internal fragmentation in classic contiguous memory allocation). PagedAttention, borrowing directly from OS virtual memory paging, divides the KV-cache into small fixed-size blocks and maintains a per-sequence block table mapping logical positions to non-contiguous physical blocks — memory is only allocated as actually needed, eliminating most fragmentation waste, and as a bonus, enables safely sharing identical blocks (like a common prompt prefix) across multiple sequences.

**11. Why does prefix caching provide especially large benefits for applications with a fixed system prompt used across many requests?**
The prefill phase (processing the initial prompt before generation starts) is computationally expensive and scales with prompt length. If many requests share an identical prefix (e.g., a fixed system prompt), the KV-cache values for that shared portion are mathematically identical across all those requests — computing them once and reusing them for every subsequent request eliminates a large, entirely redundant chunk of computation that would otherwise be repeated from scratch on every single request.

**12. Compare static batching and continuous batching, and explain precisely where the throughput advantage of continuous batching comes from.**
Static batching processes a fixed group of requests together and only returns any results once every request in that batch has finished generating, meaning shorter requests finish early but their GPU slot sits idle (or processes wasted padding) until the longest request in the batch completes. Continuous batching operates at the individual decoding-step level, immediately evicting any sequence that finishes and admitting a new waiting request into that freed slot right away — this keeps the effective batch size, and therefore GPU utilization, consistently high rather than periodically dropping as requests within a static batch finish at different times.

**13. Why do continuous batching and PagedAttention complement each other particularly well in a system like vLLM?**
Continuous batching constantly adds and removes sequences from the active batch as generation proceeds, meaning the set of sequences needing KV-cache memory is highly dynamic and unpredictable in advance. PagedAttention's block-based, non-contiguous memory allocation is specifically well-suited to this dynamic pattern — new sequences can be allocated blocks on demand as they join the batch, and finished sequences' blocks can be immediately freed and reused, without needing large pre-reserved contiguous memory regions that would conflict with a constantly-changing set of active sequences.

**14. Explain speculative decoding step by step, and explain why it produces zero quality loss despite using a smaller, less capable draft model.**
A small draft model proposes several candidate tokens autoregressively; the large target model then verifies all of these candidates in a single parallel forward pass (which costs roughly the same wall-clock time as generating just one token normally, due to the memory-bandwidth-bound nature of inference). A mathematically-derived acceptance/rejection sampling rule then decides which draft tokens to keep — this rule is specifically constructed so the resulting distribution of accepted tokens is provably identical to what the target model alone would have produced token-by-token; tokens are never accepted based on the draft model's judgment being "good enough," they're accepted or rejected via a procedure that exactly preserves the target model's true output distribution.

**15. Why does verifying k draft tokens in one parallel forward pass cost roughly the same wall-clock time as generating just 1 token normally?**
Because LLM inference is memory-bandwidth bound, the dominant cost of a forward pass is loading the model's weights from memory, which happens once per forward pass regardless of how many tokens are being processed within that pass. Processing a handful of extra tokens (k, in this case) within the same forward pass adds relatively little additional compute cost on top of that fixed weight-loading cost, so verifying multiple tokens in parallel is nearly as cheap, time-wise, as verifying just one.

**16. What makes a draft model well-suited or poorly-suited for speculative decoding with a given target model?**
A good draft model needs to be fast (small parameter count, quick to run autoregressively) but also well-aligned in behavior with the target model, so that its proposed tokens are frequently the same as what the target model would have generated — often achieved by using a smaller model from the same family or a distilled version of the target model. A poorly-matched draft model (fast but frequently wrong) results in low acceptance rates, meaning most speculative tokens get rejected and the process falls back to slow one-at-a-time generation frequently, providing little to no actual speedup despite the draft model's low individual cost.

**17. Compare vLLM and TensorRT-LLM in terms of design philosophy and typical use case.**
vLLM prioritizes ease of use, dynamic model loading, and Python-native flexibility, built around PagedAttention and continuous batching to maximize throughput on GPU serving with relatively straightforward setup. TensorRT-LLM is built specifically to extract maximum raw performance from NVIDIA hardware via deep, hardware-specific kernel optimization and ahead-of-time compilation into optimized "engines," at the cost of a more involved setup/compilation process and less flexibility for rapidly swapping between different models — chosen when squeezing out the absolute best latency/throughput on committed NVIDIA infrastructure justifies the added engineering complexity.

**18. Why is llama.cpp particularly relevant for CPU or edge/mobile deployment, and what enables this?**
llama.cpp is implemented in C/C++ with minimal dependencies and heavy support for aggressive quantization (via the GGUF format), specifically engineered to run efficient inference on CPUs (in addition to optional GPU acceleration) rather than requiring dedicated GPU hardware. This combination of a lightweight, portable runtime and a wide range of quantization options makes it possible to run reasonably capable LLMs on consumer laptops, and even resource-constrained edge/mobile devices, where a dedicated GPU and the memory/power budget for full-precision inference simply aren't available.

**19. Explain the difference between tensor parallelism and pipeline parallelism, and why tensor parallelism requires faster interconnect.**
Tensor parallelism splits individual weight matrices within a single layer across multiple GPUs, meaning computing even one layer's output requires combining partial results from multiple GPUs via communication — this happens within every layer's forward pass, requiring very frequent, low-latency communication, which is why it's best suited to GPUs connected by fast interconnect like NVLink within the same server. Pipeline parallelism instead assigns entire different layers to different GPUs, with data flowing sequentially between them like an assembly line — communication only happens at the boundary between pipeline stages (much less frequently than within-layer tensor-parallel communication), making it tolerant of slower inter-node networking and suitable for spanning across separate machines.

**20. You need to deploy a 70B parameter model for a low-latency chat application expecting variable-length, bursty traffic from many concurrent users, on a multi-GPU NVIDIA server. Walk through which techniques from this material you'd combine and why.**
Given multi-GPU NVIDIA hardware and a model too large for a single GPU, use tensor parallelism to split the model across GPUs within the server (justified by fast NVLink interconnect). Use GQA (if the model architecture supports it, which most modern models do) to keep the KV-cache manageable. Serve via vLLM (or TensorRT-LLM if maximum raw performance justifies the added setup complexity) to get PagedAttention (minimizing KV-cache fragmentation and enabling prefix sharing if there's a common system prompt) plus continuous batching (essential for the described bursty, variable-length traffic pattern, since it keeps GPU utilization high despite requests finishing at very different times). Apply quantization (AWQ or GPTQ INT4) if further memory/throughput headroom is needed, accepting a small, well-studied quality tradeoff. Consider speculative decoding with a smaller same-family draft model if further latency reduction is needed and a suitable draft model is available, since it provides speedup with zero quality loss. This combination directly addresses the stated constraints: large model size (tensor parallelism, quantization), variable/bursty traffic (continuous batching), and low-latency requirements (PagedAttention, speculative decoding).

---

# INDEX / REVISION SHEET

## 1. Quantization
- Memory-bandwidth-bound inference → quantization speeds up wall-clock time, not just memory
- Formula: x_q = round(x/s) + z; dequant: x̂ = s(x_q − z)
- Per-channel > per-tensor for accuracy (finer-grained scale factors)
- **GPTQ**: sequential, error-compensated, Hessian-based, needs calibration data, slower to quantize
- **AWQ**: activation-aware channel rescaling, no backprop needed, faster to quantize, popular in production

## 2. Pruning & Distillation
- **Unstructured pruning**: high sparsity possible, needs special sparse kernels for real speedup
- **Structured pruning**: removes whole units, dense output, easy speedup, less aggressive compression
- **Distillation**: student learns from teacher's soft targets (full probability distribution), not just hard labels
- Temperature softens teacher distribution, revealing more "dark knowledge" for the student to learn from
- Core technique behind creating efficient SLMs from large models

## 3. KV-Cache Optimization
- KV-cache size = 2 × L × h × d × s × b — can exceed model weight size at scale
- **PagedAttention**: OS-paging-inspired block-based cache management; eliminates fragmentation, enables prefix sharing (core of vLLM)
- **Prefix caching**: reuse KV-cache for shared prompt prefixes across requests, skips redundant prefill compute
- **Sliding window / attention sink**: bound cache size for long contexts; keep first few tokens even when sliding, per StreamingLLM findings

## 4. Batching
- **Static batching**: waits for entire batch to finish; short requests waste idle GPU time
- **Continuous/dynamic batching**: iteration-level eviction/admission; keeps GPU utilization consistently high; core of vLLM/TensorRT-LLM throughput advantage

## 5. Speculative Decoding
- Small draft model proposes k tokens → large target model verifies all k in one parallel forward pass
- Verification cost ≈ cost of generating 1 token normally (memory-bandwidth-bound reasoning)
- Acceptance/rejection sampling guarantees mathematically identical output distribution to target-model-alone generation — zero quality loss
- Draft model choice matters: needs to be fast AND well-aligned with target model for good acceptance rate

## 6. Serving Frameworks
- **vLLM**: PagedAttention + continuous batching, GPU, Python-native, moderate setup
- **TensorRT-LLM**: NVIDIA-specific, max raw performance, compilation step, high setup complexity
- **llama.cpp**: CPU/edge-focused, GGUF quantization, minimal dependencies, very portable
- **Ollama**: user-friendly wrapper over llama.cpp, easiest setup, best for local dev/prototyping

## 7. Model Parallelism
- **Data parallelism**: full model replicated, data split across GPUs — doesn't help if model itself is too big
- **Tensor parallelism**: weight matrices split within a layer across GPUs — needs fast interconnect (NVLink), within-node
- **Pipeline parallelism**: different layers on different GPUs, assembly-line flow — less frequent communication, can span nodes; "pipeline bubble" idle time addressed via micro-batching
- Large systems combine all three: tensor (within node) + pipeline (across nodes) + data (scale-out replicas)