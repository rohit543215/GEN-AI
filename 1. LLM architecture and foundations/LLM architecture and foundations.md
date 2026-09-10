# LLM Architecture & Foundations — Complete Masterclass

---

# 1. TOKENIZATION

## 1.1 Why Tokenization Exists

Neural networks can't process raw text — they need numbers. Tokenization is the process of breaking text into smaller units (tokens) and mapping each to an integer ID. The core challenge: do you tokenize by word, character, or something in between?

- **Word-level tokenization**: `["I", "love", "machine", "learning"]` — problem: huge vocabulary, can't handle unseen words (out-of-vocabulary/OOV problem)
- **Character-level tokenization**: `["I", " ", "l", "o", "v", "e", ...]` — problem: sequences become very long, loses semantic meaning of words
- **Subword tokenization** (the modern solution): breaks rare words into meaningful pieces while keeping common words whole. E.g. `"unhappiness"` → `["un", "happi", "ness"]`

This gives you the best of both worlds: small vocabulary, no OOV problem (any word can be built from subword pieces or even individual characters), and reasonable sequence lengths.

## 1.2 Byte Pair Encoding (BPE)

**How it works:**
1. Start with a vocabulary of individual characters (or bytes)
2. Count the frequency of every adjacent pair of tokens in your training corpus
3. Merge the most frequent pair into a new single token, add it to the vocabulary
4. Repeat steps 2-3 for a fixed number of merges (this determines final vocab size)

**Worked example:**
Corpus: `"low low low lower lowest"`

Start: characters → `l o w`, `l o w`, `l o w`, `l o w e r`, `l o w e s t`

Most frequent pair: `(l, o)` appears 5 times → merge into `lo`
Now: `lo w`, `lo w`, `lo w`, `lo w e r`, `lo w e s t`

Next most frequent: `(lo, w)` appears 5 times → merge into `low`
Now: `low`, `low`, `low`, `low e r`, `low e s t`

Continue merging (`e r` → `er`, etc.) until you hit your target vocabulary size. The final vocabulary contains characters, common subwords, and full words — determined entirely by frequency in your training data.

**Why this matters for LLMs:** GPT-2 and GPT-3 use BPE (specifically byte-level BPE, operating on raw UTF-8 bytes so *any* possible input, even emojis or unusual Unicode, can always be tokenized).

## 1.3 WordPiece

Very similar to BPE, but instead of merging the *most frequent* pair, it merges the pair that **maximizes the likelihood of the training data** when added to the vocabulary — essentially picking merges based on a language-model-style scoring function rather than raw frequency.

**Key formula (merge score):**
$$\text{score}(a,b) = \frac{\text{freq}(ab)}{\text{freq}(a) \times \text{freq}(b)}$$

This favors merging pairs that occur together much more often than you'd expect if `a` and `b` were independent — a signal that they form a meaningful unit.

**Used by:** BERT and its variants.

## 1.4 SentencePiece

SentencePiece isn't a different merging algorithm per se — it's a tokenization *framework* that treats the input text as a raw stream of Unicode characters (including spaces!) rather than pre-splitting on whitespace first. It can implement BPE or a Unigram Language Model algorithm underneath.

**Key innovation:** By treating spaces as a regular character (often represented as `▁`), SentencePiece is **language-agnostic** — it works just as well for languages without whitespace-based word boundaries (like Japanese or Chinese) as it does for English.

**Unigram Language Model (the other algorithm SentencePiece supports):**
- Start with a large vocabulary of candidate subwords
- Iteratively remove the subwords that least hurt the overall corpus likelihood when removed, until reaching the target vocab size
- This is the *opposite* direction of BPE (top-down pruning vs bottom-up merging)

**Used by:** T5, LLaMA, many multilingual models.

### 🔧 Activity 1: Tokenization
1. Install the `tokenizers` library (`pip install tokenizers`) and train a small BPE tokenizer from scratch on a paragraph of your own text. Print the resulting vocabulary and merge rules.
2. Take the word `"unbelievably"` and manually tokenize it using your trained BPE tokenizer. Does it split into meaningful subword pieces?
3. Compare: tokenize the same sentence using `tiktoken` (OpenAI's BPE tokenizer) vs. a BERT WordPiece tokenizer (`transformers` library, `BertTokenizer`). Count how many tokens each produces for the same sentence — which is more efficient (fewer tokens)?
4. Find a word that gets split very differently between GPT's tokenizer and BERT's tokenizer, and explain why (hint: try a rare technical term or a made-up word).

---

✅ **STOPPED HERE — 2026-09-10**


# 2. TRANSFORMER DECODER ARCHITECTURES (GPT-STYLE)

## 2.1 Encoder-Decoder vs Decoder-Only

The original Transformer (Vaswani et al., 2017) had two halves:
- **Encoder**: processes the full input bidirectionally (sees all tokens at once, both past and future)
- **Decoder**: generates output autoregressively (one token at a time, only sees past tokens)

**GPT-style models are decoder-only** — they drop the encoder entirely and just stack decoder blocks. This makes sense because GPT's task is pure text generation: given everything so far, predict the next token. There's no separate "input" to encode; the prompt and the generated continuation are all just one growing sequence.

## 2.2 The Decoder Block, Piece by Piece

A single GPT decoder block contains, in order:
1. **Normalization** (pre-norm, applied before each sub-layer in modern architectures). The original Transformer and GPT-2 use standard **LayerNorm**; most modern open LLMs (LLaMA, Mistral, and others) use **RMSNorm** instead — a simplified variant that skips mean-centering and only rescales activations by their root-mean-square, which is computationally cheaper and empirically works just as well.
2. **Masked Multi-Head Self-Attention** — "masked" because each token can only attend to itself and previous tokens, never future ones (this is what makes it autoregressive/causal)
3. **Residual connection** (add the block's input back to its output)
4. **Normalization** again (same LayerNorm-vs-RMSNorm choice as step 1)
5. **Feed-Forward Network (FFN)** — the classic GPT-2-style FFN uses two linear layers with a GELU non-linearity in between, expanding to a larger hidden dimension and back. Most modern LLMs (LLaMA, Mistral) instead use a **SwiGLU** FFN, which needs **three** weight matrices rather than two: a gate projection and an up projection (both applied to the input, with the gate passed through a SiLU/Swish activation and multiplied elementwise against the up projection), followed by a down projection back to the model dimension. This gated design generally outperforms a plain GELU FFN at the same parameter budget.
6. **Residual connection** again

**Note — weight tying:** many of these architectures (GPT-2, LLaMA, and others) tie the input token embedding matrix and the output LM head (the final projection to vocabulary logits), so the two share the same weight matrix. This meaningfully cuts parameter count for large vocabularies and acts as a mild regularizer.

This block is stacked N times (GPT-3 has 96 layers, for example).

## 2.3 Causal Masking — the Key Mechanism

Since the decoder must never "peek" at future tokens during training (otherwise it would trivially learn to copy the answer), a **causal mask** is applied to the attention scores before the softmax: positions corresponding to future tokens get set to `-∞` so their softmax probability becomes `0`.

```
Attention scores for token at position i (rows = query position, cols = key position):

       tok1  tok2  tok3  tok4
tok1 [ 0.9  -∞    -∞    -∞  ]
tok2 [ 0.3   0.5  -∞    -∞  ]
tok3 [ 0.1   0.2   0.6  -∞  ]
tok4 [ 0.2   0.1   0.3   0.4]
```

Notice the upper-triangular part is `-∞` — token 1 can never attend to tokens 2, 3, or 4, only to itself.

## 2.4 Why This Enables Both Training Efficiency AND Generation

**During training:** the entire sequence is fed in at once, and because of the causal mask, the model computes the "predict next token" loss for *every position simultaneously* in a single forward pass — this is called **teacher forcing** and it's why training is parallelizable despite the task being inherently sequential.

**During inference/generation:** tokens must be generated one at a time (autoregressively) since each new token depends on all previous ones — this is why LLM inference is fundamentally sequential and slower than training, and exactly why techniques like KV-caching and speculative decoding (which you'll cover in the Inference Optimization section) exist.

### 🔧 Activity 2: Decoder Architecture
1. Draw (on paper or in a diagram tool) a single GPT decoder block, labeling every sub-layer and where residual connections attach.
2. In PyTorch, implement a causal attention mask for a sequence length of 6 using `torch.triu` or `torch.tril`. Print it out and verify it matches the upper-triangular `-∞` pattern shown above.
3. Explain in your own words (write 3-4 sentences): why can't you just use a regular encoder (bidirectional attention) for a text-generation model like GPT?
4. Load a small pretrained GPT-2 model via HuggingFace `transformers`, and print out `model.config` — identify the number of layers, hidden size, and number of attention heads.

---

# 3. ATTENTION MECHANISM

## 3.1 The Core Idea

Attention lets each token "look at" every other token in the sequence (subject to the causal mask) and decide how much to weigh each one when building its own representation. Instead of treating all context equally, attention lets the model learn: "when processing this word, which other words actually matter?"

**Real-world analogy:** Reading the sentence "The trophy doesn't fit in the suitcase because **it** is too big" — to understand what "it" refers to, you (a human reader) pay *attention* to "trophy" and "suitcase" more than other words, and decide "it" = "trophy" based on context. Self-attention is the mechanism that lets a model learn to do exactly this, automatically, from data.

## 3.2 Queries, Keys, and Values

Every input token embedding is projected into three different vectors using learned weight matrices:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

- **Query (Q)**: "what am I looking for?" — represents the current token's request for information
- **Key (K)**: "what do I contain?" — represents what each token offers, to be matched against queries
- **Value (V)**: "what do I actually communicate?" — the actual content that gets aggregated once attention weights are computed

**Analogy:** think of a search engine. Your search term is the **Query**. Every webpage has **Keys** (metadata/tags describing what it's about) that get matched against your query. Once a match is found, you retrieve the **Value** (actual page content).

## 3.3 The Scaled Dot-Product Attention Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Breaking this down step by step:**

1. **$QK^T$** — compute the dot product between every query and every key. This produces a score matrix where entry `(i,j)` represents "how relevant is token j's key to token i's query." High dot product = high similarity/relevance.

2. **$\frac{1}{\sqrt{d_k}}$** — scale down by the square root of the key dimension. Why? Without scaling, as $d_k$ grows large, dot products grow large in magnitude, pushing the softmax into regions with extremely small gradients (saturation). Dividing by $\sqrt{d_k}$ keeps the variance of the dot products roughly constant regardless of dimension, keeping softmax gradients healthy.

3. **softmax(...)** — convert the scaled scores into a probability distribution (all values between 0-1, summing to 1) across the key positions. This gives you the "attention weights" — how much to weigh each token.

4. **... × V** — use these attention weights to compute a weighted sum of the Value vectors. The output for each token is now a blend of all the tokens it attends to, weighted by relevance.

## 3.4 Multi-Head Attention

Instead of computing attention once with the full embedding dimension, multi-head attention splits Q, K, V into `h` smaller "heads," each with dimension $d_k = d_{model}/h$, and computes attention independently in each head, then concatenates the results:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W_O$$

$$\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)$$

**Why multiple heads instead of one big attention computation?** Each head can specialize in learning different types of relationships — one head might learn to track syntactic dependencies (subject-verb agreement), another might track long-range coreference (like the "trophy/suitcase" example), another might focus on adjacent-word relationships. A single attention computation with the full dimension would be forced to average all these patterns together into one representation, losing this specialization.

### 🔧 Activity 3: Attention Mechanism
1. By hand, compute scaled dot-product attention for a toy example: 3 tokens, each with a 2-dimensional embedding. Pick simple numbers, compute Q, K, V (you can use identity-like weight matrices for simplicity), compute the attention scores, apply softmax, and get the final output.
2. Implement scaled dot-product attention from scratch in PyTorch/NumPy (just the formula — no need for full multi-head yet). Verify it against `torch.nn.functional.scaled_dot_product_attention`.
3. Implement multi-head attention from scratch, splitting an embedding of dimension 512 into 8 heads. Print the shape of Q, K, V at each stage to make sure your reshaping/splitting logic is correct.
4. Answer in your own words: why do we divide by $\sqrt{d_k}$ specifically, and not just $d_k$ or a fixed constant? (Hint: think about variance of the dot product as dimension grows.)

---

# 4. POSITIONAL EMBEDDINGS

## 4.1 Why Position Information Is Needed At All

Attention, as described above, is **permutation-invariant** — it computes weighted sums based on content similarity (Q·K), with no inherent notion of *order*. If you shuffled the tokens in a sentence, plain self-attention would produce the same set of outputs, just shuffled — it has no idea that "dog bites man" and "man bites dog" mean different things based on word order.

So we need to inject positional information into the model somehow.

## 4.2 Absolute Positional Embeddings (the original approach)

**Learned absolute positions:** simply add a learned embedding vector for each position (0, 1, 2, ... up to max sequence length) directly to the token embedding before feeding into the transformer:

$$x_i = \text{TokenEmbedding}(t_i) + \text{PositionEmbedding}(i)$$

**Sinusoidal positional embeddings** (the original Transformer paper's approach, not learned but computed with a fixed formula):

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**Why sine/cosine specifically?** These functions have a useful property: the positional encoding for position `pos+k` can be expressed as a *linear function* of the encoding at position `pos`, which theoretically helps the model learn to attend by relative position, even though the encoding itself is absolute. They also naturally generalize to sequence lengths not seen during training (since sine/cosine are defined for any real number).

**Limitation of absolute embeddings:** they struggle to generalize well beyond the maximum sequence length seen during training, and don't naturally capture *relative* distance between tokens (which often matters more than exact position — "the word two positions back" is a more useful signal than "this is token #47").

## 4.3 RoPE (Rotary Positional Embeddings)

RoPE, used in LLaMA, GPT-NeoX, and most modern open LLMs, takes a fundamentally different approach: instead of *adding* a positional vector, it **rotates** the query and key vectors by an angle proportional to their position, using 2D rotation matrices applied to pairs of dimensions.

**Core idea:** for a 2D subspace of the embedding, rotate the vector by angle $m\theta$ where `m` is the position:

$$f(x, m) = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

**Why rotation is elegant here:** when you compute the dot product between a rotated query at position `m` and a rotated key at position `n`, the result depends *only on the relative distance* `(m-n)`, not on the absolute positions themselves. This is a beautiful mathematical property — relative position information emerges naturally from an absolute-looking rotation operation, giving you the best of both worlds (implementation simplicity of absolute encoding + the relative-position benefits that matter for generalization).

**Practical benefit:** RoPE generalizes much better to longer sequences than trained on (better length extrapolation) compared to learned absolute embeddings.

## 4.4 ALiBi (Attention with Linear Biases)

ALiBi takes yet another approach: it doesn't modify the Q/K vectors at all. Instead, it adds a **static, non-learned bias** directly to the attention scores, penalizing attention between tokens proportional to their distance:

$$\text{score}(i,j) = q_i \cdot k_j - m \cdot |i - j|$$

Where `m` is a head-specific slope (different heads get different penalty strengths — some heads focus more on local context with a steep penalty, others attend more globally with a gentle penalty).

**Why this is simple and effective:** since it's just a fixed linear penalty added at attention-score time (no vectors to rotate, no embeddings to learn), it's computationally cheap and has been shown to extrapolate to much longer sequences than seen during training — a model trained on 1024-token sequences with ALiBi can often perform reasonably on 4096+-token sequences at inference time, something absolute embeddings handle very poorly.

### 🔧 Activity 4: Positional Embeddings
1. Implement sinusoidal positional encoding from scratch in NumPy for a sequence length of 50 and embedding dimension 64. Plot it as a heatmap — do you see the wave-like pattern across dimensions?
2. Implement the RoPE rotation for a toy 2D vector at positions 0, 1, 2, 3. Verify that the dot product between the rotated vectors at positions 2 and 5 depends only on their *difference* (3), not their absolute values, by also computing the dot product between rotated vectors at positions 10 and 13 — confirm you get the same dot product both times.
3. Read the HuggingFace implementation of RoPE in the LLaMA model source code and identify where the rotation is applied to Q and K.
4. Explain in your own words: why does ALiBi generalize better to longer sequences than learned absolute positional embeddings?

---

# 5. ATTENTION VARIANTS (MULTI-QUERY, GROUPED-QUERY ATTENTION)

## 5.1 The Problem These Solve: KV-Cache Memory

During autoregressive generation, to avoid recomputing attention over the entire sequence at every new token, models cache the previously computed Key and Value tensors (the **KV-cache**). This makes generation much faster — but the cache grows linearly with sequence length and consumes significant GPU memory, especially with many attention heads.

In standard **Multi-Head Attention (MHA)**, every one of the `h` heads has its own separate K and V projections — meaning the KV-cache size scales with the number of heads. For a large model with many heads and long context windows, this becomes a serious memory bottleneck during inference.

## 5.2 Multi-Query Attention (MQA)

**The idea:** keep separate Query heads (still `h` of them, preserving representational diversity for queries), but share a **single** Key and Value projection across *all* heads.

$$\text{head}_i = \text{Attention}(QW_Q^i, K, V) \quad \text{for } i = 1...h$$

Notice: `K` and `V` no longer have a superscript `i` — every head uses the *same* K and V.

**Effect:** the KV-cache size shrinks by a factor of `h` (the number of heads) — a massive memory saving, directly speeding up inference and allowing larger batch sizes / longer contexts to fit in memory.

**Tradeoff:** because all heads now share the same K/V, there's less representational diversity, which can hurt model quality slightly compared to full MHA.

## 5.3 Grouped-Query Attention (GQA)

**The idea:** a middle ground between MHA (every head has its own K/V) and MQA (all heads share one K/V). GQA divides the `h` query heads into `g` groups, and each group shares one K/V projection.

- If `g = h` → this is just standard MHA (every head is its own group)
- If `g = 1` → this is just MQA (all heads share one group)
- Somewhere in between (e.g. `g = 8` groups for `h = 32` heads) gives a tunable tradeoff between memory savings and model quality

**Why this matters in practice:** GQA is used in LLaMA 2/3, Mistral, and most modern production LLMs because it recovers most of the quality of full MHA while still getting the large majority of MQA's memory/speed benefits — empirically, going from MHA to GQA barely hurts quality, but going all the way to MQA can measurably hurt quality on some tasks.

### 🔧 Activity 5: Attention Variants
1. Diagram all three attention types (MHA, MQA, GQA) side by side, showing how many separate K/V projections each has for a model with 8 query heads.
2. Calculate: if a model has 32 attention heads, each head dimension 128, and a sequence length of 4096, compute the KV-cache memory size (in number of floating point values) for standard MHA vs MQA vs GQA with 8 groups. (Formula: `2 (K and V) × num_kv_heads × head_dim × seq_len` per layer)
3. Look up which attention variant LLaMA 2 7B, LLaMA 2 70B, and Mistral 7B use respectively — note any differences between model sizes.
4. Explain in your own words: why does MQA save memory but potentially hurt quality, while GQA is often described as "the best of both worlds"?

---

# 20 DEEP INTERVIEW QUESTIONS — LLM ARCHITECTURE & FOUNDATIONS

**1. Why do LLMs use subword tokenization instead of word-level or character-level tokenization?**
Word-level tokenization creates massive vocabularies and can't handle unseen/rare words (OOV problem). Character-level avoids OOV but creates very long sequences that lose semantic meaning per token and are computationally expensive to process. Subword tokenization (BPE/WordPiece/Unigram) balances both: common words stay as single tokens, rare words decompose into meaningful subword pieces, keeping vocabulary size manageable (~30k-100k) and sequences reasonably short while eliminating the OOV problem entirely (any string can be built from the base vocabulary, down to individual bytes/characters if needed).

**2. Walk through exactly how BPE builds its vocabulary.**
Start with a base vocabulary of individual characters/bytes. Count frequency of every adjacent symbol pair across the training corpus. Merge the most frequent pair into a new token and add it to the vocabulary. Repeat this process for a fixed number of merge operations (which determines final vocabulary size). The result is a vocabulary containing individual characters, common subword units, and frequently-occurring whole words, all determined purely by co-occurrence statistics in the training data.

**3. What's the key difference between BPE and WordPiece merge selection?**
BPE always merges the pair with the highest raw frequency. WordPiece instead merges the pair that maximizes the likelihood of the training data — computed via score(a,b) = freq(ab)/(freq(a)×freq(b)) — which favors pairs that co-occur far more often than chance would predict, rather than simply the most common pair overall.

**4. Why is GPT decoder-only rather than encoder-decoder like the original Transformer?**
GPT's task is autoregressive text generation — predicting the next token given everything so far. There's no separate "source" sequence to encode bidirectionally and a "target" sequence to generate, as there is in translation (the original Transformer's use case). Since the entire input and output are just one continuously growing sequence, a decoder-only architecture (which processes tokens causally/autoregressively) is a more natural and simpler fit, and it scales well for the general-purpose "predict next token" pretraining objective.

**5. Explain causal masking and why it's necessary during training.**
Causal masking sets attention scores for any "future" token position to negative infinity before the softmax, so their resulting attention weight becomes zero — each token can only attend to itself and prior tokens. This is necessary because during training, the entire target sequence is fed in at once for parallelization (teacher forcing); without masking, a token could "cheat" by attending to the very token it's supposed to predict, since that token is already present in the input sequence during training.

**6. Why can transformer training be parallelized across the sequence, but inference cannot?**
During training, the full ground-truth sequence is already available, so with causal masking applied, the model can compute predictions (and loss) for every position in the sequence simultaneously in one forward pass. During inference, the model doesn't know future tokens yet — each new token must be generated based on all previously generated tokens, and that new token then needs to be fed back in to generate the next one — making generation inherently sequential, one token at a time.

**7. Derive/explain the scaled dot-product attention formula, including why the scaling factor exists.**
Attention(Q,K,V) = softmax(QKᵀ/√dk)V. QKᵀ computes similarity scores between every query-key pair. Without scaling, as the key dimension dk grows, the variance of the dot products grows proportionally to dk, pushing values into the extreme tails of the softmax function where gradients vanish (saturation). Dividing by √dk normalizes the variance back to approximately 1 regardless of dimension, keeping the softmax in a well-behaved gradient region, which stabilizes training.

**8. What do Query, Key, and Value represent conceptually, and why do we need three separate projections instead of using the raw embeddings directly?**
Query represents what the current token is "looking for," Key represents what each token "offers" for matching against queries, and Value represents the actual content to be aggregated once relevance is determined. Using three separate learned linear projections (rather than the same embedding for all three roles) lets the model learn distinct representations optimized for each role — e.g., the optimal vector for "being searched for" is not necessarily the optimal vector for "being the content that gets retrieved."

**9. Why does multi-head attention outperform a single attention computation with the same total dimension?**
A single large attention computation is forced to average all relationship patterns (syntax, long-range coreference, local adjacency, etc.) into one shared representation. Multiple smaller heads allow different heads to specialize — empirically, different heads do learn to track different linguistic phenomena (e.g., some heads focus on adjacent tokens, others on long-range dependencies) — and concatenating these specialized representations captures more nuanced relationships than one head could alone, at the same total parameter/compute budget.

**10. Why is attention inherently permutation-invariant, and what problem does this create?**
Attention computes outputs as a weighted sum over Value vectors based on Query-Key similarity — this computation has no built-in concept of sequence order; shuffling the input tokens would just shuffle the corresponding outputs identically, with the same relative weightings. This means without additional information, a transformer literally cannot distinguish "dog bites man" from "man bites dog," since both produce the same set of (permuted) attention computations. This necessitates explicitly injecting positional information into the model.

**11. Compare absolute positional embeddings, RoPE, and ALiBi in terms of how they inject positional information.**
Absolute embeddings add a position-specific vector directly to the token embedding before the transformer layers. RoPE rotates the Query and Key vectors by an angle proportional to their position, so that the resulting dot product between two rotated vectors depends mathematically only on their relative distance. ALiBi doesn't touch the Q/K vectors at all — instead it adds a fixed linear penalty directly to the attention scores, proportional to the distance between the two token positions, with different penalty strengths (slopes) per attention head.

**12. Why does RoPE tend to generalize better to longer sequences than learned absolute positional embeddings?**
Learned absolute embeddings only have parameters for positions seen during training (e.g., up to position 2048) — anything beyond that has no learned representation and behaves unpredictably. RoPE, being based on a continuous rotation formula rather than learned lookup vectors, can mathematically extend to any position, and because the resulting attention naturally encodes *relative* rather than absolute position, the model has learned a pattern (attending based on relative distance) that remains meaningful even at distances beyond the training length.

**13. What is the key mathematical property of RoPE that makes the relative-position behavior "emerge" from an absolute-looking rotation?**
The dot product between two vectors rotated by angles proportional to their respective positions m and n depends only on (m − n), not on m or n individually — a consequence of how rotation matrices compose (rotating by angle mθ then taking a dot product with a vector rotated by nθ is mathematically equivalent to a rotation by (m−n)θ applied once). This means even though each vector is rotated based on its own absolute position, the attention score that results is a function purely of relative distance.

**14. What is the KV-cache, and why does it matter for LLM inference speed?**
During autoregressive generation, the Key and Value vectors for all previously generated tokens don't change — recomputing them at every generation step would be wasteful. The KV-cache stores these computed K/V tensors so that at each new generation step, the model only needs to compute Q/K/V for the *newest* token and can reuse the cached K/V for all prior tokens, dramatically reducing redundant computation. Without KV-caching, generating each new token would require reprocessing the entire sequence from scratch, making generation quadratically expensive in sequence length instead of roughly linear.

**15. Explain Multi-Query Attention (MQA) and the tradeoff it makes.**
MQA keeps separate Query projections for each of the h attention heads (preserving query-side diversity) but uses a single shared Key and Value projection across all heads, instead of each head having its own K/V. This shrinks the KV-cache size by a factor of h, substantially reducing memory usage and speeding up inference, especially for long contexts or large batch sizes — at the cost of reduced representational diversity in what each head can attend to, which can slightly hurt model quality compared to full multi-head attention.

**16. What is Grouped-Query Attention (GQA), and why has it become the default choice in most modern production LLMs?**
GQA divides the h query heads into g groups, where each group shares a single K/V projection — a middle ground between full MHA (g=h, every head has its own K/V) and MQA (g=1, all heads share one K/V). It's become the default in models like LLaMA 2/3 and Mistral because empirically it recovers nearly all the quality of full MHA while still capturing the large majority of MQA's memory and inference-speed benefits, making it a favorable tradeoff point rather than an extreme in either direction.

**17. If a model has 32 query heads but only 8 KV heads, what attention variant is it using, and what does that number "8" represent?**
This is Grouped-Query Attention with 8 groups — the 32 query heads are divided into 8 groups of 4 heads each, and each group of 4 query heads shares one Key/Value projection pair. The "8" represents the number of distinct K/V projections actually stored/computed and cached, versus the 32 separate Query projections.

**18. Why is layer normalization typically applied *before* the attention/FFN sub-layers (pre-norm) in modern LLMs, rather than after (post-norm) as in the original Transformer?**
Pre-norm (applying normalization before each sub-layer, with the residual connection bypassing the normalized computation) leads to more stable gradients during training of very deep networks, because the residual path stays "clean" (unnormalized) throughout the network, allowing gradients to flow backward more directly. Post-norm architectures (as in the original Transformer) can suffer from training instability at greater depths, requiring careful learning rate warmup; pre-norm architectures are generally easier to train stably at the depths modern LLMs use (dozens to over a hundred layers).

**19. What's the difference between how SentencePiece treats spaces compared to a "traditional" whitespace-based tokenizer, and why does this matter for multilingual models?**
Traditional tokenizers first split text on whitespace before applying subword merging, implicitly assuming that words are separated by spaces — a Western-language assumption. SentencePiece treats the entire input as a raw character stream, including spaces (often represented internally as a special character like ▁), and learns subword units directly over this stream. This makes it language-agnostic, working equally well for languages like Japanese or Chinese that don't use whitespace to separate words, since it never assumes space-delimited word boundaries in the first place.

**20. In a system design interview, you're asked to reduce the memory footprint of a 70B parameter LLM's inference server without significantly hurting quality. Name two architectural attention-related choices from this material that could help, and explain the tradeoff of each.**
(1) Switch from Multi-Head Attention to Grouped-Query Attention (or the model was already trained with GQA — you'd need to use a model checkpoint trained this way, since attention variant is a training-time architectural choice, not something you can change post-hoc) — this reduces KV-cache memory substantially with minimal quality loss. (2) Use a positional encoding scheme like RoPE or ALiBi rather than learned absolute embeddings if extending context length is also a goal, since they don't require additional learned parameters per position and handle longer sequences more gracefully — though this is again a training-time architectural decision. In both cases, the key tradeoff is that these are choices baked into the model architecture at pretraining time — you generally can't retrofit a different attention variant onto an already-trained model without retraining or at least fine-tuning, so the real answer in an interview should acknowledge you'd need to select or fine-tune a model that already uses these memory-efficient choices, or apply post-hoc techniques like quantization (a separate lever) if you're stuck with a fixed architecture.

---

# INDEX / REVISION SHEET

## 1. Tokenization
- Subword tokenization solves the word-level OOV problem and the character-level long-sequence problem
- **BPE**: bottom-up, merges most *frequent* adjacent pair repeatedly (GPT-2/3)
- **WordPiece**: bottom-up, merges pair maximizing likelihood score = freq(ab)/(freq(a)·freq(b)) (BERT)
- **SentencePiece**: framework treating text as raw character stream (incl. spaces), language-agnostic, supports BPE or Unigram LM (T5, LLaMA)

## 2. Transformer Decoder (GPT-style)
- Decoder-only: no separate encoder, since generation is one continuously growing sequence
- Block = Norm (LayerNorm in GPT-2, RMSNorm in most modern LLMs) → Masked Multi-Head Self-Attention → Residual → Norm → FFN (GELU, 2 matrices, in GPT-2; SwiGLU, 3 matrices — gate/up/down, in LLaMA/Mistral) → Residual
- Weight tying: input embedding and output LM head often share the same weight matrix (GPT-2, LLaMA)
- Causal mask sets future-token attention scores to -∞ before softmax
- Training parallelizable (teacher forcing, full sequence at once); inference is inherently sequential

## 3. Attention Mechanism
- Q = "what am I looking for," K = "what do I offer," V = "what do I actually contribute"
- Formula: softmax(QKᵀ/√dk)V
- √dk scaling prevents dot-product variance from growing with dimension, avoiding softmax saturation
- Multi-head: splits into h smaller attention computations so different heads can specialize

## 4. Positional Embeddings
- Attention is permutation-invariant by default — needs explicit position information
- **Absolute (learned or sinusoidal)**: added to token embedding; poor length extrapolation
- **RoPE**: rotates Q/K vectors by angle ∝ position; dot product depends only on relative distance; good extrapolation (LLaMA)
- **ALiBi**: adds fixed linear distance-based penalty directly to attention scores; excellent length extrapolation, no learned params

## 5. Attention Variants
- **MHA**: every head has its own K/V — best quality, largest KV-cache
- **MQA**: all heads share one K/V — smallest KV-cache, some quality loss
- **GQA**: query heads split into g groups, each group shares one K/V — tunable middle ground, used in LLaMA 2/3, Mistral
- KV-cache size scales with number of *KV heads*, not query heads — this is the lever these variants pull