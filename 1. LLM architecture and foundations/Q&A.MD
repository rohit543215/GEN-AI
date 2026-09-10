# Top 20 Interview & Learning Questions — LLM Architecture & Foundations

---

## Section A: Conceptual Depth

**1.** You have a corpus of legal documents with very rare technical terms. You're choosing between word-level, character-level, and subword tokenization. Which do you pick and why — walk me through the failure modes of the other two.

**2.** Given a corpus, how would you actually build a BPE vocabulary by hand? Show me the merge steps for `"aaabdaaabac"`.

**3.** Derive why we divide by √d_k in scaled dot-product attention. What goes wrong if we divide by d_k instead?

**4.** Attention is permutation-invariant. Explain precisely what that means mathematically and prove it with a 2-token example.

**5.** RoPE's magic is that an absolute rotation produces relative-position behavior. Prove the key property.

**6.** Why does GQA beat both MHA and MQA in practice? Give me the empirical reasoning, not just "it's a middle ground."

---

## Section B: Applied & Calculation

**7.** Calculate the KV-cache size for a 70B model: 80 layers, 64 query heads, 8 KV heads, head_dim = 128, sequence length = 8192, batch size = 4, fp16.

**8.** You're training a model with a 4096-token context but want it to work at 32K at inference. Which positional encoding scheme do you pick, and how would you extend it?

**9.** You have a 100-token prompt and need to generate 500 tokens. Compare FLOPs (roughly) with and without KV-cache.

**10.** A junior engineer says "let's just increase the number of attention heads from 32 to 64 to make the model smarter." What's wrong with this reasoning?

**11.** Your LLM server keeps OOM-ing at batch size 8 with 8K context. You can't retrain. What are your options, in order of impact?

---

## Section C: Code & Implementation

**12.** Implement causal masking in PyTorch for sequence length 6. Show the mask matrix and explain the shape.

**13.** Implement scaled dot-product attention from scratch (NumPy or PyTorch). Then verify against `F.scaled_dot_product_attention`.

**14.** Implement multi-head attention: d_model = 512, 8 heads. Show the reshape logic and verify output shape.

**15.** Write code to train a BPE tokenizer from scratch on a paragraph. Print vocab and merges.

**16.** Load GPT-2 with HuggingFace and print every architectural detail that matters for the KV-cache size.

---

## Section D: System Design & Judgment

**17.** You're designing a 7B model for a startup with limited GPUs. Walk me through your architectural decisions for tokenization, attention, and positional encoding.

**18.** A teammate says: "We fine-tuned our MHA model on 200K tokens and it now handles long context fine." Where's the flaw?

**19.** Design an inference server for a 70B GQA model, 128K context, serving 100 concurrent users. What's your memory budget?

**20.** You're asked to make an existing MHA model "inference-efficient" without retraining. Which of these can you actually change at inference time, and which require retraining?