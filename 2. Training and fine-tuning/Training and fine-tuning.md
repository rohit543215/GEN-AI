# Training & Fine-Tuning — Complete Masterclass

---

# 1. PRETRAINING OBJECTIVES

## 1.1 Why Pretraining Objectives Matter

Before a model can be fine-tuned for any downstream task, it needs to first learn general language understanding from massive amounts of raw text. The *pretraining objective* is the self-supervised task the model is trained on — self-supervised meaning the "labels" are derived automatically from the raw text itself, with no human annotation needed. The choice of objective fundamentally shapes what kind of model you end up with (generative decoder vs bidirectional encoder).

## 1.2 Causal Language Modeling (Causal LM)

**The task:** given all tokens before position `i`, predict token `i`. The model only ever sees left-context (past tokens), never future ones — this is exactly the causal masking behavior you covered in the attention/decoder architecture section.

$$P(x) = \prod_{i=1}^{n} P(x_i \mid x_1, x_2, ..., x_{i-1})$$

This decomposes the probability of an entire sequence into a product of next-token predictions, each conditioned only on what came before.

**Loss function:** Cross-entropy loss between the predicted probability distribution over the vocabulary and the actual next token, averaged over all positions in the sequence:

$$\mathcal{L} = -\sum_{i=1}^{n} \log P(x_i \mid x_{<i})$$

**Used by:** GPT family, LLaMA, Mistral — essentially all modern general-purpose LLMs designed for open-ended text generation.

**Why this objective is so powerful:** predicting the next token, at massive scale across diverse internet text, forces the model to implicitly learn grammar, facts, reasoning patterns, and world knowledge — not because it was explicitly taught these things, but because accurately predicting the next word in billions of diverse sentences requires understanding all of them.

## 1.3 Masked Language Modeling (Masked LM)

**The task:** randomly mask out (hide) a percentage of tokens in the input (typically 15%), and train the model to predict the original masked tokens using **bidirectional context** — meaning it can see both the tokens before AND after each masked position.

**Example:** `"The [MASK] sat on the mat"` → model must predict `"cat"` using context from both sides (`"The ___ sat"` and `"___ sat on the mat"`).

**BERT's specific masking strategy (the 80/10/10 rule):** of the 15% of tokens selected for masking:
- 80% are replaced with an actual `[MASK]` token
- 10% are replaced with a random other token
- 10% are left unchanged

**Why the 80/10/10 split exists:** if you *always* replaced masked tokens with the literal `[MASK]` token, the model would only ever learn to make predictions in the presence of `[MASK]` — but at fine-tuning/inference time, there is no `[MASK]` token in real text. Mixing in random replacements and unchanged tokens forces the model to build genuinely robust contextual representations for *every* token, not just ones flagged as "hidden," since it can never be sure which tokens might secretly be "wrong" and need correcting.

**Used by:** BERT and its variants (RoBERTa, ALBERT, DistilBERT).

## 1.4 Causal LM vs Masked LM — Why the Difference Matters

| | Causal LM | Masked LM |
|---|---|---|
| Context direction | Left-to-right only | Bidirectional |
| Best suited for | Text generation | Understanding/classification tasks |
| Can generate new text? | Yes, naturally | No, not naturally |
| Example models | GPT, LLaMA | BERT |

**Why bidirectional context helps understanding but hurts generation:** for tasks like sentiment classification or named entity recognition, seeing the *entire* sentence at once (both directions) gives richer context for understanding meaning. But for generation, the model must work with only what's been generated so far — it fundamentally cannot see "future" tokens, because those future tokens don't exist yet at generation time. This is precisely why GPT-style (causal) models are the standard for the general-purpose "LLM" use case, even though BERT-style models can outperform them on some pure classification benchmarks.

### 🔧 Activity 1: Pretraining Objectives
1. Take a sentence of your choice and manually create a Masked LM training example following BERT's 80/10/10 rule — mask ~15% of tokens, showing which get `[MASK]`, which get randomly replaced, and which stay unchanged.
2. Write out (on paper) the causal LM factorization for a 5-word sentence, showing each conditional probability term explicitly.
3. Load `bert-base-uncased` and `gpt2` from HuggingFace `transformers`. Try to get each model to complete a sentence via next-word-style prediction — notice how BERT's architecture makes this awkward (it's not designed for it) compared to GPT.
4. Explain in your own words: why can't you easily use a Masked LM (like BERT) to generate long, coherent, open-ended text the way GPT can?

---

# 2. FINE-TUNING APPROACHES

## 2.1 Full Fine-Tuning

**The approach:** take a pretrained model and continue training **all** of its parameters on a smaller, task-specific labeled dataset, typically with a much smaller learning rate than pretraining used.

**Why it works:** the pretrained weights already encode rich general language understanding; full fine-tuning nudges every parameter slightly to specialize that general knowledge toward your specific task/domain.

**The problem at LLM scale:** a 7B parameter model requires updating and storing gradients + optimizer states (Adam stores 2 extra values per parameter) for all 7 billion parameters. This demands enormous GPU memory (often 4-6x the raw model size when you include gradients and optimizer states), making full fine-tuning of large models expensive and often impractical on consumer/limited hardware — and you need a *full copy* of the model's weights saved for every fine-tuned variant you create.

## 2.2 LoRA (Low-Rank Adaptation)

**Core insight:** the *change* in weights needed to adapt a pretrained model to a new task tends to have a low "intrinsic rank" — meaning you don't need to update the full, massive weight matrix; a much smaller, low-rank approximation of the update captures most of the useful adaptation.

**How it works mathematically:** instead of updating the full weight matrix $W$ (dimensions $d \times k$) directly, LoRA **freezes** the original $W$ entirely and introduces two small trainable matrices $A$ (dimensions $d \times r$) and $B$ (dimensions $r \times k$), where $r$ (the "rank") is a small number like 4, 8, or 16 — much smaller than $d$ or $k$.

$$W' = W + \Delta W = W + BA$$

During the forward pass, the output becomes:

$$h = Wx + BAx$$

**Why this saves so much memory:** you only need to train and store $A$ and $B$, whose combined parameter count ($d \times r + r \times k$) is dramatically smaller than the full $d \times k$ matrix when $r$ is small. For example, if $d = k = 4096$ and $r = 8$: full matrix = 16.7M parameters, LoRA matrices = $4096 \times 8 \times 2 = 65,536$ parameters — roughly **250x fewer trainable parameters** for that layer.

**Additional benefits:**
- The original pretrained weights $W$ never change, so you can keep a single base model and swap in different lightweight LoRA "adapters" for different tasks
- At inference time, you can merge $BA$ back into $W$ with zero added latency (since $W + BA$ is precomputed once), so LoRA doesn't slow down inference at all once merged
- Since gradients are only needed for the small $A$/$B$ matrices, memory usage during training drops dramatically compared to full fine-tuning

## 2.3 QLoRA (Quantized LoRA)

**Core insight:** combine LoRA's parameter-efficiency with **quantization** of the frozen base model, pushing memory savings even further.

**How it works:** the frozen pretrained weights $W$ are stored in a heavily compressed 4-bit format (using a technique called NF4 — 4-bit NormalFloat, designed specifically to represent normally-distributed weight values efficiently) instead of the usual 16-bit or 32-bit floating point. The small trainable LoRA matrices ($A$, $B$) are still kept in higher precision (typically 16-bit) since they need to be trained via gradient descent.

**Key additional techniques QLoRA introduces:**
- **Double quantization**: quantizing the quantization constants themselves, squeezing out a bit more memory savings
- **Paged optimizers**: using NVIDIA unified memory to handle occasional GPU memory spikes gracefully (spilling to CPU RAM when needed) rather than crashing

**Why this matters practically:** QLoRA made it possible to fine-tune a 65B parameter model on a *single* 48GB GPU — something that would have required multiple high-end GPUs with full fine-tuning or even standard LoRA (since the frozen base model itself, at 16-bit precision, is still huge). This dramatically democratized fine-tuning of large models.

## 2.4 PEFT Methods (Parameter-Efficient Fine-Tuning) — The Broader Family

LoRA/QLoRA are the most popular, but they're part of a broader category called PEFT — techniques that fine-tune only a small subset or small addition of parameters rather than the whole model:

- **Prompt Tuning**: freeze the entire model, and instead learn a small set of "soft prompt" embedding vectors prepended to the input — the model itself never changes, only these prepended vectors are trained
- **Prefix Tuning**: similar idea but prepends learnable vectors to the *keys and values* at every transformer layer (not just the input embedding), giving more expressive power than prompt tuning
- **Adapter Layers**: insert small trainable bottleneck feed-forward layers between existing transformer layers, freezing everything else — an earlier PEFT approach that predates LoRA
- **(IA)³**: learns per-layer rescaling vectors that multiply activations, an even more parameter-efficient approach than LoRA in some cases

**Common thread across all PEFT methods:** freeze the vast majority of the pretrained model, train a small number of additional/modified parameters, and get most of the benefit of full fine-tuning at a small fraction of the memory/compute cost.

### 🔧 Activity 2: Fine-Tuning Approaches
1. Calculate the number of trainable parameters for a LoRA adapter with rank `r=16` applied to a weight matrix of shape 4096×4096. Compare this to the full matrix's parameter count and compute the percentage reduction.
2. Using HuggingFace's `peft` library, load a small model (e.g. `gpt2` or a small LLaMA variant) and apply a LoRA configuration. Print `model.print_trainable_parameters()` to see the actual trainable vs frozen parameter counts.
3. Read about NF4 quantization (the format QLoRA uses) — explain in your own words why a format designed for normally-distributed values is well-suited to neural network weights specifically (hint: think about how weights are typically distributed after training).
4. Design decision exercise: you have a single RTX 4090 (24GB VRAM) and want to fine-tune a 13B parameter model. Would you choose full fine-tuning, LoRA, or QLoRA? Justify your answer with rough memory estimates.

---

# 3. INSTRUCTION TUNING

## 3.1 The Problem It Solves

A raw pretrained LLM (just trained with the causal LM objective on internet text) is very good at *continuing* text in a statistically plausible way, but it has no inherent notion of "follow the user's instruction and produce a helpful response." Ask a raw pretrained model "What's the capital of France?" and it might just continue with more similar-looking trivia questions, rather than actually answering yours — because that's what would statistically follow such text in its training data (e.g., a quiz webpage).

**Instruction tuning** fixes this by fine-tuning the pretrained model on a dataset of **(instruction, response)** pairs, teaching it the *behavior* of directly following instructions and producing helpful, on-topic completions.

## 3.2 How It Works

**Dataset format:** typically structured as instruction-response pairs, sometimes with an additional "input" field for tasks that need extra context:

```
Instruction: "Summarize the following text in one sentence."
Input: "[a long paragraph of text]"
Response: "[a one-sentence summary]"
```

**Training procedure:** this is still just supervised fine-tuning using the causal LM objective — the model is trained to predict the response tokens given the instruction (and input) as context, using standard cross-entropy loss. What's different from further-pretraining is purely the *structure and diversity* of the data (instruction-formatted, covering many different task types) rather than the loss function itself.

**Why diversity of instruction types matters so much:** the goal isn't to teach the model any single task, but to teach it the *general skill* of "read an instruction, understand what's being asked, and respond appropriately" — this generalizes to instructions the model has never seen before, provided the training set covers a wide enough variety of instruction types, phrasings, and domains.

## 3.3 Notable Instruction-Tuning Datasets/Approaches

- **FLAN** (Google): fine-tuned on a large collection of tasks reformatted as instructions, showed instruction tuning significantly improves zero-shot performance on unseen tasks
- **Self-Instruct**: uses an LLM itself to generate new instruction-response pairs from a small seed set, bootstrapping a large instruction dataset with minimal human labeling
- **Alpaca**: fine-tuned LLaMA using instructions generated by GPT-3 (via Self-Instruct-style bootstrapping), demonstrating that a relatively small instruction dataset (~52k examples) can meaningfully improve instruction-following behavior

### 🔧 Activity 3: Instruction Tuning
1. Write 5 diverse instruction-response pairs covering different task types (summarization, question-answering, code generation, creative writing, classification) — this is exactly the kind of data instruction-tuning datasets are built from.
2. Compare the output of a base (non-instruction-tuned) model like `gpt2` vs an instruction-tuned model like `google/flan-t5-base` on the same instruction prompt (e.g. "Explain photosynthesis in simple terms"). Note the qualitative difference in how directly each one addresses the instruction.
3. Read about the Self-Instruct paper's bootstrapping process and diagram the pipeline: how does a small seed set of instructions turn into a large training dataset using an LLM?
4. Explain in your own words why instruction tuning is considered "teaching a behavior" rather than "teaching new knowledge."

---

# 4. RLHF & DPO

## 4.1 Why Instruction Tuning Alone Isn't Enough

Instruction tuning teaches a model to follow instructions and produce *reasonable* responses, but "reasonable" isn't the same as "what humans actually prefer." There can be many valid ways to respond to an instruction, and instruction tuning (via straightforward supervised learning on fixed responses) doesn't have a mechanism to learn nuanced human preferences like "this response is more helpful, honest, and harmless than that one," especially when both responses are grammatically fine and on-topic.

## 4.2 RLHF (Reinforcement Learning from Human Feedback)

RLHF is a **three-stage pipeline**:

**Stage 1 — Supervised Fine-Tuning (SFT):** this is instruction tuning, as covered above — produces a model that follows instructions reasonably well. This becomes the starting point for the next stages.

**Stage 2 — Reward Model Training:** 
- Generate multiple responses (e.g., 4-9 candidates) to the same prompt using the SFT model
- Have human annotators **rank** these responses from best to worst (ranking is used rather than absolute scoring because it's much easier and more consistent for humans to say "A is better than B" than to assign a precise numerical quality score)
- Train a separate **reward model** (usually initialized from the SFT model, with the output layer replaced with a single scalar output) to predict a scalar "quality score" that's consistent with the human rankings

**Reward model loss function** (based on the Bradley-Terry preference model):

$$\mathcal{L}(\theta) = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

Where $y_w$ is the "winning" (preferred) response, $y_l$ is the "losing" response, and $r_\theta$ is the reward model's scalar output — this loss pushes the reward model to assign a higher score to the preferred response than the rejected one.

**Stage 3 — Reinforcement Learning (typically PPO — Proximal Policy Optimization):**
- Use the trained reward model as the "reward signal" in an RL loop
- The SFT model (now called the "policy") generates responses to prompts, the reward model scores them, and the policy's weights are updated via PPO to maximize expected reward
- **Critical addition — KL divergence penalty:** the RL objective also includes a penalty term that keeps the policy's outputs close to the original SFT model's distribution, preventing the policy from drifting too far and "reward hacking" (finding weird, unnatural outputs that fool the reward model into giving high scores without actually being good responses)

$$\text{Objective} = \mathbb{E}[r_\theta(x,y)] - \beta \cdot D_{KL}(\pi_{RL} || \pi_{SFT})$$

## 4.3 Why RLHF Is Complicated (Practical Challenges)

- Requires training and maintaining **four** separate models simultaneously during the RL stage: the policy model (being trained), a frozen reference copy of the SFT model (for the KL penalty), the reward model, and often a value/critic model for PPO — this is compute and memory intensive
- PPO itself is notoriously finicky to tune (sensitive to hyperparameters, prone to instability)
- The reward model can be "gamed" (reward hacking) if not carefully regularized, since it's an imperfect proxy for true human preference

## 4.4 DPO (Direct Preference Optimization)

**The key insight:** DPO reformulates the RLHF objective mathematically so that you can directly optimize the policy on preference data **without needing a separate reward model or the RL loop at all**. It shows that the optimal policy under the RLHF objective has a closed-form relationship to the reward function, allowing you to substitute this relationship back into the preference loss and optimize the policy *directly* using simple supervised-learning-style gradient descent on the preference pairs.

**DPO loss function:**

$$\mathcal{L}_{DPO}(\theta) = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

Where $\pi_\theta$ is the model being trained, $\pi_{ref}$ is the frozen reference (SFT) model, and $y_w$/$y_l$ are the winning/losing responses from the same preference data used for RLHF's reward model.

**Why this is a big deal practically:**
- No separate reward model needs to be trained
- No RL loop (no PPO, no value/critic model, no instability from RL training dynamics)
- Only two models needed in memory (the policy being trained + the frozen reference model), vs RLHF's four
- Empirically achieves comparable or sometimes better alignment results than full RLHF, with dramatically simpler and more stable training

**Why DPO works despite looking so different from RLHF:** it's mathematically *derived* from the same underlying RLHF objective (maximize reward subject to a KL constraint from the reference policy) — DPO isn't a different goal, it's an equivalent but much more directly optimizable reformulation of the exact same goal.

### 🔧 Activity 4: RLHF & DPO
1. Diagram the full RLHF pipeline (SFT → Reward Model → PPO), labeling which model(s) are frozen vs trainable at each stage.
2. Given two candidate responses to a prompt, write out by hand what a human preference-ranking annotation might look like, and manually compute the Bradley-Terry loss value for a toy example (pick simple reward model output values for the winning/losing response).
3. Read the DPO paper's abstract and introduction — write 3-4 sentences in your own words explaining why DPO can skip the RL loop entirely while still optimizing "the same thing" RLHF optimizes.
4. Compare: list the number of models that must be held in memory simultaneously for RLHF vs DPO, and explain the practical infrastructure benefit this gives DPO.

---

# 5. PROMPT ENGINEERING, FEW-SHOT & ZERO-SHOT LEARNING

## 5.1 Zero-Shot Learning

**Definition:** asking a model to perform a task it was never explicitly trained on, using only a natural language instruction/description, with **no examples** provided in the prompt.

```
Prompt: "Classify the sentiment of this review as positive or negative: 
'The food was cold and the service was slow.'"
```

The model relies entirely on knowledge and patterns learned during pretraining/instruction-tuning to infer what's being asked and produce a reasonable answer, without ever having seen a labeled example of this specific task format in the prompt itself.

**Why this works at all (and didn't work well with older, smaller models):** large-scale instruction-tuned LLMs have seen an enormous diversity of task formats during training, so they've learned a generalized ability to map natural-language task descriptions to appropriate behaviors — this emergent capability scales with model size, which is why zero-shot performance was poor on small/older models but became remarkably strong on large modern LLMs.

## 5.2 Few-Shot Learning (In-Context Learning)

**Definition:** providing a small number of example input-output pairs directly within the prompt before asking the actual question, without updating any model weights at all — the model infers the task pattern purely from the examples shown "in context."

```
Prompt:
"Review: 'Amazing experience, will come back!' → Positive
Review: 'Never eating here again.' → Negative
Review: 'The food was cold and the service was slow.' → ?"
```

**Why this is remarkable:** this is fundamentally different from traditional machine learning "training" — no gradients are computed, no weights change. The model is using its pretrained knowledge to recognize the *pattern* being demonstrated (input → output mapping) and apply that same pattern to the new input, entirely through the forward pass. This capability is called **in-context learning**, and it's one of the most studied emergent behaviors of large language models.

**Practical considerations:**
- More examples generally help up to a point, but consume more of the context window (and thus more compute/cost per request)
- The *order* and *selection* of examples can meaningfully affect output quality — this is an active area of "prompt engineering" research
- Few-shot examples should ideally be diverse and representative of the range of cases the model might see at inference time

## 5.3 Prompt Engineering — Core Techniques

**Chain-of-Thought (CoT) Prompting:** explicitly instructing the model to "think step by step" or showing few-shot examples that include intermediate reasoning steps, not just final answers. This dramatically improves performance on tasks requiring multi-step reasoning (math word problems, logic puzzles) because it gives the model "space" to work through intermediate steps token-by-token, rather than forcing it to jump straight to a final answer.

```
Without CoT: "What is 17 × 24? Answer:" 
With CoT: "What is 17 × 24? Let's think step by step: 17 × 24 = 17 × 20 + 17 × 4 = 340 + 68 = 408"
```

**Role/Persona Prompting:** framing the prompt with a specific role ("You are an expert Python developer...") to bias the model's response style and content toward that domain/expertise.

**Structured Output Prompting:** explicitly specifying the desired output format (JSON schema, bullet points, specific fields) to make outputs more reliably parseable in downstream applications.

**Self-Consistency:** generating multiple independent chain-of-thought responses to the same prompt (using sampling with temperature > 0) and taking a majority vote over the final answers — trades additional inference compute for improved accuracy on reasoning tasks.

**Why prompt engineering matters practically (ties to your inference optimization interest):** a well-engineered prompt can sometimes achieve results comparable to fine-tuning, at zero training cost — but it does consume more tokens per request (few-shot examples, CoT reasoning traces), which directly impacts inference latency and cost, creating a real engineering tradeoff between prompt complexity and serving efficiency.

### 🔧 Activity 5: Prompt Engineering & In-Context Learning
1. Write a zero-shot prompt and a 3-shot prompt for the same classification task (e.g., classifying customer support tickets by urgency). Test both against a model you have access to and compare output quality/consistency.
2. Design a Chain-of-Thought prompt for a multi-step reasoning problem (e.g., a word problem involving several calculation steps). Compare the model's answer with and without the "let's think step by step" instruction.
3. Implement a simple self-consistency setup: generate 5 responses to the same reasoning prompt with temperature=0.7, extract the final answer from each, and take a majority vote. Does this improve correctness over a single greedy-decoded response?
4. Explain in your own words: why does few-shot in-context learning work without any weight updates — what is the model actually "doing" when it picks up a pattern from examples in the prompt?

---

# 20 DEEP INTERVIEW QUESTIONS — TRAINING & FINE-TUNING

**1. Explain the causal LM objective mathematically, and why it naturally supports parallelized training but sequential inference.**
Causal LM factorizes the probability of a sequence as a product of next-token conditional probabilities, each conditioned only on prior tokens: P(x) = ∏ P(xᵢ | x_{<i}). During training, since the full ground-truth sequence is available upfront, the loss for every position can be computed simultaneously in one forward pass (with causal masking preventing information leakage from future positions) — this is teacher forcing, and it's what allows training to be parallelized across the sequence dimension. During inference, future tokens don't exist yet since they haven't been generated, so each token must be produced one at a time, with each new token depending on all previously generated ones, making generation inherently sequential.

**2. Walk through BERT's masking strategy and explain why it uses the 80/10/10 split instead of always using [MASK].**
Of the 15% of tokens selected for masking, 80% are replaced with the literal [MASK] token, 10% are replaced with a random other token, and 10% are left unchanged. If tokens were always replaced with [MASK], the model would only learn to make predictions in the presence of that specific token, but at fine-tuning/inference time there's no [MASK] token in real text, creating a train-test mismatch. The 80/10/10 split forces the model to build robust contextual representations for every token, since it can never be certain which tokens are "trustworthy" as-is versus potentially wrong/masked, improving representation quality when later used for downstream tasks.

**3. Why can't you use a Masked LM like BERT to generate long, open-ended text the way GPT can?**
BERT was trained to fill in masked positions using bidirectional context — it expects to see, and depends on having access to, tokens both before and after the position it's predicting. Open-ended generation requires producing tokens sequentially where "future" tokens genuinely don't exist yet, so BERT's bidirectional training objective doesn't naturally support autoregressive generation; it has no learned mechanism for producing a coherent continuation one token at a time without already knowing what comes after.

**4. Explain the core mathematical idea behind LoRA, including the low-rank decomposition and why it saves memory.**
LoRA freezes the pretrained weight matrix W and represents the weight *update* needed for adaptation as the product of two much smaller matrices, A (d×r) and B (r×k), where the rank r is small (e.g., 8 or 16) relative to the original dimensions. The forward pass becomes h = Wx + BAx. Because only A and B are trained (and their combined parameter count is far smaller than the full d×k matrix when r is small), gradient computation and optimizer state storage — which normally scale with the number of trainable parameters — shrink dramatically, cutting training memory requirements by orders of magnitude while W itself remains untouched.

**5. What is the "intrinsic rank" hypothesis that motivates LoRA?**
The hypothesis is that the change in weights needed to adapt a large pretrained model to a new, narrower task doesn't require the full expressiveness of a dense d×k weight update — the useful adaptation actually lies in a much lower-dimensional subspace. This means a low-rank approximation (BA, with small r) can capture nearly all the useful adaptation signal that a full-rank update would, which is why LoRA's small matrices are sufficient rather than a compromise.

**6. How does QLoRA extend LoRA, and what specific quantization technique does it use for the frozen base model?**
QLoRA stores the frozen pretrained base model weights in 4-bit precision using NF4 (4-bit NormalFloat), a quantization format specifically designed to represent normally-distributed values (which neural network weights typically approximate) efficiently, while keeping the small trainable LoRA matrices in higher precision (e.g., 16-bit) since they need to be updated via gradient descent. It also introduces double quantization (quantizing the quantization constants themselves for additional memory savings) and paged optimizers (using unified memory to handle GPU memory spikes without crashing), together enabling fine-tuning of models like 65B parameters on a single consumer/prosumer-class GPU.

**7. Compare full fine-tuning, LoRA, and QLoRA in terms of memory requirements and when you'd choose each.**
Full fine-tuning updates all parameters, requiring memory for the full weights plus gradients plus optimizer states for every parameter (often 4-6x the base model size) — chosen when you have ample compute/memory and need maximum adaptation flexibility, or for smaller models where this cost is manageable. LoRA freezes the base model and trains only small low-rank adapter matrices, cutting trainable-parameter memory dramatically while keeping the base model at full precision — chosen as a strong default for most fine-tuning needs on moderately-sized hardware. QLoRA additionally quantizes the frozen base model itself to 4-bit, further shrinking the memory footprint of the (much larger, in absolute terms) frozen weights — chosen specifically when GPU memory is the binding constraint, e.g., fine-tuning a very large model on a single consumer GPU.

**8. What is instruction tuning, and why is it described as "teaching a behavior" rather than "teaching new knowledge"?**
Instruction tuning fine-tunes a pretrained model on (instruction, response) pairs so it learns to directly follow instructions and produce helpful, on-topic completions rather than just statistically plausible continuations. It's described as teaching a behavior because the model already possesses the underlying knowledge from pretraining (facts, language patterns, reasoning ability) — instruction tuning doesn't add new knowledge, it teaches the model *how to use* that existing knowledge in a helpful, instruction-following format, generalizing this behavior to instruction types never explicitly seen during fine-tuning.

**9. Explain the Self-Instruct approach and why it's useful for building instruction-tuning datasets.**
Self-Instruct uses an existing LLM to generate new instruction-response pairs automatically, bootstrapping from a small seed set of human-written instructions — the LLM generates new instructions in a similar style, then generates corresponding responses, with filtering to ensure quality/diversity. This dramatically reduces the human labor required to build large, diverse instruction-tuning datasets (like the ~52k examples behind Alpaca), since manually writing tens of thousands of high-quality instruction-response pairs would be prohibitively expensive and slow.

**10. Why is instruction tuning alone insufficient for producing a well-aligned, helpful assistant, motivating the need for RLHF/DPO?**
Instruction tuning trains on a fixed set of "correct" responses via standard supervised learning, but many prompts have multiple plausible responses that differ in subtle but important ways (helpfulness, tone, safety, honesty) that a single fixed target response can't fully capture. Supervised fine-tuning has no mechanism to learn "response A is better than response B" when both are reasonable — it can only learn to imitate whatever specific responses were in the training data, which doesn't teach nuanced preference-based quality judgments the way comparing and ranking multiple candidate responses does.

**11. Walk through all three stages of the RLHF pipeline in order.**
Stage 1 (SFT): instruction-tune the pretrained model on instruction-response pairs to get a model that follows instructions reasonably well. Stage 2 (Reward Model): generate multiple candidate responses per prompt using the SFT model, have humans rank them by preference, and train a separate reward model (usually initialized from the SFT model with a scalar output head) to predict scores consistent with those rankings, using a Bradley-Terry-style pairwise preference loss. Stage 3 (RL/PPO): use the reward model as the reward signal in a reinforcement learning loop (typically PPO), updating the SFT model's weights (now called the policy) to maximize expected reward, while a KL-divergence penalty against the original SFT model prevents the policy from drifting too far and reward-hacking.

**12. Why does RLHF use response *ranking* rather than asking humans to assign absolute quality scores?**
Humans are much more consistent and reliable at comparative judgments ("A is better than B") than at assigning precise absolute numerical scores to individual outputs — absolute scoring is subjective and inconsistent across annotators and even across time for the same annotator, whereas relative preference judgments are more stable and easier to elicit reliably, producing higher-quality training signal for the reward model.

**13. What is reward hacking in the context of RLHF, and what mechanism helps prevent it?**
Reward hacking occurs when the policy model, during RL optimization, discovers outputs that achieve high scores from the reward model without actually being genuinely good responses — since the reward model is only an imperfect learned proxy for true human preference, the policy can exploit its blind spots or quirks. The KL-divergence penalty term in the RL objective, which penalizes the policy for drifting too far from the original SFT model's output distribution, helps constrain this by keeping the policy's behavior anchored close to a model already known to produce reasonable, human-like text.

**14. Explain the key mathematical reformulation that DPO uses to avoid needing a separate reward model or RL loop.**
DPO shows that the optimal policy under the standard RLHF objective (maximize reward subject to a KL constraint from a reference policy) has a closed-form analytical relationship to the reward function. By substituting this relationship into the pairwise preference loss, the reward function can be eliminated algebraically, leaving a loss expressed purely in terms of the policy's own log-probabilities (relative to the frozen reference model) for the preferred versus rejected responses — allowing direct supervised-style optimization on the preference data without ever training an explicit reward model or running an RL loop.

**15. Why is DPO considered practically simpler and more stable to train than RLHF?**
DPO only requires two models in memory (the policy being trained and the frozen reference model), compared to RLHF's four (policy, frozen reference, reward model, and often a separate value/critic model for PPO). It also avoids PPO entirely, which is known to be sensitive to hyperparameters and prone to training instability — DPO's loss is optimized via straightforward gradient descent similar to standard supervised fine-tuning, making the training loop far simpler to implement and more stable to run in practice.

**16. Is DPO optimizing a fundamentally different objective than RLHF, or the same one? Explain.**
DPO optimizes mathematically the *same* underlying objective as RLHF — maximizing expected reward subject to a KL-divergence constraint keeping the policy close to the reference model. DPO isn't a different goal or an approximation; it's an algebraically equivalent reformulation that happens to be directly optimizable via supervised-style loss on preference pairs, rather than requiring the indirect route of training a reward model first and then running RL to optimize against it.

**17. What is zero-shot learning, and why does it work well on large modern LLMs but poorly on smaller/older models?**
Zero-shot learning asks a model to perform a task using only a natural-language instruction, with no examples provided. It relies on the model having learned, during large-scale pretraining and instruction-tuning, a generalized ability to map natural-language task descriptions to appropriate behaviors across an enormous diversity of task types seen during training. This generalized instruction-following capability is considered an emergent property that scales with model size and training data diversity — smaller/older models simply haven't seen enough diverse task framings during training to generalize this way, so they perform much better when given explicit examples (few-shot) rather than relying on instruction understanding alone.

**18. Explain in-context learning (few-shot prompting) and why it's notable that it requires no weight updates.**
In-context learning provides a handful of example input-output pairs directly in the prompt, and the model infers the underlying task pattern purely through its forward pass, without any gradient computation or parameter updates — the "learning" happens entirely within the model's activations as it processes the context, not through any change to the model's stored weights. This is notable because it behaves functionally like learning a new task, but mechanistically it's just inference — the same frozen model, given different context, produces task-appropriate outputs, which is a qualitatively different phenomenon from traditional ML training.

**19. What is Chain-of-Thought prompting, and why does it improve performance on multi-step reasoning tasks?**
Chain-of-Thought prompting explicitly instructs the model to work through intermediate reasoning steps before producing a final answer, either via an instruction like "think step by step" or by showing few-shot examples that include worked-out reasoning traces, not just final answers. It improves performance because it gives the model "computational space," spread across multiple generated tokens, to work through sub-steps of a problem sequentially — since each token is generated conditioned on all prior tokens (including the reasoning steps just generated), the model effectively gets to use its own intermediate outputs as additional context/scratch space, which single-token "jump straight to the answer" prompting doesn't allow.

**20. In a real production system, you need to decide between fine-tuning a model versus using few-shot prompting for a new task. What tradeoffs would you weigh, and how does this connect to inference cost?**
Fine-tuning (full, LoRA, or QLoRA) requires upfront training cost and data collection but results in a model that performs the task efficiently at inference time — no extra tokens needed per request beyond the actual input, keeping per-request latency and cost low, and behavior is baked directly into the weights rather than depending on prompt engineering. Few-shot prompting requires no training at all and can be deployed instantly, but every request must include the few-shot examples in the context window, increasing token count, per-request compute, and API/serving cost, and results can be more sensitive to example selection/ordering. For a high-volume production system where the same task is performed repeatedly at scale, fine-tuning often wins on long-run inference efficiency despite the upfront cost; for rapidly prototyping, low-volume, or frequently-changing tasks, few-shot prompting's flexibility and zero training cost usually wins — this exact tradeoff is central to inference cost optimization work.

---

# INDEX / REVISION SHEET

## 1. Pretraining Objectives
- **Causal LM**: predict next token from left-context only; used by GPT/LLaMA; enables generation
- **Masked LM**: predict masked tokens using bidirectional context; used by BERT; 80/10/10 masking rule prevents train-test mismatch
- Causal LM → best for generation; Masked LM → best for understanding/classification

## 2. Fine-Tuning Approaches
- **Full fine-tuning**: update all parameters; most memory-intensive; most flexible
- **LoRA**: freeze W, learn low-rank update BA; ~100-250x fewer trainable params typical; merges into W at inference with zero added latency
- **QLoRA**: LoRA + 4-bit NF4 quantization of frozen base model + double quantization + paged optimizers; enables fine-tuning 65B models on a single GPU
- **PEFT family**: Prompt Tuning, Prefix Tuning, Adapter Layers, (IA)³ — all freeze most of the model, train a small addition

## 3. Instruction Tuning
- Fine-tunes on (instruction, response) pairs to teach instruction-following *behavior*, not new knowledge
- Still uses causal LM cross-entropy loss — what changes is data structure/diversity, not the objective
- Self-Instruct: bootstraps large instruction datasets from a small seed set using an LLM itself
- Notable datasets/models: FLAN, Alpaca

## 4. RLHF & DPO
- **RLHF 3 stages**: SFT (=instruction tuning) → Reward Model (trained on human rankings, Bradley-Terry loss) → PPO (RL, with KL penalty against SFT model to prevent reward hacking)
- RLHF needs 4 models in memory: policy, frozen reference, reward model, value/critic
- **DPO**: mathematically equivalent reformulation that eliminates the reward model + RL loop, optimizing preference data directly via a supervised-style loss
- DPO needs only 2 models: policy + frozen reference — simpler, more stable, same underlying objective as RLHF

## 5. Prompt Engineering, Few-Shot & Zero-Shot
- **Zero-shot**: task description only, no examples; relies on emergent generalized instruction-following at scale
- **Few-shot / in-context learning**: examples in the prompt, no weight updates; pattern inferred purely via forward pass
- **Chain-of-Thought**: explicit step-by-step reasoning improves multi-step task performance by giving the model "space" to reason token-by-token
- **Self-Consistency**: sample multiple CoT responses, majority-vote the final answer
- Prompt complexity (few-shot examples, CoT traces) directly trades off against inference cost/latency — a real engineering consideration