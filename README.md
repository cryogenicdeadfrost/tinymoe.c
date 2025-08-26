# moe.c

> A Mixture-of-Experts Transformer you can **read** in one sitting and **run** on your laptop.  
> Training, inference, saving, loading—**all in one C file**. No frameworks, no magic.

Inspired by (and eternally grateful to) Andrej Karpathy’s minimalist deep-learning projects like [`llama2.c`](https://github.com/karpathy/llama2.c) and `llm.c`, this repo reconstructs a modern sparse Transformer from scratch so you can see how it actually ticks.

---

## Why?

State-of-the-art LLMs increasingly use **Mixture-of-Experts (MoE)** layers to scale parameters without scaling compute. But how does that *really* work?

- How does a **gating network** decide which experts should handle a token?
- How do you route only to the **top-k** experts and still backprop through those choices?
- How do you **balance load** so the gate doesn’t fall in love with one expert and ghost the rest?

This project answers those by building the whole thing—forward pass, backward pass, optimizer, data loader, checkpointing—**in a single, heavily commented C file**.

---

## Design Philosophy

- **One File to Rule Them All.** Everything lives in `moe.c`. If you can read C, you can understand the stack.
- **No Magic.** Only the C standard library. No BLAS, no CUDA, no PyTorch. It’s just you and the beautiful horror of backprop.
- **Clarity Over Cleverness.** The code is a narrative. Comments explain the *why*, not just the *what*.

---

## What’s Inside

- A GPT-2-style Transformer with a **Mixture-of-Experts FFN** in each block.
- **Top-k routing** (e.g., k=2) via a tiny **gating network**.
- **Auxiliary load-balancing loss** so experts are actually used.
- **Adam optimizer** implemented from scratch.
- **From-scratch training** on a tiny Shakespeare sample (bundled).
- **Checkpointing** to `tinymoe.bin` and **reload + inference**.

---

## Architecture (High Level)

Each Transformer block swaps the classic dense FFN for an MoE layer:

```

Token h ──► \[Self-Attn] ──► h'
│
└► \[Gate] ─► scores over E experts ─► take top-k indices & weights
│
├──► Expert #i: FFN(h') ─┐
├──► Expert #j: FFN(h') ─┤  weighted sum ► MoE output
└──► ... (only top-k) ───┘

````

- **Gatekeeper (Gating Network):** a tiny linear layer producing expert scores per token.  
- **Specialists (Experts):** independent FFNs (same shape), each free to specialize (punctuation, numbers, code, who knows).  
- **Council (Top-K):** we route each token to the top-k experts; outputs are blended by the gate’s normalized weights.  
- **Load-Balancing:** an auxiliary loss nudges the gate to *use* all experts, not just a favorite.  

---

## Quick Start

### 1) Build

You’ll want optimizations; otherwise training finishes around the heat death of the universe.

```bash
gcc -O2 -o moe moe.c -lm
# optional: squeeze a bit more
# gcc -O3 -march=native -ffast-math -o moe moe.c -lm
````

### 2) Run

```bash
./moe
```

It will start training on a tiny Shakespeare corpus and print losses every \~50 steps:

```
Step    0 | Loss: 4.1234 (Primary: 4.0123, Aux: 1.1111) | Best: 4.1234
Step   50 | Loss: 3.5678 (Primary: 3.4567, Aux: 1.1111) | Best: 3.5678
...
Step  950 | Loss: 1.8765 (Primary: 1.7654, Aux: 1.1111) | Best: 1.8765
```

* **Primary** is the language-model loss (cross-entropy).
* **Aux** is the load-balancing term—expected to hover roughly stable as routing improves.

At the end, weights are saved to `tinymoe.bin`, and the program demos text generation from a few prompts.

---

## Usage

### Training (default)

Just run `./moe`. Defaults (dataset path, model dims, experts, top-k, steps, etc.) live at the top of `moe.c`. Tweak them and rebuild.

Common knobs you’ll find near the top of the file:

* `N_LAYERS`, `N_HEADS`, `D_MODEL`, `D_FF`
* `N_EXPERTS`, `TOP_K`
* `SEQ_LEN`, `BATCH_SIZE`
* `STEPS`, `LR`, `WEIGHT_DECAY`
* `AUX_LOSS_WEIGHT`
* dataset path / random seed

### Inference-only

You can flip the `main()` flow to **load** `tinymoe.bin` and skip training, or run with a small compile-time switch (documented in the header comments):

```c
// e.g., set RUN_INFERENCE_ONLY 1 at the top, or call a load_and_generate() entry point
```

Then:

```bash
./moe
# prints generations for built-in prompts
```

### Checkpoint format

`tinymoe.bin` is a raw dump of parameters in a fixed order (documented in comments). Loading just maps the bytes back into the parameter buffers—no frameworks harmed.

---

## How Routing Works (in practice)

* Gate computes scores `g` for each token → softmax to probabilities `p`.
* Take **top-k** indices `I = argtopk(p)`. Only those experts run.
* Each selected expert processes the token independently → outputs `y_i`.
* Final output is `Σ_{i∈I} p_i * y_i` (using the *normalized* probabilities over the selected experts).
* Backprop flows through the gate (via the softmaxed scores) and through the selected experts; non-selected experts get no gradient from that token.
* The **aux loss** pushes the *average* assignment distribution toward uniform usage across experts.

---

## Performance Notes

* It’s CPU only. With `-O3 -march=native`, it’s pleasantly snappy for toy scales.
* MoE speeds come from **sparsity**: each token hits only `k` experts, not all `E`.
* There’s no threading by default. If you want to experiment, the code is structured so hot loops are easy to spot.

---

## Limits & Scope

* This is an **educational** implementation. It’s small, readable, and faithful—but not a production engine.
* No fused kernels, no kv-cache tricks for generation speed, no GPU, no mixed precision.
* Dataset is tiny by default. You can point it at something bigger, but keep expectations healthy.

---

## FAQ

**Q: How do I change the number of experts or top-k?**
Edit `N_EXPERTS` and `TOP_K` at the top of `moe.c`. Keep `TOP_K ≤ N_EXPERTS` (the gate will sulk otherwise).

**Q: Can I use my own text?**
Yes—set the dataset path macro/variable, rebuild, and run. The loader is intentionally simple (byte-level).

**Q: Why is my aux loss weird?**
If one expert is hogging traffic, increase `AUX_LOSS_WEIGHT` a touch, or run longer. Also check initialization and learning rate.

**Q: Can I resume from a checkpoint?**
Yep. Point the loader at `tinymoe.bin`. The parameter layout is deterministic.

---

## Reading Guide (where to look in `moe.c`)

* **Params & Hyperparams:** top of file
* **Tensor helpers & RNG:** near the top
* **Model structs:** embedding, attention, **MoE**, layernorm, output head
* **Forward pass:** search `forward_transformer(...)`
* **Backward pass:** search `backward_transformer(...)`
* **Optimizer (Adam):** search `adam_step(...)`
* **Training loop:** `main()` — prints losses, saves checkpoints, runs a small inference demo

---

## Contributing

Bugs, PRs, and questions welcome! Please keep changes minimalistic and well-commented so we preserve the “read it in one go” ethos.

* Style: stay C99/C11-ish, stick to the standard library.
* Keep functions small and documented.
* If you add a feature, include a tiny example in comments.

---

## License

See `LICENSE`.

---

## Acknowledgments

* Andrej Karpathy for the blueprint and infectious minimalism.
* The open-source community for the inspiration and Shakespeare for the dataset and the drama.

---

### TL;DR

* **Run:** `gcc -O2 -o moe moe.c -lm && ./moe`
* **Learn:** open `moe.c` and read the comments.
* **Smile:** you just trained a sparse Transformer from scratch.

```
