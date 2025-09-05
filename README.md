## moe.c

> A **Mixture-of-Experts** Transformer you can **read in one sitting** and **run on your laptop**.  
> Training, inference, saving, loading—**all in one C file**.  
> No frameworks. No magic. Just you, the compiler, and a very confused gatekeeper.

---

## Why?

State-of-the-art LLMs use **sparse experts** to scale parameters without scaling compute.  
But how does that *actually* work?

- How does a **gating network** decide which experts touch a token?  
- How do you **route only to top-k** and still backprop through choices?  
- How do you **balance load** so the gate doesn’t fall in love with one expert and ghost the rest?

This repo answers those questions by building the **whole stack**—forward, backward, optimizer, data loader, checkpointing—in **one heavily commented C file (please ignore most of them)**.

If you can read C, you can understand MoE.

---

## What’s Inside

- GPT-2-style Transformer with **MoE feed-forward layers**.  
- **Top-k routing** (e.g. k=2) via a tiny **gating network**.  
- **Auxiliary load-balancing loss** so experts actually get used.  
- **Adam optimizer** from scratch.  
- **Training** on a tiny Shakespeare sample (bundled).  
- **Checkpointing** to `tinymoe.bin` and **reload + inference**.  

All in **~4000 lines**. No dependencies. except BLAS maybe.

---

## Quick Start

### 1) Build

```bash
gcc -O3 -march=native -ffast-math -o moe moe.c -lm
```

### 2) Run

```bash
./moe
```

You’ll see losses every ~50 steps:

```
Step    0 | Loss: 4.12 (Primary: 4.01, Aux: 1.11)
Step   50 | Loss: 3.57 (Primary: 3.46, Aux: 1.11)
...
Step  950 | Loss: 1.88 (Primary: 1.77, Aux: 1.11)
```

At the end it saves `tinymoe.bin` and generates text from a few prompts.

---

## Usage

### Train (default)

Just `./moe`. Hyperparams live at the top of `moe.c`—tweak, rebuild, rerun.

### Inference-only

Set `RUN_INFERENCE_ONLY 1` at the top, rebuild, run:

```bash
./moe
# streams completions for built-in prompts
```

### Checkpoint format

`tinymoe.bin` is a raw byte dump. Loading is literally `fread(&weights, sizeof(float), count, fp)`.

---

## Architecture Cheat-Sheet

```text
Token h ──► [Self-Attn] ──► h'
│
└─► [Gate] ─► scores over E experts ─► top-k indices & weights
│
├──► Expert i: FFN(h') ─┐
├──► Expert j: FFN(h') ─┤  weighted sum ► MoE output
└──► ... (only top-k) ───┘
```

- **Gatekeeper**: linear layer → softmax → top-k.  
- **Specialists**: independent FFNs (same shape).  
- **Council**: blend outputs by normalized gate weights.  
- **Aux loss**: nudges gate to use **all** experts, not a favorite.

---

## Reading Guide (where to look in `moe.c`)

| What | Where |
|---|---|
| Hyperparams | top of file |
| Tensor helpers | near top |
| Model structs | embedding, attention, **MoE**, layernorm, head |
| Forward | `forward_transformer(...)` |
| Backward | `backward_transformer(...)` |
| Optimizer | `adam_step(...)` |
| Training loop | `main()` |

---

## Performance Notes

- CPU only. With `-O3 -march=native` it’s pleasantly snappy on toy scales.  
- MoE speed comes from **sparsity**: each token hits only `k` experts, not all `E`.  
- No threading by default—hot loops are easy to spot if you want to add OpenMP.
- Also has BLAS with standard methods of optimization (whatever you like)

---

## Limits & Scope

- **Educational**. Small, readable, faithful—not production.  
- No fused kernels, no kv-cache, no GPU, no mixed precision.  
- Dataset is tiny. Point it at something bigger, but keep expectations healthy.

---

## FAQ

**Q:** How do I change experts or top-k?  
**A:** Edit `N_EXPERTS` and `TOP_K` at the top. Keep `TOP_K ≤ N_EXPERTS`.

**Q:** Can I use my own text?  
**A:** Yes—set the dataset path macro, rebuild, run.

**Q:** Why is aux loss weird?  
**A:** If one expert hogs traffic, bump `AUX_LOSS_WEIGHT` or train longer.

---

## Contributing

Bugs, PRs, questions welcome. Keep changes **minimalistic and well-commented** so we preserve the “read it in one go” ethos.

---

## License

MIT

---

### TL;DR

```bash
gcc -O3 -march=native -ffast-math -o moe moe.c -lm && ./moe
```

Open `moe.c`, read the comments, smile—you just trained a sparse Transformer from scratch.
