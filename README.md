moe.c

I took a Mixture-of-Experts (MoE) Transformer, threw it in a blender with the C programming language, and refused to stop until it all fit into one file. This is the result.

This project is a from-scratch, single-file implementation of a complete MoE Transformer lifecycle: training, inference, saving, and loading. It's designed to be a deeply educational tool for understanding how modern, sparse models work under the hood.

Inspired by (and eternally grateful for) Andrej Karpathy's minimalist deep learning projects like llama2.c and llm.c.
What is this?

State-of-the-art language models are increasingly using a Mixture-of-Experts architecture to scale up parameter counts without a proportional increase in compute. But how does that actually work? How does a "gating network" route tokens? How do you backpropagate through a sparse, dynamic choice?

This project answers those questions by building one from the ground up. The philosophy is simple:

    One File to Rule Them All. The entire model—architecture, forward(), backward(), the Adam optimizer, data loading, everything—is in moe.c. If you can read C, you can understand the entire stack.

    No Magic. The only #includes are for the standard library. There are no dependencies, no BLAS, no CUDA, no PyTorch. Just you, the C code, and the horrifyingly beautiful dance of backpropagation.

    Clarity Over All. The code is heavily commented and structured to be a narrative. It's meant to be read, not just run.

By stripping away all abstractions, we can get a raw, unfiltered look at the mechanics of a modern neural network.
Architecture

This is a standard GPT-2 style Transformer, but we've performed a bit of surgery on its feed-forward network (FFN) block. Instead of a single, dense FFN, each layer has a Mixture of Experts (MoE) layer:

    The Gatekeeper (Gating Network): A tiny linear layer that acts as a bouncer. It looks at each incoming token and produces a set of scores, deciding which "expert" is best suited for the job.

    The Specialists (Experts): A pool of several independent FFNs. Each one might learn to specialize in something different (e.g., punctuation, nouns, Python code, who knows?).

    The Council of Experts (Top-K Routing): We don't just send a token to one expert. The gating network picks the top k (e.g., 2) experts, and the token is processed by all of them. Their final outputs are then combined in a weighted average based on the gate's scores.

This means for any given token, most of the model's parameters are completely ignored, which is the key to MoE's efficiency.
Quick Start
1. Compile

You'll need gcc or clang. We need the math library (-lm), and you'll really want optimizations (-O2), otherwise training might finish sometime next year.

gcc -o moe moe.c -lm -O2

2. Run

Execute the compiled program. That's it.

./moe

The program will immediately start training on a small sample of Shakespeare.
Training

Unlike llama2.c which trains in Python and infers in C, this project handles the entire training process in C. The main() function is the training script.

When you run ./moe, you will see the training loss printed every 50 steps. The most important sign that the universe is in order is that this loss value steadily decreases.

Step    0 | Loss: 4.1234 (Primary: 4.0123, Aux: 1.1111) | Best: 4.1234
Step   50 | Loss: 3.5678 (Primary: 3.4567, Aux: 1.1111) | Best: 3.5678
...
Step  950 | Loss: 1.8765 (Primary: 1.7654, Aux: 1.1111) | Best: 1.8765

The Aux loss is the special "load balancing" loss that encourages the gating network to use all its experts, preventing it from getting lazy and only sending tokens to one favorite.

After training, a tinymoe.bin file containing the model weights will be saved to disk.
Inference

Once training is complete, the program will automatically demonstrate text generation from a few sample prompts. You can also modify main() to load the tinymoe.bin file and run inference directly.

This project was created as a learning exercise. It's a testament to the idea that to truly understand something, you should try to build it from scratch.
