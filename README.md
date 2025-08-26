moe.c

A from-scratch implementation of a Mixture-of-Experts (MoE) Transformer language model in a single C file. This project is a complete end-to-end pipeline for training and inference, designed for educational purposes.

Inspired by Andrej Karpathy's minimalist, from-scratch deep learning projects like llm.c.
Philosophy

The goal of this project is to demystify the Mixture-of-Experts architecture, which is becoming increasingly prevalent in state-of-the-art language models. While frameworks like PyTorch and TensorFlow provide powerful abstractions, they can obscure the fundamental mechanics of how these models work.

moe.c follows a different philosophy:

    Single File: The entire model—architecture, forward pass, backward pass, optimizer, and data loader—is contained in this one file.

    No Dependencies: The code is written in pure C with only the standard library and the math library. No BLAS, no CUDA, no magic.

    Educational Clarity: The code is heavily commented and structured to be read from top to bottom, explaining each component from the ground up.

By building everything from scratch, we can gain a deeper intuition for the entire lifecycle of a neural network, from memory allocation and weight initialization to the intricate dance of backpropagation.
Architecture

This project implements a standard GPT-2 style Transformer, but with a key modification: the feed-forward network (FFN) in each block is replaced by a Mixture of Experts (MoE) layer.

Each MoE layer consists of:

    A Gating Network: A small, learnable router that examines each token and decides which experts are best suited to process it.

    A Set of Experts: Multiple independent FFNs that specialize in different aspects of the data.

    Top-K Routing: For each token, the gating network selects the top k experts, and their outputs are combined with a weighted sum.

This allows the model to activate only a sparse subset of its weights for each token, leading to more efficient inference.
How to Run
1. Compile

You will need a C compiler (like gcc or clang). The -lm flag is required to link the math library. Using optimization flags like -O2 is highly recommended for performance.

gcc -o moe moe.c -lm -O2

2. Run

Execute the compiled program from your terminal.

./moe

What to Expect

The program will start training immediately on a small sample of Shakespeare's text. You should see the following output:

    Initialization: The model configuration and approximate parameter count will be printed.

    Training Progress: Every 50 steps, the current loss will be printed. The most important sign of success is watching this loss value steadily decrease.

    Step    0 | Loss: 4.1234 (Primary: 4.0123, Aux: 1.1111) | Best: 4.1234
    Step   50 | Loss: 3.5678 (Primary: 3.4567, Aux: 1.1111) | Best: 3.5678
    ...

    Expert Utilization: Periodically, the program will analyze and print how often each expert is being used. A healthy model will learn to distribute the workload across all experts.

    Text Generation: After training is complete, the program will generate text from a few sample prompts.

    Model Checkpoint: The final trained model weights will be saved to a file named tinymoe.bin.

Future Work

This project is a foundation. There are many exciting directions to take it:

    Data Loaders: Implement a more robust data loader to train on larger text files.

    Tokenizer: Integrate a more advanced tokenizer (like SentencePiece) instead of the simple character-level one.

    Performance: Optimize the matrix multiplication and other hot paths with techniques like SIMD intrinsics.

    GGUF Support: Add the capability to load quantized weights from popular MoE models like Mixtral.

This project was created as a learning exercise and is heavily inspired by the work of Andrej Karpathy.
