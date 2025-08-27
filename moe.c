#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h> // Add this at the top with other includes
#include <omp.h>
#define DEBUG 1

// ============================================================================
// ## 1. ARCHITECTURE DEFINITION
// ============================================================================

// Configuration struct to hold all hyperparameters
typedef struct
{
    int vocab_size;  // size of the vocabulary
    int seq_len;     // sequence length (e.g., 256)
    int embed_dim;   // embedding dimension (e.g., 128)
    int num_layers;  // number of transformer layers (e.g., 4)
    int num_heads;   // number of attention heads (e.g., 4)
    int num_experts; // number of experts in each MoE layer (e.g., 8)
    int top_k;       // number of experts to route to for each token (e.g., 2)
    int hidden_dim;  // hidden dimension for expert FFNs (e.g., 512)
} Config;

// A single Feed-Forward Network (FFN), which will serve as an "Expert"
typedef struct
{
    float *w1; // weight matrix for the first linear layer
    float *b1; // bias for the first linear layer
    float *w2; // weight matrix for the second linear layer
    float *b2; // bias for the second linear layer
} Expert;

// The Mixture of Experts (MoE) Layer
typedef struct
{
    // The Gating Network: a simple linear layer that decides which expert to use
    float *gating_w; // weights for the gating network
    float *gating_b; // bias for the gating network

    // An array of expert networks
    Expert *experts; // pointer to an array of 'num_experts' Expert structs
} MoELayer;

// A single Transformer Block/Layer
typedef struct
{
    // Attention mechanism weights
    float *attn_qkv_w;  // combined weights for Q, K, V projections
    float *attn_qkv_b;  // combined biases for Q, K, V projections
    float *attn_proj_w; // weights for the output projection
    float *attn_proj_b; // biases for the output projection

    // Layer normalization weights/biases
    float *ln1_gamma; // scale for the first LayerNorm
    float *ln1_beta;  // shift for the first LayerNorm
    float *ln2_gamma; // scale for the second LayerNorm
    float *ln2_beta;  // shift for the second LayerNorm

    // The MoE layer that replaces the standard FFN
    MoELayer moe_layer;
} TransformerBlock;

// The complete GPT-2 MoE Model
typedef struct
{
    Config config;

    // Model weights
    float *token_embeddings;  // (vocab_size, embed_dim)
    float *pos_embeddings;    // (seq_len, embed_dim)
    TransformerBlock *layers; // pointer to an array of 'num_layers' TransformerBlock structs
    float *final_ln_gamma;    // final layer norm scale
    float *final_ln_beta;     // final layer norm shift
} GPT2_MoE_Model;

// Struct to hold the activations during the forward pass (the "state")
typedef struct
{
    float *embedding_out; // output of embedding layer (seq_len, embed_dim)
    float *layer_outputs; // intermediate outputs from each transformer block
    float *attn_out;      // attention output buffer
    float *moe_out;       // MoE output buffer
    float *logits;        // final output logits

    // MoE specific state for loss calculation
    float *gating_logits;  // raw output of the gating network for each token
    int *expert_indices;   // indices of the top_k experts chosen for each token
    float *expert_weights; // weights for the chosen experts
    float *expert_outputs; // outputs from individual experts

    // Temporary buffers
    float *temp_buffer;
    float *temp_buffer2;

    float *gating_scores_buffer;    // (num_experts)
    float *hidden_buffer;           // (hidden_dim)
    float *expert_out_buffer;       // (embed_dim)
    float *attention_scores_buffer; // (seq_len * seq_len)
    float *qkv_buffer;

    // Forward pass intermediate storage
    float *ln1_outputs;    // (num_layers * seq_len * embed_dim)
    float *ln2_outputs;    // (num_layers * seq_len * embed_dim)
    float *attn_outputs;   // (num_layers * seq_len * embed_dim)
    float *attn_residual;  // (num_layers * seq_len * embed_dim)
    float *moe_outputs;    // (num_layers * seq_len * embed_dim)
    float *gating_scores;  // (num_layers * seq_len * num_experts)
    float *expert_hiddens; // (num_layers * num_experts * seq_len * hidden_dim)
    float *attn_concat;

    // Gradient buffers
    float *d_embedding_out;
    float *d_layer_outputs;
    float *d_attn_out;
    float *d_moe_out;
    float *d_temp_buffer;
    float *d_temp_buffer2;

    // Model parameter gradients
    float *d_token_embeddings;
    float *d_pos_embeddings;
    float *d_final_ln_gamma;
    float *d_final_ln_beta;

    // Layer gradients (will be arrays)
    float **d_attn_qkv_w;
    float **d_attn_qkv_b;
    float **d_attn_proj_w;
    float **d_attn_proj_b;
    float **d_ln1_gamma;
    float **d_ln1_beta;
    float **d_ln2_gamma;
    float **d_ln2_beta;
    float **d_gating_w;
    float **d_gating_b;

    // Expert gradients (3D arrays: [layer][expert][weights])
    float ***d_expert_w1;
    float ***d_expert_b1;
    float ***d_expert_w2;
    float ***d_expert_b2;

    float *d_ln1_out;           // (seq_len * embed_dim)
    float *d_ln2_out;           // (seq_len * embed_dim)
    float *d_attn_residual_out; // (seq_len * embed_dim)
    float *d_attn_concat;       // (seq_len * embed_dim)
    float *d_attention_scores;  // (seq_len * seq_len)
    float *d_query;             // (seq_len * head_dim)
    float *d_key;               // (seq_len * head_dim)
    float *d_value;             // (seq_len * head_dim)
    float *d_softmax_scores;    // (seq_len * seq_len)

    float *attention_probs; // (num_layers * seq_len * seq_len)

} RunState;

typedef struct
{
    // Adam optimizer state for token embeddings
    float *m_token_embeddings;
    float *v_token_embeddings;

    // Adam optimizer state for positional embeddings
    float *m_pos_embeddings;
    float *v_pos_embeddings;

    // Adam optimizer state for final layer norm
    float *m_final_ln_gamma;
    float *v_final_ln_gamma;
    float *m_final_ln_beta;
    float *v_final_ln_beta;

    // Adam optimizer state for layers
    float **m_attn_qkv_w;
    float **v_attn_qkv_w;
    float **m_attn_qkv_b;
    float **v_attn_qkv_b;
    float **m_attn_proj_w;
    float **v_attn_proj_w;
    float **m_attn_proj_b;
    float **v_attn_proj_b;
    float **m_ln1_gamma;
    float **v_ln1_gamma;
    float **m_ln1_beta;
    float **v_ln1_beta;
    float **m_ln2_gamma;
    float **v_ln2_gamma;
    float **m_ln2_beta;
    float **v_ln2_beta;
    float **m_gating_w;
    float **v_gating_w;
    float **m_gating_b;
    float **v_gating_b;

    // Adam optimizer state for experts
    float ***m_expert_w1;
    float ***v_expert_w1;
    float ***m_expert_b1;
    float ***v_expert_b1;
    float ***m_expert_w2;
    float ***v_expert_w2;
    float ***m_expert_b2;
    float ***v_expert_b2;

    // Adam hyperparameters
    float beta1;
    float beta2;
    float eps;
    int step;
} Optimizer;

typedef struct
{
    char *data;
    int size;
    int *tokens;
    int num_tokens;
    char *vocab[256];
    int vocab_size;
} Dataset;

// ============================================================================
// ## FORWARD DECLARATIONS
// ============================================================================
void analyze_expert_usage(RunState *state, Config *config, int num_steps);
int load_model(GPT2_MoE_Model *model, const char *filename);
void save_model(GPT2_MoE_Model *model, const char *filename);
void generate_text(GPT2_MoE_Model *model, RunState *state, Dataset *dataset,
                   const char *prompt, int max_tokens, float temperature, float top_p);
void free_model(GPT2_MoE_Model *model);
void free_state(RunState *state, Config *config);
void free_dataset(Dataset *dataset);
void free_optimizer(Optimizer *opt, Config *config);

// ============================================================================
// ## 2. UTILITY FUNCTIONS
// ============================================================================

// Random number generator
float randn()
{
    static int has_spare = 0;
    static float spare;
    if (has_spare)
    {
        has_spare = 0;
        return spare;
    }
    has_spare = 1;
    static float u, v, mag;
    do
    {
        u = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        v = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        mag = u * u + v * v;
    } while (mag >= 1.0f || mag == 0.0f);
    mag = sqrt(-2.0f * log(mag) / mag);
    spare = v * mag;
    return u * mag;
}

// Initialize weights with Xavier/Glorot initialization
void init_weights(float *weights, int size, int fan_in)
{
    float std = sqrt(2.0f / fan_in);
    for (int i = 0; i < size; i++)
    {
        weights[i] = randn() * std;
    }
}

// Softmax function
void softmax(float *x, int size)
{
    if (size <= 0)
        return;

    // Find maximum value for numerical stability
    float max_val = x[0];
#pragma omp parallel for
    for (int i = 1; i < size; i++)
    {
#pragma omp critical
        if (x[i] > max_val)
            max_val = x[i];
    }

    // Compute exponentials and sum
    float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < size; i++)
    {
        x[i] = expf(x[i] - max_val);
        // Handle potential NaN/inf values
        if (!isfinite(x[i]) || x[i] < 0.0f)
            x[i] = 1e-8f;
        sum += x[i];
    }

    // Normalize with safety check
    if (sum < 1e-8f)
    {
        // If sum is too small, use uniform distribution
        float uniform_val = 1.0f / size;
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            x[i] = uniform_val;
        }
    }
    else
    {
// Normal softmax normalization
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            x[i] /= sum;
            // Clamp to prevent extreme values
            if (x[i] < 1e-8f)
                x[i] = 1e-8f;
            if (x[i] > 1.0f)
                x[i] = 1.0f;
        }
    }
}
// Layer normalization
void layer_norm(float *out, float *x, float *gamma, float *beta, int size)
{
    if (size <= 0)
        return;

    float mean = 0.0f;
    for (int i = 0; i < size; i++)
    {
        mean += x[i];
    }
    mean /= size;

    float var = 0.0f;
    for (int i = 0; i < size; i++)
    {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= size;

    float std = sqrtf(var + 1e-5f);
    for (int i = 0; i < size; i++)
    {
        out[i] = (x[i] - mean) / std * gamma[i] + beta[i];
        if (!isfinite(out[i]))
            out[i] = 0.0f; // Handle NaN/inf
    }
}

// Matrix multiplication: C = A * B^T
void matmul(float *c, float *a, float *b, int n, int d, int k)
{
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
        {
            float sum = 0.0f;
            for (int l = 0; l < d; l++)
            {
                sum += a[i * d + l] * b[j * d + l];
            }
            c[i * k + j] = sum;
        }
    }
}

// Add bias
void add_bias(float *x, float *bias, int n, int d)
{
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < d; j++)
        {
            x[i * d + j] += bias[j];
        }
    }
}

// ReLU activation
void relu(float *x, int size)
{
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        if (x[i] < 0)
            x[i] = 0;
    }
}

// Top-K selection for MoE gating
void topk_indices(int *indices, float *values, float *scores, int num_experts, int k)
{
    if (DEBUG)
        printf("    topk_indices: num_experts=%d, k=%d\n", num_experts, k);

    // Simple approach: find k largest elements
    for (int i = 0; i < k; i++)
    {
        int best_idx = 0;
        float best_score = -INFINITY;

        for (int j = 0; j < num_experts; j++)
        {
            // Skip already selected
            int already_selected = 0;
            for (int prev = 0; prev < i; prev++)
            {
                if (indices[prev] == j)
                {
                    already_selected = 1;
                    break;
                }
            }
            if (already_selected)
                continue;

            if (scores[j] > best_score)
            {
                best_score = scores[j];
                best_idx = j;
            }
        }

        indices[i] = best_idx;
        values[i] = scores[best_idx];

        if (DEBUG)
            printf("      Selected expert %d with score %.6f\n", best_idx, best_score);
    }
}

void init_optimizer(Optimizer *opt, Config *config)
{
    opt->beta1 = 0.9f;
    opt->beta2 = 0.999f;
    opt->eps = 1e-8f;
    opt->step = 0;

    // Allocate Adam state for token embeddings
    opt->m_token_embeddings = calloc(config->vocab_size * config->embed_dim, sizeof(float));
    opt->v_token_embeddings = calloc(config->vocab_size * config->embed_dim, sizeof(float));

    // Allocate Adam state for positional embeddings
    opt->m_pos_embeddings = calloc(config->seq_len * config->embed_dim, sizeof(float));
    opt->v_pos_embeddings = calloc(config->seq_len * config->embed_dim, sizeof(float));

    // Allocate Adam state for final layer norm
    opt->m_final_ln_gamma = calloc(config->embed_dim, sizeof(float));
    opt->v_final_ln_gamma = calloc(config->embed_dim, sizeof(float));
    opt->m_final_ln_beta = calloc(config->embed_dim, sizeof(float));
    opt->v_final_ln_beta = calloc(config->embed_dim, sizeof(float));

    // Allocate layer arrays
    opt->m_attn_qkv_w = malloc(config->num_layers * sizeof(float *));
    opt->v_attn_qkv_w = malloc(config->num_layers * sizeof(float *));
    opt->m_attn_qkv_b = malloc(config->num_layers * sizeof(float *));
    opt->v_attn_qkv_b = malloc(config->num_layers * sizeof(float *));
    opt->m_attn_proj_w = malloc(config->num_layers * sizeof(float *));
    opt->v_attn_proj_w = malloc(config->num_layers * sizeof(float *));
    opt->m_attn_proj_b = malloc(config->num_layers * sizeof(float *));
    opt->v_attn_proj_b = malloc(config->num_layers * sizeof(float *));
    opt->m_ln1_gamma = malloc(config->num_layers * sizeof(float *));
    opt->v_ln1_gamma = malloc(config->num_layers * sizeof(float *));
    opt->m_ln1_beta = malloc(config->num_layers * sizeof(float *));
    opt->v_ln1_beta = malloc(config->num_layers * sizeof(float *));
    opt->m_ln2_gamma = malloc(config->num_layers * sizeof(float *));
    opt->v_ln2_gamma = malloc(config->num_layers * sizeof(float *));
    opt->m_ln2_beta = malloc(config->num_layers * sizeof(float *));
    opt->v_ln2_beta = malloc(config->num_layers * sizeof(float *));
    opt->m_gating_w = malloc(config->num_layers * sizeof(float *));
    opt->v_gating_w = malloc(config->num_layers * sizeof(float *));
    opt->m_gating_b = malloc(config->num_layers * sizeof(float *));
    opt->v_gating_b = malloc(config->num_layers * sizeof(float *));

    // Allocate expert arrays
    opt->m_expert_w1 = malloc(config->num_layers * sizeof(float **));
    opt->v_expert_w1 = malloc(config->num_layers * sizeof(float **));
    opt->m_expert_b1 = malloc(config->num_layers * sizeof(float **));
    opt->v_expert_b1 = malloc(config->num_layers * sizeof(float **));
    opt->m_expert_w2 = malloc(config->num_layers * sizeof(float **));
    opt->v_expert_w2 = malloc(config->num_layers * sizeof(float **));
    opt->m_expert_b2 = malloc(config->num_layers * sizeof(float **));
    opt->v_expert_b2 = malloc(config->num_layers * sizeof(float **));

    for (int l = 0; l < config->num_layers; l++)
    {
        // Allocate layer Adam states
        opt->m_attn_qkv_w[l] = calloc(config->embed_dim * 3 * config->embed_dim, sizeof(float));
        opt->v_attn_qkv_w[l] = calloc(config->embed_dim * 3 * config->embed_dim, sizeof(float));
        opt->m_attn_qkv_b[l] = calloc(3 * config->embed_dim, sizeof(float));
        opt->v_attn_qkv_b[l] = calloc(3 * config->embed_dim, sizeof(float));
        opt->m_attn_proj_w[l] = calloc(config->embed_dim * config->embed_dim, sizeof(float));
        opt->v_attn_proj_w[l] = calloc(config->embed_dim * config->embed_dim, sizeof(float));
        opt->m_attn_proj_b[l] = calloc(config->embed_dim, sizeof(float));
        opt->v_attn_proj_b[l] = calloc(config->embed_dim, sizeof(float));
        opt->m_ln1_gamma[l] = calloc(config->embed_dim, sizeof(float));
        opt->v_ln1_gamma[l] = calloc(config->embed_dim, sizeof(float));
        opt->m_ln1_beta[l] = calloc(config->embed_dim, sizeof(float));
        opt->v_ln1_beta[l] = calloc(config->embed_dim, sizeof(float));
        opt->m_ln2_gamma[l] = calloc(config->embed_dim, sizeof(float));
        opt->v_ln2_gamma[l] = calloc(config->embed_dim, sizeof(float));
        opt->m_ln2_beta[l] = calloc(config->embed_dim, sizeof(float));
        opt->v_ln2_beta[l] = calloc(config->embed_dim, sizeof(float));
        opt->m_gating_w[l] = calloc(config->embed_dim * config->num_experts, sizeof(float));
        opt->v_gating_w[l] = calloc(config->embed_dim * config->num_experts, sizeof(float));
        opt->m_gating_b[l] = calloc(config->num_experts, sizeof(float));
        opt->v_gating_b[l] = calloc(config->num_experts, sizeof(float));

        // Allocate expert Adam states
        opt->m_expert_w1[l] = malloc(config->num_experts * sizeof(float *));
        opt->v_expert_w1[l] = malloc(config->num_experts * sizeof(float *));
        opt->m_expert_b1[l] = malloc(config->num_experts * sizeof(float *));
        opt->v_expert_b1[l] = malloc(config->num_experts * sizeof(float *));
        opt->m_expert_w2[l] = malloc(config->num_experts * sizeof(float *));
        opt->v_expert_w2[l] = malloc(config->num_experts * sizeof(float *));
        opt->m_expert_b2[l] = malloc(config->num_experts * sizeof(float *));
        opt->v_expert_b2[l] = malloc(config->num_experts * sizeof(float *));

        for (int e = 0; e < config->num_experts; e++)
        {
            opt->m_expert_w1[l][e] = calloc(config->embed_dim * config->hidden_dim, sizeof(float));
            opt->v_expert_w1[l][e] = calloc(config->embed_dim * config->hidden_dim, sizeof(float));
            opt->m_expert_b1[l][e] = calloc(config->hidden_dim, sizeof(float));
            opt->v_expert_b1[l][e] = calloc(config->hidden_dim, sizeof(float));
            opt->m_expert_w2[l][e] = calloc(config->hidden_dim * config->embed_dim, sizeof(float));
            opt->v_expert_w2[l][e] = calloc(config->hidden_dim * config->embed_dim, sizeof(float));
            opt->m_expert_b2[l][e] = calloc(config->embed_dim, sizeof(float));
            opt->v_expert_b2[l][e] = calloc(config->embed_dim, sizeof(float));
        }
    }
}

// ============================================================================
// ## 3. MEMORY ALLOCATION AND INITIALIZATION
// ============================================================================

void build_model(GPT2_MoE_Model *model, Config *config)
{
    model->config = *config;

    // Allocate token embeddings
    model->token_embeddings = malloc(config->vocab_size * config->embed_dim * sizeof(float));
    init_weights(model->token_embeddings, config->vocab_size * config->embed_dim, config->embed_dim);

    // Allocate positional embeddings
    model->pos_embeddings = malloc(config->seq_len * config->embed_dim * sizeof(float));
    init_weights(model->pos_embeddings, config->seq_len * config->embed_dim, config->embed_dim);

    // Allocate transformer layers
    model->layers = malloc(config->num_layers * sizeof(TransformerBlock));

    for (int l = 0; l < config->num_layers; l++)
    {
        TransformerBlock *layer = &model->layers[l];

        // Attention weights
        int qkv_size = config->embed_dim * 3 * config->embed_dim;
        layer->attn_qkv_w = malloc(qkv_size * sizeof(float));
        layer->attn_qkv_b = malloc(3 * config->embed_dim * sizeof(float));
        layer->attn_proj_w = malloc(config->embed_dim * config->embed_dim * sizeof(float));
        layer->attn_proj_b = malloc(config->embed_dim * sizeof(float));

        init_weights(layer->attn_qkv_w, qkv_size, config->embed_dim);
        memset(layer->attn_qkv_b, 0, 3 * config->embed_dim * sizeof(float));
        init_weights(layer->attn_proj_w, config->embed_dim * config->embed_dim, config->embed_dim);
        memset(layer->attn_proj_b, 0, config->embed_dim * sizeof(float));

        // Layer norm
        layer->ln1_gamma = malloc(config->embed_dim * sizeof(float));
        layer->ln1_beta = malloc(config->embed_dim * sizeof(float));
        layer->ln2_gamma = malloc(config->embed_dim * sizeof(float));
        layer->ln2_beta = malloc(config->embed_dim * sizeof(float));

        for (int i = 0; i < config->embed_dim; i++)
        {
            layer->ln1_gamma[i] = 1.0f;
            layer->ln1_beta[i] = 0.0f;
            layer->ln2_gamma[i] = 1.0f;
            layer->ln2_beta[i] = 0.0f;
        }

        // MoE layer
        MoELayer *moe = &layer->moe_layer;

        // Gating network
        moe->gating_w = malloc(config->embed_dim * config->num_experts * sizeof(float));
        moe->gating_b = malloc(config->num_experts * sizeof(float));
        init_weights(moe->gating_w, config->embed_dim * config->num_experts, config->embed_dim);
        memset(moe->gating_b, 0, config->num_experts * sizeof(float));

        // Experts
        moe->experts = malloc(config->num_experts * sizeof(Expert));
        for (int e = 0; e < config->num_experts; e++)
        {
            Expert *expert = &moe->experts[e];

            expert->w1 = malloc(config->embed_dim * config->hidden_dim * sizeof(float));
            expert->b1 = malloc(config->hidden_dim * sizeof(float));
            expert->w2 = malloc(config->hidden_dim * config->embed_dim * sizeof(float));
            expert->b2 = malloc(config->embed_dim * sizeof(float));

            init_weights(expert->w1, config->embed_dim * config->hidden_dim, config->embed_dim);
            memset(expert->b1, 0, config->hidden_dim * sizeof(float));
            init_weights(expert->w2, config->hidden_dim * config->embed_dim, config->hidden_dim);
            memset(expert->b2, 0, config->embed_dim * sizeof(float));
        }
    }

    // Final layer norm
    model->final_ln_gamma = malloc(config->embed_dim * sizeof(float));
    model->final_ln_beta = malloc(config->embed_dim * sizeof(float));
    for (int i = 0; i < config->embed_dim; i++)
    {
        model->final_ln_gamma[i] = 1.0f;
        model->final_ln_beta[i] = 0.0f;
    }
}

void build_state(RunState *state, Config *config)
{
    // Add null pointer checks for all memory allocations
    state->embedding_out = malloc(config->seq_len * config->embed_dim * sizeof(float));
    if (!state->embedding_out)
    {
        printf("Failed to allocate embedding_out\n");
        exit(1);
    }

    state->layer_outputs = malloc(config->num_layers * config->seq_len * config->embed_dim * sizeof(float));
    if (!state->layer_outputs)
    {
        printf("Failed to allocate layer_outputs\n");
        exit(1);
    }

    state->attn_out = malloc(config->seq_len * config->embed_dim * sizeof(float));
    if (!state->attn_out)
    {
        printf("Failed to allocate attn_out\n");
        exit(1);
    }

    state->moe_out = malloc(config->seq_len * config->embed_dim * sizeof(float));
    if (!state->moe_out)
    {
        printf("Failed to allocate moe_out\n");
        exit(1);
    }

    state->attention_probs = malloc(config->num_layers * config->seq_len * config->seq_len * sizeof(float));
    if (!state->attention_probs)
    {
        printf("Failed to allocate attention_probs\n");
        exit(1);
    }

    state->logits = malloc(config->seq_len * config->vocab_size * sizeof(float));
    if (!state->logits)
    {
        printf("Failed to allocate logits\n");
        exit(1);
    }

    state->gating_logits = malloc(config->num_layers * config->seq_len * config->num_experts * sizeof(float));
    if (!state->gating_logits)
    {
        printf("Failed to allocate gating_logits\n");
        exit(1);
    }

    state->expert_indices = malloc(config->num_layers * config->seq_len * config->top_k * sizeof(int));
    if (!state->expert_indices)
    {
        printf("Failed to allocate expert_indices\n");
        exit(1);
    }

    state->expert_weights = malloc(config->num_layers * config->seq_len * config->top_k * sizeof(float));
    if (!state->expert_weights)
    {
        printf("Failed to allocate expert_weights\n");
        exit(1);
    }

    state->expert_outputs = malloc(config->num_layers * config->num_experts * config->seq_len * config->embed_dim * sizeof(float));
    if (!state->expert_outputs)
    {
        printf("Failed to allocate expert_outputs\n");
        exit(1);
    }

    state->temp_buffer = malloc(config->seq_len * config->embed_dim * sizeof(float));
    if (!state->temp_buffer)
    {
        printf("Failed to allocate temp_buffer\n");
        exit(1);
    }

    state->temp_buffer2 = malloc(config->seq_len * config->embed_dim * sizeof(float));
    if (!state->temp_buffer2)
    {
        printf("Failed to allocate temp_buffer2\n");
        exit(1);
    }

    state->gating_scores_buffer = malloc(config->num_experts * sizeof(float));
    if (!state->gating_scores_buffer)
    {
        printf("Failed to allocate gating_scores_buffer\n");
        exit(1);
    }

    state->hidden_buffer = malloc(config->hidden_dim * sizeof(float));
    if (!state->hidden_buffer)
    {
        printf("Failed to allocate hidden_buffer\n");
        exit(1);
    }

    state->expert_out_buffer = malloc(config->embed_dim * sizeof(float));
    if (!state->expert_out_buffer)
    {
        printf("Failed to allocate expert_out_buffer\n");
        exit(1);
    }

    state->attention_scores_buffer = malloc(config->seq_len * config->seq_len * sizeof(float));
    if (!state->attention_scores_buffer)
    {
        printf("Failed to allocate attention_scores_buffer\n");
        exit(1);
    }

    state->qkv_buffer = malloc(config->seq_len * 3 * config->embed_dim * sizeof(float));
    if (!state->qkv_buffer)
    {
        printf("Failed to allocate qkv_buffer\n");
        exit(1);
    }

    state->ln1_outputs = malloc(config->num_layers * config->seq_len * config->embed_dim * sizeof(float));
    if (!state->ln1_outputs)
    {
        printf("Failed to allocate ln1_outputs\n");
        exit(1);
    }

    state->ln2_outputs = malloc(config->num_layers * config->seq_len * config->embed_dim * sizeof(float));
    if (!state->ln2_outputs)
    {
        printf("Failed to allocate ln2_outputs\n");
        exit(1);
    }

    state->attn_outputs = malloc(config->num_layers * config->seq_len * config->embed_dim * sizeof(float));
    if (!state->attn_outputs)
    {
        printf("Failed to allocate attn_outputs\n");
        exit(1);
    }

    state->attn_residual = malloc(config->num_layers * config->seq_len * config->embed_dim * sizeof(float));
    if (!state->attn_residual)
    {
        printf("Failed to allocate attn_residual\n");
        exit(1);
    }

    state->moe_outputs = malloc(config->num_layers * config->seq_len * config->embed_dim * sizeof(float));
    if (!state->moe_outputs)
    {
        printf("Failed to allocate moe_outputs\n");
        exit(1);
    }

    state->gating_scores = malloc(config->num_layers * config->seq_len * config->num_experts * sizeof(float));
    if (!state->gating_scores)
    {
        printf("Failed to allocate gating_scores\n");
        exit(1);
    }

    state->expert_hiddens = malloc(config->num_layers * config->num_experts * config->seq_len * config->hidden_dim * sizeof(float));
    if (!state->expert_hiddens)
    {
        printf("Failed to allocate expert_hiddens\n");
        exit(1);
    }

    state->attn_concat = malloc(config->num_layers * config->seq_len * config->embed_dim * sizeof(float));
    if (!state->attn_concat)
    {
        printf("Failed to allocate attn_concat\n");
        exit(1);
    }

    state->d_embedding_out = calloc(config->seq_len * config->embed_dim, sizeof(float));
    if (!state->d_embedding_out)
    {
        printf("Failed to allocate d_embedding_out\n");
        exit(1);
    }

    state->d_layer_outputs = calloc(config->num_layers * config->seq_len * config->embed_dim, sizeof(float));
    if (!state->d_layer_outputs)
    {
        printf("Failed to allocate d_layer_outputs\n");
        exit(1);
    }

    state->d_attn_out = calloc(config->seq_len * config->embed_dim, sizeof(float));
    if (!state->d_attn_out)
    {
        printf("Failed to allocate d_attn_out\n");
        exit(1);
    }

    state->d_moe_out = calloc(config->seq_len * config->embed_dim, sizeof(float));
    if (!state->d_moe_out)
    {
        printf("Failed to allocate d_moe_out\n");
        exit(1);
    }

    state->d_temp_buffer = calloc(config->seq_len * config->embed_dim, sizeof(float));
    if (!state->d_temp_buffer)
    {
        printf("Failed to allocate d_temp_buffer\n");
        exit(1);
    }

    state->d_temp_buffer2 = calloc(config->seq_len * config->embed_dim, sizeof(float));
    if (!state->d_temp_buffer2)
    {
        printf("Failed to allocate d_temp_buffer2\n");
        exit(1);
    }

    state->d_token_embeddings = calloc(config->vocab_size * config->embed_dim, sizeof(float));
    if (!state->d_token_embeddings)
    {
        printf("Failed to allocate d_token_embeddings\n");
        exit(1);
    }

    state->d_pos_embeddings = calloc(config->seq_len * config->embed_dim, sizeof(float));
    if (!state->d_pos_embeddings)
    {
        printf("Failed to allocate d_pos_embeddings\n");
        exit(1);
    }

    state->d_final_ln_gamma = calloc(config->embed_dim, sizeof(float));
    if (!state->d_final_ln_gamma)
    {
        printf("Failed to allocate d_final_ln_gamma\n");
        exit(1);
    }

    state->d_final_ln_beta = calloc(config->embed_dim, sizeof(float));
    if (!state->d_final_ln_beta)
    {
        printf("Failed to allocate d_final_ln_beta\n");
        exit(1);
    }

    state->d_attn_qkv_w = malloc(config->num_layers * sizeof(float *));
    if (!state->d_attn_qkv_w)
    {
        printf("Failed to allocate d_attn_qkv_w\n");
        exit(1);
    }

    state->d_attn_qkv_b = malloc(config->num_layers * sizeof(float *));
    if (!state->d_attn_qkv_b)
    {
        printf("Failed to allocate d_attn_qkv_b\n");
        exit(1);
    }

    state->d_attn_proj_w = malloc(config->num_layers * sizeof(float *));
    if (!state->d_attn_proj_w)
    {
        printf("Failed to allocate d_attn_proj_w\n");
        exit(1);
    }

    state->d_attn_proj_b = malloc(config->num_layers * sizeof(float *));
    if (!state->d_attn_proj_b)
    {
        printf("Failed to allocate d_attn_proj_b\n");
        exit(1);
    }

    state->d_ln1_gamma = malloc(config->num_layers * sizeof(float *));
    if (!state->d_ln1_gamma)
    {
        printf("Failed to allocate d_ln1_gamma\n");
        exit(1);
    }

    state->d_ln1_beta = malloc(config->num_layers * sizeof(float *));
    if (!state->d_ln1_beta)
    {
        printf("Failed to allocate d_ln1_beta\n");
        exit(1);
    }

    state->d_ln2_gamma = malloc(config->num_layers * sizeof(float *));
    if (!state->d_ln2_gamma)
    {
        printf("Failed to allocate d_ln2_gamma\n");
        exit(1);
    }

    state->d_ln2_beta = malloc(config->num_layers * sizeof(float *));
    if (!state->d_ln2_beta)
    {
        printf("Failed to allocate d_ln2_beta\n");
        exit(1);
    }

    state->d_gating_w = malloc(config->num_layers * sizeof(float *));
    if (!state->d_gating_w)
    {
        printf("Failed to allocate d_gating_w\n");
        exit(1);
    }

    state->d_gating_b = malloc(config->num_layers * sizeof(float *));
    if (!state->d_gating_b)
    {
        printf("Failed to allocate d_gating_b\n");
        exit(1);
    }

    state->d_expert_w1 = malloc(config->num_layers * sizeof(float **));
    if (!state->d_expert_w1)
    {
        printf("Failed to allocate d_expert_w1\n");
        exit(1);
    }

    state->d_expert_b1 = malloc(config->num_layers * sizeof(float **));
    if (!state->d_expert_b1)
    {
        printf("Failed to allocate d_expert_b1\n");
        exit(1);
    }

    state->d_expert_w2 = malloc(config->num_layers * sizeof(float **));
    if (!state->d_expert_w2)
    {
        printf("Failed to allocate d_expert_w2\n");
        exit(1);
    }

    state->d_expert_b2 = malloc(config->num_layers * sizeof(float **));
    if (!state->d_expert_b2)
    {
        printf("Failed to allocate d_expert_b2\n");
        exit(1);
    }

    for (int l = 0; l < config->num_layers; l++)
    {
        state->d_attn_qkv_w[l] = calloc(config->embed_dim * 3 * config->embed_dim, sizeof(float));
        if (!state->d_attn_qkv_w[l])
        {
            printf("Failed to allocate d_attn_qkv_w[%d]\n", l);
            exit(1);
        }

        state->d_attn_qkv_b[l] = calloc(3 * config->embed_dim, sizeof(float));
        if (!state->d_attn_qkv_b[l])
        {
            printf("Failed to allocate d_attn_qkv_b[%d]\n", l);
            exit(1);
        }

        state->d_attn_proj_w[l] = calloc(config->embed_dim * config->embed_dim, sizeof(float));
        if (!state->d_attn_proj_w[l])
        {
            printf("Failed to allocate d_attn_proj_w[%d]\n", l);
            exit(1);
        }

        state->d_attn_proj_b[l] = calloc(config->embed_dim, sizeof(float));
        if (!state->d_attn_proj_b[l])
        {
            printf("Failed to allocate d_attn_proj_b[%d]\n", l);
            exit(1);
        }

        state->d_ln1_gamma[l] = calloc(config->embed_dim, sizeof(float));
        if (!state->d_ln1_gamma[l])
        {
            printf("Failed to allocate d_ln1_gamma[%d]\n", l);
            exit(1);
        }

        state->d_ln1_beta[l] = calloc(config->embed_dim, sizeof(float));
        if (!state->d_ln1_beta[l])
        {
            printf("Failed to allocate d_ln1_beta[%d]\n", l);
            exit(1);
        }

        state->d_ln2_gamma[l] = calloc(config->embed_dim, sizeof(float));
        if (!state->d_ln2_gamma[l])
        {
            printf("Failed to allocate d_ln2_gamma[%d]\n", l);
            exit(1);
        }

        state->d_ln2_beta[l] = calloc(config->embed_dim, sizeof(float));
        if (!state->d_ln2_beta[l])
        {
            printf("Failed to allocate d_ln2_beta[%d]\n", l);
            exit(1);
        }

        state->d_gating_w[l] = calloc(config->embed_dim * config->num_experts, sizeof(float));
        if (!state->d_gating_w[l])
        {
            printf("Failed to allocate d_gating_w[%d]\n", l);
            exit(1);
        }

        state->d_gating_b[l] = calloc(config->num_experts, sizeof(float));
        if (!state->d_gating_b[l])
        {
            printf("Failed to allocate d_gating_b[%d]\n", l);
            exit(1);
        }

        state->d_expert_w1[l] = malloc(config->num_experts * sizeof(float *));
        if (!state->d_expert_w1[l])
        {
            printf("Failed to allocate d_expert_w1[%d]\n", l);
            exit(1);
        }

        state->d_expert_b1[l] = malloc(config->num_experts * sizeof(float *));
        if (!state->d_expert_b1[l])
        {
            printf("Failed to allocate d_expert_b1[%d]\n", l);
            exit(1);
        }

        state->d_expert_w2[l] = malloc(config->num_experts * sizeof(float *));
        if (!state->d_expert_w2[l])
        {
            printf("Failed to allocate d_expert_w2[%d]\n", l);
            exit(1);
        }

        state->d_expert_b2[l] = malloc(config->num_experts * sizeof(float *));
        if (!state->d_expert_b2[l])
        {
            printf("Failed to allocate d_expert_b2[%d]\n", l);
            exit(1);
        }

        state->d_ln1_out = calloc(config->seq_len * config->embed_dim, sizeof(float));
        if (!state->d_ln1_out)
        {
            printf("Failed to allocate d_ln1_out\n");
            exit(1);
        }

        state->d_ln2_out = calloc(config->seq_len * config->embed_dim, sizeof(float));
        if (!state->d_ln2_out)
        {
            printf("Failed to allocate d_ln2_out\n");
            exit(1);
        }

        state->d_attn_residual_out = calloc(config->seq_len * config->embed_dim, sizeof(float));
        if (!state->d_attn_residual_out)
        {
            printf("Failed to allocate d_attn_residual_out\n");
            exit(1);
        }

        state->d_attn_concat = calloc(config->seq_len * config->embed_dim, sizeof(float));
        if (!state->d_attn_concat)
        {
            printf("Failed to allocate d_attn_concat\n");
            exit(1);
        }

        state->d_attention_scores = calloc(config->seq_len * config->seq_len, sizeof(float));
        if (!state->d_attention_scores)
        {
            printf("Failed to allocate d_attention_scores\n");
            exit(1);
        }

        state->d_query = calloc(config->seq_len * (config->embed_dim / config->num_heads), sizeof(float));
        if (!state->d_query)
        {
            printf("Failed to allocate d_query\n");
            exit(1);
        }

        state->d_key = calloc(config->seq_len * (config->embed_dim / config->num_heads), sizeof(float));
        if (!state->d_key)
        {
            printf("Failed to allocate d_key\n");
            exit(1);
        }

        state->d_value = calloc(config->seq_len * (config->embed_dim / config->num_heads), sizeof(float));
        if (!state->d_value)
        {
            printf("Failed to allocate d_value\n");
            exit(1);
        }

        state->d_softmax_scores = calloc(config->seq_len * config->seq_len, sizeof(float));
        if (!state->d_softmax_scores)
        {
            printf("Failed to allocate d_softmax_scores\n");
            exit(1);
        }

        for (int e = 0; e < config->num_experts; e++)
        {
            state->d_expert_w1[l][e] = calloc(config->embed_dim * config->hidden_dim, sizeof(float));
            if (!state->d_expert_w1[l][e])
            {
                printf("Failed to allocate d_expert_w1[%d][%d]\n", l, e);
                exit(1);
            }

            state->d_expert_b1[l][e] = calloc(config->hidden_dim, sizeof(float));
            if (!state->d_expert_b1[l][e])
            {
                printf("Failed to allocate d_expert_b1[%d][%d]\n", l, e);
                exit(1);
            }

            state->d_expert_w2[l][e] = calloc(config->hidden_dim * config->embed_dim, sizeof(float));
            if (!state->d_expert_w2[l][e])
            {
                printf("Failed to allocate d_expert_w2[%d][%d]\n", l, e);
                exit(1);
            }

            state->d_expert_b2[l][e] = calloc(config->embed_dim, sizeof(float));
            if (!state->d_expert_b2[l][e])
            {
                printf("Failed to allocate d_expert_b2[%d][%d]\n", l, e);
                exit(1);
            }
        }
    }
}

void update_weights(GPT2_MoE_Model *model, RunState *state, Optimizer *opt, float learning_rate)
{
    Config *config = &model->config;
    opt->step++;

    float bias_correction1 = 1.0f - pow(opt->beta1, opt->step);
    float bias_correction2 = 1.0f - pow(opt->beta2, opt->step);
    float corrected_lr = learning_rate * sqrt(bias_correction2) / bias_correction1;

    // Update token embeddings
    for (int i = 0; i < config->vocab_size * config->embed_dim; i++)
    {
        float grad = state->d_token_embeddings[i];
        opt->m_token_embeddings[i] = opt->beta1 * opt->m_token_embeddings[i] + (1.0f - opt->beta1) * grad;
        opt->v_token_embeddings[i] = opt->beta2 * opt->v_token_embeddings[i] + (1.0f - opt->beta2) * grad * grad;
        model->token_embeddings[i] -= corrected_lr * opt->m_token_embeddings[i] / (sqrt(opt->v_token_embeddings[i]) + opt->eps);
    }

    // Update positional embeddings
    for (int i = 0; i < config->seq_len * config->embed_dim; i++)
    {
        float grad = state->d_pos_embeddings[i];
        opt->m_pos_embeddings[i] = opt->beta1 * opt->m_pos_embeddings[i] + (1.0f - opt->beta1) * grad;
        opt->v_pos_embeddings[i] = opt->beta2 * opt->v_pos_embeddings[i] + (1.0f - opt->beta2) * grad * grad;
        model->pos_embeddings[i] -= corrected_lr * opt->m_pos_embeddings[i] / (sqrt(opt->v_pos_embeddings[i]) + opt->eps);
    }

    // Update final layer norm
    for (int i = 0; i < config->embed_dim; i++)
    {
        float grad_gamma = state->d_final_ln_gamma[i];
        float grad_beta = state->d_final_ln_beta[i];

        opt->m_final_ln_gamma[i] = opt->beta1 * opt->m_final_ln_gamma[i] + (1.0f - opt->beta1) * grad_gamma;
        opt->v_final_ln_gamma[i] = opt->beta2 * opt->v_final_ln_gamma[i] + (1.0f - opt->beta2) * grad_gamma * grad_gamma;
        model->final_ln_gamma[i] -= corrected_lr * opt->m_final_ln_gamma[i] / (sqrt(opt->v_final_ln_gamma[i]) + opt->eps);

        opt->m_final_ln_beta[i] = opt->beta1 * opt->m_final_ln_beta[i] + (1.0f - opt->beta1) * grad_beta;
        opt->v_final_ln_beta[i] = opt->beta2 * opt->v_final_ln_beta[i] + (1.0f - opt->beta2) * grad_beta * grad_beta;
        model->final_ln_beta[i] -= corrected_lr * opt->m_final_ln_beta[i] / (sqrt(opt->v_final_ln_beta[i]) + opt->eps);
    }

    // Update layer parameters
    for (int l = 0; l < config->num_layers; l++)
    {
        TransformerBlock *layer = &model->layers[l];

        // Update attention weights
        for (int i = 0; i < config->embed_dim * 3 * config->embed_dim; i++)
        {
            float grad = state->d_attn_qkv_w[l][i];
            opt->m_attn_qkv_w[l][i] = opt->beta1 * opt->m_attn_qkv_w[l][i] + (1.0f - opt->beta1) * grad;
            opt->v_attn_qkv_w[l][i] = opt->beta2 * opt->v_attn_qkv_w[l][i] + (1.0f - opt->beta2) * grad * grad;
            layer->attn_qkv_w[i] -= corrected_lr * opt->m_attn_qkv_w[l][i] / (sqrt(opt->v_attn_qkv_w[l][i]) + opt->eps);
        }

        for (int i = 0; i < 3 * config->embed_dim; i++)
        {
            float grad = state->d_attn_qkv_b[l][i];
            opt->m_attn_qkv_b[l][i] = opt->beta1 * opt->m_attn_qkv_b[l][i] + (1.0f - opt->beta1) * grad;
            opt->v_attn_qkv_b[l][i] = opt->beta2 * opt->v_attn_qkv_b[l][i] + (1.0f - opt->beta2) * grad * grad;
            layer->attn_qkv_b[i] -= corrected_lr * opt->m_attn_qkv_b[l][i] / (sqrt(opt->v_attn_qkv_b[l][i]) + opt->eps);
        }

        // Update MoE gating
        for (int i = 0; i < config->embed_dim * config->num_experts; i++)
        {
            float grad = state->d_gating_w[l][i];
            opt->m_gating_w[l][i] = opt->beta1 * opt->m_gating_w[l][i] + (1.0f - opt->beta1) * grad;
            opt->v_gating_w[l][i] = opt->beta2 * opt->v_gating_w[l][i] + (1.0f - opt->beta2) * grad * grad;
            layer->moe_layer.gating_w[i] -= corrected_lr * opt->m_gating_w[l][i] / (sqrt(opt->v_gating_w[l][i]) + opt->eps);
        }

        // Update experts
        for (int e = 0; e < config->num_experts; e++)
        {
            Expert *expert = &layer->moe_layer.experts[e];

            for (int i = 0; i < config->embed_dim * config->hidden_dim; i++)
            {
                float grad = state->d_expert_w1[l][e][i];
                opt->m_expert_w1[l][e][i] = opt->beta1 * opt->m_expert_w1[l][e][i] + (1.0f - opt->beta1) * grad;
                opt->v_expert_w1[l][e][i] = opt->beta2 * opt->v_expert_w1[l][e][i] + (1.0f - opt->beta2) * grad * grad;
                expert->w1[i] -= corrected_lr * opt->m_expert_w1[l][e][i] / (sqrt(opt->v_expert_w1[l][e][i]) + opt->eps);
            }

            for (int i = 0; i < config->hidden_dim * config->embed_dim; i++)
            {
                float grad = state->d_expert_w2[l][e][i];
                opt->m_expert_w2[l][e][i] = opt->beta1 * opt->m_expert_w2[l][e][i] + (1.0f - opt->beta1) * grad;
                opt->v_expert_w2[l][e][i] = opt->beta2 * opt->v_expert_w2[l][e][i] + (1.0f - opt->beta2) * grad * grad;
                expert->w2[i] -= corrected_lr * opt->m_expert_w2[l][e][i] / (sqrt(opt->v_expert_w2[l][e][i]) + opt->eps);
            }
        }
    }
}

// ============================================================================
// ## 4. FORWARD PASS IMPLEMENTATION
// ============================================================================

void multi_head_attention(float *out, float *x, TransformerBlock *layer, Config *config, RunState *state, int layer_idx)
{
    if (DEBUG)
        printf("  Multi-head attention\n");

    int seq_len = config->seq_len;
    int embed_dim = config->embed_dim;
    int num_heads = config->num_heads;
    int head_dim = embed_dim / num_heads;

    float *qkv = state->qkv_buffer;

    // Compute QKV for all positions
    for (int t = 0; t < seq_len; t++)
    {
        for (int d = 0; d < 3 * embed_dim; d++)
        {
            qkv[t * 3 * embed_dim + d] = layer->attn_qkv_b[d];
            for (int e = 0; e < embed_dim; e++)
            {
                qkv[t * 3 * embed_dim + d] += x[t * embed_dim + e] * layer->attn_qkv_w[e * 3 * embed_dim + d];
            }
        }
    }

    // Initialize output
    memset(out, 0, seq_len * embed_dim * sizeof(float));

    // Process each attention head
    for (int h = 0; h < num_heads; h++)
    {
        // Compute attention for this head
        for (int i = 0; i < seq_len; i++)
        {
            // Compute attention scores for position i
            for (int j = 0; j < seq_len; j++)
            {
                if (j > i)
                {
                    // Causal masking
                    state->attention_scores_buffer[i * seq_len + j] = -1e9f;
                }
                else
                {
                    float score = 0.0f;
                    // Compute Q*K^T for this head
                    for (int d = 0; d < head_dim; d++)
                    {
                        float qi = qkv[i * 3 * embed_dim + h * head_dim + d];
                        float kj = qkv[j * 3 * embed_dim + embed_dim + h * head_dim + d];
                        score += qi * kj;
                    }
                    score /= sqrtf((float)head_dim);
                    state->attention_scores_buffer[i * seq_len + j] = score;
                }
            }

            // Apply softmax to row i
            float *row = &state->attention_scores_buffer[i * seq_len];
            softmax(row, seq_len);

            // Save probabilities for backpropagation
            int prob_offset = (layer_idx * seq_len * seq_len) + (i * seq_len);
            for (int j = 0; j < seq_len; j++)
            {
                state->attention_probs[prob_offset + j] = row[j];
            }

            // Compute weighted sum for this head
            for (int d = 0; d < head_dim; d++)
            {
                float sum = 0.0f;
                for (int j = 0; j <= i; j++)
                {
                    float vj = qkv[j * 3 * embed_dim + 2 * embed_dim + h * head_dim + d];
                    sum += row[j] * vj;
                }
                state->temp_buffer[i * embed_dim + h * head_dim + d] = sum;
            }
        }
    }

    memcpy(state->attn_concat, state->temp_buffer, seq_len * embed_dim * sizeof(float));

    // Output projection
    for (int t = 0; t < seq_len; t++)
    {
        for (int d = 0; d < embed_dim; d++)
        {
            out[t * embed_dim + d] = layer->attn_proj_b[d];
            for (int e = 0; e < embed_dim; e++)
            {
                out[t * embed_dim + d] += state->temp_buffer[t * embed_dim + e] * layer->attn_proj_w[e * embed_dim + d];
            }
        }
    }

    if (DEBUG)
        printf("  Multi-head attention complete\n");
}

void moe_forward(float *out, float *x, MoELayer *moe, RunState *state, Config *config, int layer_idx)
{
    if (DEBUG)
        printf("  MoE forward: layer %d\n", layer_idx);

    int seq_len = config->seq_len;
    int embed_dim = config->embed_dim;
    int num_experts = config->num_experts;
    int top_k = config->top_k;
    int hidden_dim = config->hidden_dim;

    memset(out, 0, seq_len * embed_dim * sizeof(float));

    for (int t = 0; t < seq_len; t++)
    {
        if (DEBUG && t == 0)
            printf("    Processing token %d\n", t);

        float *token_x = &x[t * embed_dim];
        float *token_out = &out[t * embed_dim];
        float *gating_scores = state->gating_scores_buffer;

        // Compute gating scores
        for (int e = 0; e < num_experts; e++)
        {
            gating_scores[e] = moe->gating_b[e];
            for (int d = 0; d < embed_dim; d++)
            {
                gating_scores[e] += token_x[d] * moe->gating_w[e * embed_dim + d];
            }
            // Clamp extreme values
            if (gating_scores[e] > 50.0f)
                gating_scores[e] = 50.0f;
            if (gating_scores[e] < -50.0f)
                gating_scores[e] = -50.0f;
        }

        // Store gating logits
        int gating_store_offset = (layer_idx * seq_len + t) * num_experts;
        if (state->gating_logits)
        {
            memcpy(&state->gating_logits[gating_store_offset], gating_scores, num_experts * sizeof(float));
        }

        // Apply softmax
        softmax(gating_scores, num_experts);

        // Get top-k experts
        int indices[8]; // Max 8 experts supported
        float weights[8];
        if (top_k > 8)
        {
            printf("Error: top_k > 8 not supported\n");
            exit(1);
        }

        topk_indices(indices, weights, gating_scores, num_experts, top_k);

        // Store the chosen expert indices in the global state for analysis
        int expert_indices_offset = (layer_idx * seq_len + t) * top_k;
        memcpy(&state->expert_indices[expert_indices_offset], indices, top_k * sizeof(int));

        // Also store the weights
        int expert_weights_offset = (layer_idx * seq_len + t) * top_k;
        memcpy(&state->expert_weights[expert_weights_offset], weights, top_k * sizeof(float));

        // Renormalize weights
        float weight_sum = 0.0f;
        for (int k = 0; k < top_k; k++)
            weight_sum += weights[k];
        if (weight_sum > 1e-10f)
        {
            for (int k = 0; k < top_k; k++)
                weights[k] /= weight_sum;
        }
        else
        {
            for (int k = 0; k < top_k; k++)
                weights[k] = 1.0f / top_k;
        }

        // Process selected experts
        for (int k = 0; k < top_k; k++)
        {
            int expert_idx = indices[k];
            if (expert_idx < 0 || expert_idx >= num_experts)
                continue;

            Expert *expert = &moe->experts[expert_idx];
            float *hidden = state->hidden_buffer;
            float *expert_out = state->expert_out_buffer;

            // First layer: hidden = ReLU(W1 * x + b1)
            for (int h = 0; h < hidden_dim; h++)
            {
                hidden[h] = expert->b1[h];
                for (int d = 0; d < embed_dim; d++)
                {
                    hidden[h] += token_x[d] * expert->w1[d * hidden_dim + h];
                }
                if (hidden[h] < 0)
                    hidden[h] = 0; // ReLU
            }

            // FIX: Store expert hidden states for backpropagation
            int hidden_offset = layer_idx * config->num_experts * seq_len * hidden_dim +
                                expert_idx * seq_len * hidden_dim +
                                t * hidden_dim;
            memcpy(&state->expert_hiddens[hidden_offset], hidden, hidden_dim * sizeof(float));

            // Second layer: out = W2 * hidden + b2
            for (int d = 0; d < embed_dim; d++)
            {
                expert_out[d] = expert->b2[d];
                for (int h = 0; h < hidden_dim; h++)
                {
                    expert_out[d] += hidden[h] * expert->w2[h * embed_dim + d];
                }
            }

            // Store expert outputs for backpropagation
            int expert_out_offset = layer_idx * config->num_experts * seq_len * embed_dim +
                                    expert_idx * seq_len * embed_dim +
                                    t * embed_dim;
            memcpy(&state->expert_outputs[expert_out_offset], expert_out, embed_dim * sizeof(float));

            // Accumulate weighted output
            for (int d = 0; d < embed_dim; d++)
            {
                token_out[d] += weights[k] * expert_out[d];
            }
        }
    }

    if (DEBUG)
        printf("  MoE forward complete\n");
}

void forward_pass(GPT2_MoE_Model *model, RunState *state, int *inputs)
{
    if (DEBUG)
        printf("Forward pass starting...\n");

    Config *config = &model->config;
    int seq_len = config->seq_len;
    int embed_dim = config->embed_dim;

    // 1. Embedding
    if (DEBUG)
        printf("1. Computing embeddings\n");
    for (int t = 0; t < seq_len; t++)
    {
        int token_id = inputs[t];
        if (token_id < 0 || token_id >= config->vocab_size)
        {
            memset(&state->embedding_out[t * embed_dim], 0, embed_dim * sizeof(float));
            continue;
        }
        for (int d = 0; d < embed_dim; d++)
        {
            state->embedding_out[t * embed_dim + d] =
                model->token_embeddings[token_id * embed_dim + d] +
                model->pos_embeddings[t * embed_dim + d];
        }
    }

    // Current layer input
    float *current_input = state->embedding_out;

    // 2. Transformer blocks
    for (int l = 0; l < config->num_layers; l++)
    {
        if (DEBUG)
            printf("2. Processing layer %d\n", l);

        TransformerBlock *layer = &model->layers[l];
        float *layer_output = &state->layer_outputs[l * seq_len * embed_dim];

        // Layer norm 1 (per token)
        for (int t = 0; t < seq_len; t++)
        {
            float *x = &current_input[t * embed_dim];
            float *out = &state->ln1_outputs[l * seq_len * embed_dim + t * embed_dim];
            layer_norm(out, x, layer->ln1_gamma, layer->ln1_beta, embed_dim);
        }

        // Attention with layer index for saving probabilities
        multi_head_attention(state->attn_out,
                             &state->ln1_outputs[l * seq_len * embed_dim],
                             layer, config, state, l);

        // Residual 1
        for (int i = 0; i < seq_len * embed_dim; i++)
        {
            state->attn_residual[l * seq_len * embed_dim + i] =
                current_input[i] + state->attn_out[i];
        }

        // Layer norm 2 (per token)
        for (int t = 0; t < seq_len; t++)
        {
            float *x = &state->attn_residual[l * seq_len * embed_dim + t * embed_dim];
            float *out = &state->ln2_outputs[l * seq_len * embed_dim + t * embed_dim];
            layer_norm(out, x, layer->ln2_gamma, layer->ln2_beta, embed_dim);
        }

        // MoE
        moe_forward(&state->moe_outputs[l * seq_len * embed_dim],
                    &state->ln2_outputs[l * seq_len * embed_dim],
                    &layer->moe_layer, state, config, l);

        // Residual 2
        for (int i = 0; i < seq_len * embed_dim; i++)
        {
            layer_output[i] = state->attn_residual[l * seq_len * embed_dim + i] +
                              state->moe_outputs[l * seq_len * embed_dim + i];
        }

        // Update current_input for next layer
        current_input = layer_output;
    }

    // 3. Final layer norm (per token)
    if (DEBUG)
        printf("3. Final layer norm\n");
    for (int t = 0; t < seq_len; t++)
    {
        float *x = &current_input[t * embed_dim];
        float *out = &state->temp_buffer2[t * embed_dim];
        layer_norm(out, x, model->final_ln_gamma, model->final_ln_beta, embed_dim);
    }

    // 4. Output projection
    if (DEBUG)
        printf("4. Output projection\n");
    for (int t = 0; t < seq_len; t++)
    {
        for (int v = 0; v < config->vocab_size; v++)
        {
            state->logits[t * config->vocab_size + v] = 0.0f;
            for (int d = 0; d < embed_dim; d++)
            {
                state->logits[t * config->vocab_size + v] +=
                    state->temp_buffer2[t * embed_dim + d] * model->token_embeddings[v * embed_dim + d];
            }
        }
    }

    if (DEBUG)
        printf("Forward pass complete!\n");
}

void backward_pass(GPT2_MoE_Model *model, RunState *state, int *inputs, int *targets, float total_loss)
{
    Config *config = &model->config;
    int seq_len = config->seq_len;
    int embed_dim = config->embed_dim;
    int vocab_size = config->vocab_size;
    int num_heads = config->num_heads;
    int head_dim = embed_dim / num_heads;

    // Zero out all gradients first
    memset(state->d_token_embeddings, 0, vocab_size * embed_dim * sizeof(float));
    memset(state->d_pos_embeddings, 0, seq_len * embed_dim * sizeof(float));
    memset(state->d_final_ln_gamma, 0, embed_dim * sizeof(float));
    memset(state->d_final_ln_beta, 0, embed_dim * sizeof(float));

    for (int l = 0; l < config->num_layers; l++)
    {
        memset(state->d_attn_qkv_w[l], 0, embed_dim * 3 * embed_dim * sizeof(float));
        memset(state->d_attn_qkv_b[l], 0, 3 * embed_dim * sizeof(float));
        memset(state->d_attn_proj_w[l], 0, embed_dim * embed_dim * sizeof(float));
        memset(state->d_attn_proj_b[l], 0, embed_dim * sizeof(float));
        memset(state->d_ln1_gamma[l], 0, embed_dim * sizeof(float));
        memset(state->d_ln1_beta[l], 0, embed_dim * sizeof(float));
        memset(state->d_ln2_gamma[l], 0, embed_dim * sizeof(float));
        memset(state->d_ln2_beta[l], 0, embed_dim * sizeof(float));
        memset(state->d_gating_w[l], 0, embed_dim * config->num_experts * sizeof(float));
        memset(state->d_gating_b[l], 0, config->num_experts * sizeof(float));

        for (int e = 0; e < config->num_experts; e++)
        {
            memset(state->d_expert_w1[l][e], 0, embed_dim * config->hidden_dim * sizeof(float));
            memset(state->d_expert_b1[l][e], 0, config->hidden_dim * sizeof(float));
            memset(state->d_expert_w2[l][e], 0, config->hidden_dim * embed_dim * sizeof(float));
            memset(state->d_expert_b2[l][e], 0, embed_dim * sizeof(float));
        }
    }

    // 1. Cross-entropy loss gradient (d_logits)
    float *d_logits = state->d_temp_buffer; // Use pre-allocated buffer
    for (int t = 0; t < seq_len; t++)
    {
        int target = targets[t];
        if (target < 0 || target >= vocab_size)
            continue;

        float *token_logits = &state->logits[t * vocab_size];
        float *d_token_logits = &d_logits[t * vocab_size];

        // Compute softmax probabilities
        float max_logit = token_logits[0];
        for (int v = 1; v < vocab_size; v++)
        {
            if (token_logits[v] > max_logit)
                max_logit = token_logits[v];
        }
        float sum = 0.0f;
        for (int v = 0; v < vocab_size; v++)
        {
            sum += expf(token_logits[v] - max_logit);
        }

        // Softmax gradient: dL/dx = (softmax(x) - one_hot(target)) / N
        for (int v = 0; v < vocab_size; v++)
        {
            float prob = expf(token_logits[v] - max_logit) / sum;
            d_token_logits[v] = prob;
            if (v == target)
            {
                d_token_logits[v] -= 1.0f;
            }
            d_token_logits[v] /= seq_len;
        }
    }

    // 2. Gradient through output projection (tied weights)
    float *final_ln_output = state->temp_buffer2;
    for (int t = 0; t < seq_len; t++)
    {
        for (int d = 0; d < embed_dim; d++)
        {
            state->d_temp_buffer[t * embed_dim + d] = 0.0f;
            for (int v = 0; v < vocab_size; v++)
            {
                // Gradient w.r.t. final layer norm output
                state->d_temp_buffer[t * embed_dim + d] +=
                    d_logits[t * vocab_size + v] * model->token_embeddings[v * embed_dim + d];

                // Gradient w.r.t. token embeddings (tied weights)
                state->d_token_embeddings[v * embed_dim + d] +=
                    d_logits[t * vocab_size + v] * final_ln_output[t * embed_dim + d];
            }
        }
    }

    // 3. Final layer norm backward
    float *d_final_ln_output = state->d_temp_buffer;
    float *final_transformer_output = (config->num_layers > 0) ? &state->layer_outputs[(config->num_layers - 1) * seq_len * embed_dim] : state->embedding_out;

    memset(state->d_temp_buffer2, 0, seq_len * embed_dim * sizeof(float));

    for (int t = 0; t < seq_len; t++)
    {
        float *x = &final_transformer_output[t * embed_dim];
        float *dy = &d_final_ln_output[t * embed_dim];
        float *dx = &state->d_temp_buffer2[t * embed_dim];

        float mean = 0.0f, var = 0.0f;
        for (int d = 0; d < embed_dim; d++)
            mean += x[d];
        mean /= embed_dim;
        for (int d = 0; d < embed_dim; d++)
            var += (x[d] - mean) * (x[d] - mean);
        var /= embed_dim;
        float rstd = 1.0f / sqrtf(var + 1e-5f);

        for (int d = 0; d < embed_dim; d++)
        {
            state->d_final_ln_gamma[d] += dy[d] * (x[d] - mean) * rstd;
            state->d_final_ln_beta[d] += dy[d];
        }

        float sum_dy = 0.0f, sum_dy_xmu = 0.0f;
        for (int d = 0; d < embed_dim; d++)
        {
            sum_dy += dy[d];
            sum_dy_xmu += dy[d] * (x[d] - mean);
        }

        for (int d = 0; d < embed_dim; d++)
        {
            dx[d] = model->final_ln_gamma[d] * rstd * (dy[d] - sum_dy / embed_dim - (x[d] - mean) * sum_dy_xmu * rstd * rstd / embed_dim);
        }
    }

    // 4. Backward through transformer layers
    float *d_residual = state->d_temp_buffer2;

    for (int l = config->num_layers - 1; l >= 0; l--)
    {
        TransformerBlock *layer = &model->layers[l];
        float *layer_input = (l == 0) ? state->embedding_out : &state->layer_outputs[(l - 1) * seq_len * embed_dim];
        float *ln1_output = &state->ln1_outputs[l * seq_len * embed_dim];
        float *attn_residual = &state->attn_residual[l * seq_len * embed_dim];
        float *ln2_output = &state->ln2_outputs[l * seq_len * embed_dim];
        float *attn_output = &state->attn_outputs[l * seq_len * embed_dim];
        float *moe_output = &state->moe_outputs[l * seq_len * embed_dim];

        // 4a. Second residual connection backward
        float *d_moe_out = state->d_moe_out;            // Use pre-allocated buffer
        float *d_attn_residual_out = state->d_attn_out; // Use pre-allocated buffer
        memcpy(d_moe_out, d_residual, seq_len * embed_dim * sizeof(float));
        memcpy(d_attn_residual_out, d_residual, seq_len * embed_dim * sizeof(float));

        // 4b. Backward through MoE Layer
        float *d_ln2_out = state->d_temp_buffer; // Use pre-allocated buffer
        memset(d_ln2_out, 0, seq_len * embed_dim * sizeof(float));

        for (int t = 0; t < seq_len; t++)
        {
            // Get expert assignments using correct layer-aware indexing
            int gating_offset = l * seq_len * config->num_experts + t * config->num_experts;
            float *gating_scores = &state->gating_scores[gating_offset];

            int expert_idx_offset = l * seq_len * config->top_k + t * config->top_k;
            int *expert_indices = &state->expert_indices[expert_idx_offset];

            int expert_weight_offset = l * seq_len * config->top_k + t * config->top_k;
            float *expert_weights = &state->expert_weights[expert_weight_offset];

            float *ln2_token_out = &ln2_output[t * embed_dim];

            // Gradient through expert mixing
            for (int k = 0; k < config->top_k; k++)
            {
                int expert_idx = expert_indices[k];
                if (expert_idx < 0 || expert_idx >= config->num_experts)
                    continue;

                float weight = expert_weights[k];

                // Get expert outputs with correct layer-aware indexing
                int expert_out_offset = l * config->num_experts * seq_len * embed_dim +
                                        expert_idx * seq_len * embed_dim + t * embed_dim;
                float *expert_out = &state->expert_outputs[expert_out_offset];

                // Gradient w.r.t. expert output
                for (int d = 0; d < embed_dim; d++)
                {
                    float d_expert_out = d_moe_out[t * embed_dim + d] * weight;

                    // Get expert hidden states with correct indexing
                    int hidden_offset = l * config->num_experts * seq_len * config->hidden_dim +
                                        expert_idx * seq_len * config->hidden_dim + t * config->hidden_dim;
                    float *expert_hidden = &state->expert_hiddens[hidden_offset];

                    // Gradient w.r.t. expert bias 2
                    state->d_expert_b2[l][expert_idx][d] += d_expert_out;

                    // Gradient w.r.t. expert weights 2 and hidden activations
                    for (int h = 0; h < config->hidden_dim; h++)
                    {
                        state->d_expert_w2[l][expert_idx][h * embed_dim + d] += d_expert_out * expert_hidden[h];

                        // Gradient w.r.t. hidden (before ReLU)
                        float d_hidden = d_expert_out * layer->moe_layer.experts[expert_idx].w2[h * embed_dim + d];

                        // ReLU gradient
                        if (expert_hidden[h] > 0.0f)
                        {
                            // Gradient w.r.t. expert bias 1
                            state->d_expert_b1[l][expert_idx][h] += d_hidden;

                            // Gradient w.r.t. expert weights 1 and input
                            for (int d2 = 0; d2 < embed_dim; d2++)
                            {
                                state->d_expert_w1[l][expert_idx][d2 * config->hidden_dim + h] += d_hidden * ln2_token_out[d2];
                                d_ln2_out[t * embed_dim + d2] += d_hidden * layer->moe_layer.experts[expert_idx].w1[d2 * config->hidden_dim + h];
                            }
                        }
                    }
                }

                // Gradient w.r.t. gating weights (FIXED: Removed division by top_k)
                float d_weight = 0.0f;
                for (int d_out = 0; d_out < embed_dim; d_out++)
                {
                    d_weight += d_moe_out[t * embed_dim + d_out] * expert_out[d_out];
                }

                // Softmax gradient for gating network
                for (int e = 0; e < config->num_experts; e++)
                {
                    float grad_contribution = d_weight * gating_scores[e];
                    if (e == expert_idx)
                        grad_contribution -= d_weight;

                    state->d_gating_b[l][e] += grad_contribution; // FIXED: Removed division by top_k

                    for (int d2 = 0; d2 < embed_dim; d2++)
                    {
                        state->d_gating_w[l][e * embed_dim + d2] += grad_contribution * ln2_token_out[d2];                  // FIXED: Removed division by top_k
                        d_ln2_out[t * embed_dim + d2] += grad_contribution * layer->moe_layer.gating_w[e * embed_dim + d2]; // FIXED: Removed division by top_k
                    }
                }
            }
        }

        // 4c. Backward through second LayerNorm
        float *d_attn_residual_out2 = state->d_temp_buffer2; // Use pre-allocated buffer
        memset(d_attn_residual_out2, 0, seq_len * embed_dim * sizeof(float));

        for (int t = 0; t < seq_len; t++)
        {
            float *x = &attn_residual[t * embed_dim];
            float *dy = &d_ln2_out[t * embed_dim];
            float *dx = &d_attn_residual_out2[t * embed_dim];

            float mean = 0.0f, var = 0.0f;
            for (int d = 0; d < embed_dim; d++)
                mean += x[d];
            mean /= embed_dim;
            for (int d = 0; d < embed_dim; d++)
                var += (x[d] - mean) * (x[d] - mean);
            var /= embed_dim;
            float rstd = 1.0f / sqrtf(var + 1e-5f);

            for (int d = 0; d < embed_dim; d++)
            {
                state->d_ln2_gamma[l][d] += dy[d] * (x[d] - mean) * rstd;
                state->d_ln2_beta[l][d] += dy[d];
            }

            float sum_dy = 0.0f, sum_dy_xmu = 0.0f;
            for (int d = 0; d < embed_dim; d++)
            {
                sum_dy += dy[d];
                sum_dy_xmu += dy[d] * (x[d] - mean);
            }

            for (int d = 0; d < embed_dim; d++)
            {
                dx[d] = layer->ln2_gamma[d] * rstd * (dy[d] - sum_dy / embed_dim - (x[d] - mean) * sum_dy_xmu * rstd * rstd / embed_dim);
            }
        }

        // Add gradients from residual connections
        for (int i = 0; i < seq_len * embed_dim; i++)
        {
            d_attn_residual_out2[i] += d_attn_residual_out[i];
        }

        // 4d. First residual connection backward
        float *d_attn_out = state->d_attn_out;   // Use pre-allocated buffer
        float *d_ln1_out = state->d_temp_buffer; // Use pre-allocated buffer
        memcpy(d_attn_out, d_attn_residual_out2, seq_len * embed_dim * sizeof(float));
        memcpy(d_ln1_out, d_attn_residual_out2, seq_len * embed_dim * sizeof(float));

        // 4e. Backward through Multi-Head Attention (FULL IMPLEMENTATION)
        float *d_ln1_out2 = state->d_temp_buffer2; // Use pre-allocated buffer
        memset(d_ln1_out2, 0, seq_len * embed_dim * sizeof(float));

        // Get attention intermediate buffers
        float *qkv_buffer = state->qkv_buffer;
        float *attention_probs = state->attention_scores_buffer; // This now contains probabilities
        float *attn_concat = state->attn_concat;                 // This should store concatenated head outputs

        // Backward through output projection
        float *d_attn_concat = state->d_temp_buffer; // Use pre-allocated buffer
        memset(d_attn_concat, 0, seq_len * embed_dim * sizeof(float));

        // Gradient w.r.t. output projection weights and bias
        for (int t = 0; t < seq_len; t++)
        {
            for (int d = 0; d < embed_dim; d++)
            {
                // Gradient w.r.t. output projection bias
                state->d_attn_proj_b[l][d] += d_attn_out[t * embed_dim + d];

                // Gradient w.r.t. output projection weights
                for (int e = 0; e < embed_dim; e++)
                {
                    state->d_attn_proj_w[l][e * embed_dim + d] += d_attn_out[t * embed_dim + d] * attn_concat[l * seq_len * embed_dim + t * embed_dim + e];
                    d_attn_concat[t * embed_dim + e] += d_attn_out[t * embed_dim + d] * layer->attn_proj_w[e * embed_dim + d];
                }
            }
        }

        // Backward through multi-head attention mechanism
        for (int h = 0; h < num_heads; h++)
        {
            int head_offset = h * head_dim;

            // Backward through attention output (weighted sum of values)
            float *d_attention_scores = state->d_temp_buffer2; // Use pre-allocated buffer
            float *d_value = state->d_temp_buffer;             // Use pre-allocated buffer
            memset(d_attention_scores, 0, seq_len * seq_len * sizeof(float));
            memset(d_value, 0, seq_len * head_dim * sizeof(float));

            for (int i = 0; i < seq_len; i++)
            {
                for (int j = 0; j <= i; j++) // Causal mask
                {
                    // Gradient w.r.t. attention scores (FIXED: Using probabilities instead of logits)
                    for (int d = 0; d < head_dim; d++)
                    {
                        int concat_idx = i * embed_dim + head_offset + d;
                        d_attention_scores[i * seq_len + j] += d_attn_concat[concat_idx] *
                                                               qkv_buffer[j * 3 * embed_dim + 2 * embed_dim + head_offset + d];

                        // Gradient w.r.t. values
                        d_value[j * head_dim + d] += d_attn_concat[concat_idx] * attention_probs[i * seq_len + j];
                    }
                }
            }

            // Backward through softmax (FIXED: Correct softmax gradient implementation)
            float *d_softmax_scores = state->d_softmax_scores; // Use pre-allocated buffer
            memset(d_softmax_scores, 0, seq_len * seq_len * sizeof(float));

            // Get the saved probabilities for this layer and head
            int prob_base_offset = l * seq_len * seq_len; // Layer offset

            for (int i = 0; i < seq_len; i++)
            {
                // Get probabilities for this query position
                float *probs = &state->attention_probs[prob_base_offset + i * seq_len];

                // Compute gradient of softmax: J[i,j] = p[i] * (delta[i,j] - p[j]) * grad_out[i]
                for (int j = 0; j <= i; j++) // Only process valid positions due to causal mask
                {
                    float p_i = probs[i];
                    float p_j = probs[j];

                    // Jacobian of softmax: J[i,j] = p[i] * (delta[i,j] - p[j])
                    float jacobian = (i == j) ? p_i * (1.0f - p_i) : -p_i * p_j;
                    d_softmax_scores[i * seq_len + j] = jacobian * d_attention_scores[i * seq_len + j];
                }
            }
            // Backward through attention score computation (Q*K^T)
            float *d_query = state->d_temp_buffer2; // Use pre-allocated buffer
            float *d_key = state->d_moe_out;        // Use pre-allocated buffer
            memset(d_query, 0, seq_len * head_dim * sizeof(float));
            memset(d_key, 0, seq_len * head_dim * sizeof(float));

            float scale = 1.0f / sqrtf((float)head_dim);

            for (int i = 0; i < seq_len; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    float d_score = d_softmax_scores[i * seq_len + j] * scale;

                    // Gradient w.r.t. query and key
                    for (int d = 0; d < head_dim; d++)
                    {
                        d_query[i * head_dim + d] += d_score * qkv_buffer[j * 3 * embed_dim + embed_dim + head_offset + d];
                        d_key[j * head_dim + d] += d_score * qkv_buffer[i * 3 * embed_dim + head_offset + d];
                    }
                }
            }

            // Backward through QKV projection
            for (int t = 0; t < seq_len; t++)
            {
                // Gradient w.r.t. QKV bias
                for (int d = 0; d < head_dim; d++)
                {
                    state->d_attn_qkv_b[l][head_offset + d] += d_query[t * head_dim + d];
                    state->d_attn_qkv_b[l][embed_dim + head_offset + d] += d_key[t * head_dim + d];
                    state->d_attn_qkv_b[l][2 * embed_dim + head_offset + d] += d_value[t * head_dim + d];
                }

                // Gradient w.r.t. QKV weights and input (ln1_output)
                for (int d = 0; d < head_dim; d++)
                {
                    for (int e = 0; e < embed_dim; e++)
                    {
                        // Q gradients
                        state->d_attn_qkv_w[l][e * 3 * embed_dim + head_offset + d] +=
                            d_query[t * head_dim + d] * ln1_output[t * embed_dim + e];
                        d_ln1_out2[t * embed_dim + e] +=
                            d_query[t * head_dim + d] * layer->attn_qkv_w[e * 3 * embed_dim + head_offset + d];

                        // K gradients
                        state->d_attn_qkv_w[l][e * 3 * embed_dim + embed_dim + head_offset + d] +=
                            d_key[t * head_dim + d] * ln1_output[t * embed_dim + e];
                        d_ln1_out2[t * embed_dim + e] +=
                            d_key[t * head_dim + d] * layer->attn_qkv_w[e * 3 * embed_dim + embed_dim + head_offset + d];

                        // V gradients
                        state->d_attn_qkv_w[l][e * 3 * embed_dim + 2 * embed_dim + head_offset + d] +=
                            d_value[t * head_dim + d] * ln1_output[t * embed_dim + e];
                        d_ln1_out2[t * embed_dim + e] +=
                            d_value[t * head_dim + d] * layer->attn_qkv_w[e * 3 * embed_dim + 2 * embed_dim + head_offset + d];
                    }
                }
            }
        }

        // Add attention input gradients
        for (int i = 0; i < seq_len * embed_dim; i++)
        {
            d_ln1_out2[i] += d_ln1_out[i];
        }

        // 4f. Backward through first LayerNorm
        float *d_layer_input = state->d_temp_buffer; // Use pre-allocated buffer
        memset(d_layer_input, 0, seq_len * embed_dim * sizeof(float));

        for (int t = 0; t < seq_len; t++)
        {
            float *x = &layer_input[t * embed_dim];
            float *dy = &d_ln1_out2[t * embed_dim];
            float *dx = &d_layer_input[t * embed_dim];

            float mean = 0.0f, var = 0.0f;
            for (int d = 0; d < embed_dim; d++)
                mean += x[d];
            mean /= embed_dim;
            for (int d = 0; d < embed_dim; d++)
                var += (x[d] - mean) * (x[d] - mean);
            var /= embed_dim;
            float rstd = 1.0f / sqrtf(var + 1e-5f);

            for (int d = 0; d < embed_dim; d++)
            {
                state->d_ln1_gamma[l][d] += dy[d] * (x[d] - mean) * rstd;
                state->d_ln1_beta[l][d] += dy[d];
            }

            float sum_dy = 0.0f, sum_dy_xmu = 0.0f;
            for (int d = 0; d < embed_dim; d++)
            {
                sum_dy += dy[d];
                sum_dy_xmu += dy[d] * (x[d] - mean);
            }

            for (int d = 0; d < embed_dim; d++)
            {
                dx[d] = layer->ln1_gamma[d] * rstd * (dy[d] - sum_dy / embed_dim - (x[d] - mean) * sum_dy_xmu * rstd * rstd / embed_dim);
            }
        }

        // Update gradient for next iteration or embeddings
        if (l == 0)
        {
            // 5. Embedding gradients for first layer (FIXED: Use inputs instead of targets)
            for (int t = 0; t < seq_len; t++)
            {
                int token_id = inputs[t]; // FIXED: Use actual input token ID
                if (token_id >= 0 && token_id < vocab_size)
                {
                    for (int d = 0; d < embed_dim; d++)
                    {
                        state->d_pos_embeddings[t * embed_dim + d] += d_layer_input[t * embed_dim + d];
                        state->d_token_embeddings[token_id * embed_dim + d] += d_layer_input[t * embed_dim + d];
                    }
                }
            }
        }
        else
        {
            d_residual = d_layer_input;
        }
    }
}
// ============================================================================
// ## 5. LOSS CALCULATION
// ============================================================================

float calculate_crossentropy_loss(float *logits, int *targets, Config *config)
{
    float loss = 0.0f;
    int seq_len = config->seq_len;
    int vocab_size = config->vocab_size;

    for (int t = 0; t < seq_len; t++)
    {
        int target = targets[t];
        if (target >= 0 && target < vocab_size)
        {
            float *token_logits = &logits[t * vocab_size];

            // Apply softmax and compute negative log likelihood
            float max_logit = token_logits[0];
            for (int v = 1; v < vocab_size; v++)
            {
                if (token_logits[v] > max_logit)
                    max_logit = token_logits[v];
            }

            float sum = 0.0f;
            for (int v = 0; v < vocab_size; v++)
            {
                sum += exp(token_logits[v] - max_logit);
            }

            loss -= (token_logits[target] - max_logit - log(sum));
        }
    }

    return loss / seq_len;
}

float calculate_moe_aux_loss(RunState *state, Config *config)
{
    int seq_len = config->seq_len;
    int num_experts = config->num_experts;

    // Calculate mean probability for each expert across all tokens
    float *expert_means = calloc(num_experts, sizeof(float));

    for (int t = 0; t < seq_len; t++)
    {
        float *gating_probs = &state->gating_logits[t * num_experts];
        // Apply softmax
        float max_logit = gating_probs[0];
        for (int e = 1; e < num_experts; e++)
        {
            if (gating_probs[e] > max_logit)
                max_logit = gating_probs[e];
        }
        float sum = 0.0f;
        for (int e = 0; e < num_experts; e++)
        {
            gating_probs[e] = exp(gating_probs[e] - max_logit);
            sum += gating_probs[e];
        }
        for (int e = 0; e < num_experts; e++)
        {
            gating_probs[e] /= sum;
            expert_means[e] += gating_probs[e];
        }
    }

    // Normalize by sequence length
    for (int e = 0; e < num_experts; e++)
    {
        expert_means[e] /= seq_len;
    }

    // Load balancing loss: coefficient of variation squared
    float mean_of_means = 0.0f;
    for (int e = 0; e < num_experts; e++)
    {
        mean_of_means += expert_means[e];
    }
    mean_of_means /= num_experts;

    float variance = 0.0f;
    for (int e = 0; e < num_experts; e++)
    {
        float diff = expert_means[e] - mean_of_means;
        variance += diff * diff;
    }

    free(expert_means);
    return num_experts * variance; // Scale by number of experts
}

// ============================================================================
// ## 6. SIMPLE TRAINING DATA LOADER
// ============================================================================

void load_tiny_shakespeare(Dataset *dataset)
{
    // Try to open the file
    FILE *file = fopen("tinyshakespeare.txt", "r");
    if (!file)
    {
        printf("Error: Could not open tinyshakespeare.txt\n");
        printf("Please download from: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt\n");
        exit(1);
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    dataset->size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Read entire file
    dataset->data = malloc(dataset->size + 1);
    fread(dataset->data, 1, dataset->size, file);
    dataset->data[dataset->size] = '\0';
    fclose(file);

    // Build character vocabulary
    int char_count[256] = {0};
    for (int i = 0; i < dataset->size; i++)
    {
        char_count[(unsigned char)dataset->data[i]]++;
    }

    dataset->vocab_size = 0;
    for (int i = 0; i < 256; i++)
    {
        if (char_count[i] > 0)
        {
            dataset->vocab[dataset->vocab_size] = malloc(2);
            dataset->vocab[dataset->vocab_size][0] = (char)i;
            dataset->vocab[dataset->vocab_size][1] = '\0';
            dataset->vocab_size++;
        }
    }

    // Tokenize
    dataset->num_tokens = dataset->size;
    dataset->tokens = malloc(dataset->num_tokens * sizeof(int));
    for (int i = 0; i < dataset->size; i++)
    {
        char c = dataset->data[i];
        for (int v = 0; v < dataset->vocab_size; v++)
        {
            if (dataset->vocab[v][0] == c)
            {
                dataset->tokens[i] = v;
                break;
            }
        }
    }
}

// ============================================================================
// ## 7. MAIN TRAINING LOOP
// ============================================================================

int min(int a, int b)
{
    return (a < b) ? a : b;
}

int main(int argc, char *argv[])
{
    srand(time(NULL));
    printf("=== GPT-2 MoE Transformer ===\n");
    printf("Starting program initialization...\n");
    fflush(stdout);

    // 1. Setup Configuration
    Config config = {
        .vocab_size = 64, // Will be adjusted based on dataset
        .seq_len = 32,    // Shorter sequences for demo
        .embed_dim = 64,  // Smaller for faster training
        .num_layers = 2,  // Fewer layers
        .num_heads = 4,
        .num_experts = 4, // Fewer experts
        .top_k = 2,
        .hidden_dim = 128};

    // Check command line arguments
    if (argc > 1)
    {
        if (strcmp(argv[1], "generate") == 0)
        {
            printf("[MODE] Text Generation Mode\n");
            printf("Loading model for text generation...\n");

            // Load model and generate text
            GPT2_MoE_Model model;
            RunState state;

            if (!load_model(&model, "moe_model.bin"))
            {
                printf("ERROR: Failed to load model. Please train first.\n");
                return 1;
            }

            config = model.config;
            printf("✓ Model loaded successfully\n");
            printf("  Configuration: seq_len=%d, embed_dim=%d, vocab_size=%d\n",
                   config.seq_len, config.embed_dim, config.vocab_size);
            printf("  Layers: %d, Experts: %d, Heads: %d\n",
                   config.num_layers, config.num_experts, config.num_heads);

            build_state(&state, &config);
            printf("✓ Run state initialized\n");

            Dataset dataset;
            printf("Loading dataset...\n");
            load_tiny_shakespeare(&dataset);
            config.vocab_size = dataset.vocab_size;
            printf("✓ Dataset loaded: %d tokens, %d vocabulary size\n",
                   dataset.num_tokens, dataset.vocab_size);

            // Parse command line arguments for generation
            const char *prompt = "To be or not to be";
            int max_tokens = 100;
            float temperature = 0.8f;
            float top_p = 0.9f;

            // Parse additional arguments
            printf("Parsing command line arguments...\n");
            for (int i = 2; i < argc; i++)
            {
                if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc)
                {
                    prompt = argv[i + 1];
                    printf("  ✓ Prompt set: \"%s\"\n", prompt);
                    i++;
                }
                else if (strcmp(argv[i], "--length") == 0 && i + 1 < argc)
                {
                    max_tokens = atoi(argv[i + 1]);
                    printf("  ✓ Generation length set: %d tokens\n", max_tokens);
                    i++;
                }
                else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc)
                {
                    temperature = atof(argv[i + 1]);
                    printf("  ✓ Temperature set: %.2f\n", temperature);
                    i++;
                }
                else if (strcmp(argv[i], "--top_p") == 0 && i + 1 < argc)
                {
                    top_p = atof(argv[i + 1]);
                    printf("  ✓ Top-p sampling set: %.2f\n", top_p);
                    i++;
                }
                else
                {
                    printf("  ⚠ Unknown argument: %s\n", argv[i]);
                }
            }

            printf("\n=== Generation Parameters ===\n");
            printf("Prompt: \"%s\"\n", prompt);
            printf("Length: %d tokens\n", max_tokens);
            printf("Temperature: %.2f\n", temperature);
            printf("Top-p: %.2f\n", top_p);

            generate_text(&model, &state, &dataset, prompt, max_tokens, temperature, top_p);

            free_state(&state, &config);
            free_model(&model);
            free_dataset(&dataset);
            printf("✓ Cleanup completed\n");
            return 0;
        }
        else if (strcmp(argv[1], "analyze") == 0)
        {
            printf("[MODE] Expert Usage Analysis Mode\n");
            printf("Loading model for expert analysis...\n");

            // Load model and analyze expert usage
            GPT2_MoE_Model model;
            RunState state;

            if (!load_model(&model, "moe_model.bin"))
            {
                printf("ERROR: Failed to load model. Please train first.\n");
                return 1;
            }

            config = model.config;
            printf("✓ Model loaded successfully\n");
            build_state(&state, &config);
            printf("✓ Run state initialized\n");

            Dataset dataset;
            printf("Loading dataset...\n");
            load_tiny_shakespeare(&dataset);
            config.vocab_size = dataset.vocab_size;
            printf("✓ Dataset loaded\n");

            // Run one forward pass to collect expert usage data
            printf("Running forward pass for analysis...\n");
            int max_start = dataset.num_tokens - config.seq_len - 1;
            if (max_start > 0)
            {
                int start_pos = rand() % max_start;
                int *inputs = &dataset.tokens[start_pos];
                printf("Processing sequence starting at position %d\n", start_pos);
                forward_pass(&model, &state, inputs);
                analyze_expert_usage(&state, &config, 1);
            }
            else
            {
                printf("ERROR: Dataset too small for analysis\n");
            }

            free_state(&state, &config);
            free_model(&model);
            free_dataset(&dataset);
            printf("✓ Analysis completed\n");
            return 0;
        }
        else if (strcmp(argv[1], "train") != 0)
        {
            printf("=== GPT-2 MoE Transformer ===\n");
            printf("Invalid command: %s\n\n", argv[1]);
            printf("Usage: %s [train|generate|analyze]\n", argv[0]);
            printf("\nCommands:\n");
            printf("  train                           Train the model\n");
            printf("  generate                       Generate text with default parameters\n");
            printf("  generate [options]             Generate text with custom parameters\n");
            printf("  analyze                        Analyze expert usage statistics\n");
            printf("\nGeneration Options:\n");
            printf("  --prompt \"text\"               Set the initial prompt\n");
            printf("  --length N                     Set number of tokens to generate\n");
            printf("  --temperature T                Set sampling temperature (0.0-2.0)\n");
            printf("  --top_p P                      Set nucleus sampling threshold (0.0-1.0)\n");
            printf("\nExamples:\n");
            printf("  %s train\n", argv[0]);
            printf("  %s generate\n", argv[0]);
            printf("  %s generate --prompt \"Once upon a time\" --length 100\n", argv[0]);
            printf("  %s generate --prompt \"Hello\" --temperature 0.7 --top_p 0.9\n", argv[0]);
            printf("  %s analyze\n", argv[0]);
            return 1;
        }
    }

    // Training mode
    printf("[MODE] Training Mode\n");

    // 2. Load Data
    printf("Loading dataset...\n");
    fflush(stdout);
    Dataset dataset;
    load_tiny_shakespeare(&dataset);
    config.vocab_size = dataset.vocab_size;

    printf("=== MoE Transformer Training ===\n");
    printf("✓ Loaded dataset with %d tokens and vocabulary size %d\n",
           dataset.num_tokens, dataset.vocab_size);
    printf("Model configuration:\n");
    printf("  Sequence length: %d\n", config.seq_len);
    printf("  Embedding dimension: %d\n", config.embed_dim);
    printf("  Number of layers: %d\n", config.num_layers);
    printf("  Number of experts: %d\n", config.num_experts);
    printf("  Top-K experts: %d\n", config.top_k);
    printf("  Hidden dimension: %d\n", config.hidden_dim);

    // 3. Initialize Model and State
    printf("Building model...\n");
    fflush(stdout);
    GPT2_MoE_Model model;
    RunState state;
    Optimizer optimizer;
    build_model(&model, &config);
    printf("✓ Model built successfully\n");

    printf("Building state...\n");
    fflush(stdout);
    build_state(&state, &config);
    printf("✓ State built successfully\n");

    int approx_params = config.vocab_size * config.embed_dim +
                        config.num_layers * config.num_experts * config.hidden_dim * config.embed_dim;
    printf("\nModel initialized with ~%d parameters\n", approx_params);

    // 4. Initialize Optimizer
    printf("Initializing optimizer...\n");
    fflush(stdout);
    init_optimizer(&optimizer, &config);
    printf("✓ Optimizer initialized\n");

    // 5. Training Loop
    printf("\n=== Training Started ===\n");
    fflush(stdout);
    int num_steps = 1000;
    float learning_rate = 0.0005f;
    float best_loss = INFINITY;

    time_t start_training_time = time(NULL);
    printf("Training will run for %d steps\n", num_steps);

    for (int step = 0; step < num_steps; step++)
    {
        printf("[STEP %4d] ", step);
        fflush(stdout);

        // Get a batch
        int max_start = dataset.num_tokens - config.seq_len - 1;
        if (max_start <= 0)
        {
            printf("ERROR: Dataset too small for sequence length\n");
            break;
        }

        int start_pos = rand() % max_start;
        int *inputs = &dataset.tokens[start_pos];
        int *targets = &dataset.tokens[start_pos + 1];

        // Forward pass
        forward_pass(&model, &state, inputs);

        // Calculate losses
        float primary_loss = calculate_crossentropy_loss(state.logits, targets, &config);
        float aux_loss = calculate_moe_aux_loss(&state, &config);
        float total_loss = primary_loss + 0.01f * aux_loss;

        if (total_loss < best_loss)
        {
            best_loss = total_loss;
        }

        if (!isfinite(total_loss))
        {
            printf("ERROR: Invalid loss detected (NaN/Inf)\n");
            break;
        }

        // Backward pass - FIXED: Pass inputs parameter
        backward_pass(&model, &state, inputs, targets, total_loss);

        // Update weights
        update_weights(&model, &state, &optimizer, learning_rate);

        // Progress reporting
        if (step % 10 == 0)
        {
            printf("Loss: %.4f (Primary: %.4f, Aux: %.4f) | Best: %.4f\n",
                   total_loss, primary_loss, aux_loss, best_loss);
        }

        // Save model periodically
        if (step % 100 == 0 && step > 0)
        {
            char filename[40];
            sprintf(filename, "moe_model_step_%d.bin", step);
            printf("  Saving checkpoint: %s\n", filename);
            save_model(&model, filename);
        }

        // Exit after first step for debugging
        if (step == 0)
        {
            printf("✓ First step completed successfully!\n");
            break;
        }
    }

    time_t end_training_time = time(NULL);
    double training_duration = difftime(end_training_time, start_training_time);

    printf("\n=== Training Completed ===\n");
    printf("Final best loss: %.4f\n", best_loss);
    printf("Training duration: %.0f seconds\n", training_duration);

    // Save final model
    printf("Saving final model...\n");
    save_model(&model, "moe_model.bin");
    printf("✓ Model saved as 'moe_model.bin'\n");

    // Analyze expert usage
    printf("Analyzing expert usage...\n");
    analyze_expert_usage(&state, &config, 1);

    // Cleanup
    printf("\n=== Cleaning Up ===\n");
    free_model(&model);
    free_state(&state, &config);
    free_dataset(&dataset);
    free_optimizer(&optimizer, &config);
    printf("✓ All resources cleaned up\n");

    printf("\n=== Training Success ===\n");
    printf("Training completed successfully in %.0f seconds!\n", training_duration);
    printf("\nNext steps:\n");
    printf("  %s generate --prompt \"Your text here\" --length 100\n", argv[0]);
    printf("  %s analyze\n", argv[0]);
    return 0;
}

// ============================================================================
// ## 8. MODEL SAVING AND LOADING
// ============================================================================

void save_model(GPT2_MoE_Model *model, const char *filename)
{
    FILE *file = fopen(filename, "wb");
    if (!file)
    {
        printf("Error: Could not open file %s for writing\n", filename);
        return;
    }

    Config *config = &model->config;

    // Write config
    fwrite(config, sizeof(Config), 1, file);

    // Write token embeddings
    fwrite(model->token_embeddings, sizeof(float),
           config->vocab_size * config->embed_dim, file);

    // Write positional embeddings
    fwrite(model->pos_embeddings, sizeof(float),
           config->seq_len * config->embed_dim, file);

    // Write transformer layers
    for (int l = 0; l < config->num_layers; l++)
    {
        TransformerBlock *layer = &model->layers[l];

        // Attention weights
        fwrite(layer->attn_qkv_w, sizeof(float),
               config->embed_dim * 3 * config->embed_dim, file);
        fwrite(layer->attn_qkv_b, sizeof(float), 3 * config->embed_dim, file);
        fwrite(layer->attn_proj_w, sizeof(float),
               config->embed_dim * config->embed_dim, file);
        fwrite(layer->attn_proj_b, sizeof(float), config->embed_dim, file);

        // Layer norm
        fwrite(layer->ln1_gamma, sizeof(float), config->embed_dim, file);
        fwrite(layer->ln1_beta, sizeof(float), config->embed_dim, file);
        fwrite(layer->ln2_gamma, sizeof(float), config->embed_dim, file);
        fwrite(layer->ln2_beta, sizeof(float), config->embed_dim, file);

        // MoE layer
        MoELayer *moe = &layer->moe_layer;
        fwrite(moe->gating_w, sizeof(float),
               config->embed_dim * config->num_experts, file);
        fwrite(moe->gating_b, sizeof(float), config->num_experts, file);

        // Experts
        for (int e = 0; e < config->num_experts; e++)
        {
            Expert *expert = &moe->experts[e];
            fwrite(expert->w1, sizeof(float),
                   config->embed_dim * config->hidden_dim, file);
            fwrite(expert->b1, sizeof(float), config->hidden_dim, file);
            fwrite(expert->w2, sizeof(float),
                   config->hidden_dim * config->embed_dim, file);
            fwrite(expert->b2, sizeof(float), config->embed_dim, file);
        }
    }

    // Write final layer norm
    fwrite(model->final_ln_gamma, sizeof(float), config->embed_dim, file);
    fwrite(model->final_ln_beta, sizeof(float), config->embed_dim, file);

    fclose(file);
    printf("Model saved successfully to %s\n", filename);
}

int load_model(GPT2_MoE_Model *model, const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        printf("Error: Could not open file %s for reading\n", filename);
        return 0;
    }

    // Read config
    if (fread(&model->config, sizeof(Config), 1, file) != 1)
    {
        printf("Error reading config from file\n");
        fclose(file);
        return 0;
    }

    Config *config = &model->config;

    // Allocate and read token embeddings
    model->token_embeddings = malloc(config->vocab_size * config->embed_dim * sizeof(float));
    fread(model->token_embeddings, sizeof(float),
          config->vocab_size * config->embed_dim, file);

    // Allocate and read positional embeddings
    model->pos_embeddings = malloc(config->seq_len * config->embed_dim * sizeof(float));
    fread(model->pos_embeddings, sizeof(float),
          config->seq_len * config->embed_dim, file);

    // Allocate transformer layers
    model->layers = malloc(config->num_layers * sizeof(TransformerBlock));

    for (int l = 0; l < config->num_layers; l++)
    {
        TransformerBlock *layer = &model->layers[l];

        // Allocate and read attention weights
        layer->attn_qkv_w = malloc(config->embed_dim * 3 * config->embed_dim * sizeof(float));
        layer->attn_qkv_b = malloc(3 * config->embed_dim * sizeof(float));
        layer->attn_proj_w = malloc(config->embed_dim * config->embed_dim * sizeof(float));
        layer->attn_proj_b = malloc(config->embed_dim * sizeof(float));

        fread(layer->attn_qkv_w, sizeof(float),
              config->embed_dim * 3 * config->embed_dim, file);
        fread(layer->attn_qkv_b, sizeof(float), 3 * config->embed_dim, file);
        fread(layer->attn_proj_w, sizeof(float),
              config->embed_dim * config->embed_dim, file);
        fread(layer->attn_proj_b, sizeof(float), config->embed_dim, file);

        // Allocate and read layer norm
        layer->ln1_gamma = malloc(config->embed_dim * sizeof(float));
        layer->ln1_beta = malloc(config->embed_dim * sizeof(float));
        layer->ln2_gamma = malloc(config->embed_dim * sizeof(float));
        layer->ln2_beta = malloc(config->embed_dim * sizeof(float));

        fread(layer->ln1_gamma, sizeof(float), config->embed_dim, file);
        fread(layer->ln1_beta, sizeof(float), config->embed_dim, file);
        fread(layer->ln2_gamma, sizeof(float), config->embed_dim, file);
        fread(layer->ln2_beta, sizeof(float), config->embed_dim, file);

        // Allocate and read MoE layer
        MoELayer *moe = &layer->moe_layer;
        moe->gating_w = malloc(config->embed_dim * config->num_experts * sizeof(float));
        moe->gating_b = malloc(config->num_experts * sizeof(float));

        fread(moe->gating_w, sizeof(float),
              config->embed_dim * config->num_experts, file);
        fread(moe->gating_b, sizeof(float), config->num_experts, file);

        // Allocate and read experts
        moe->experts = malloc(config->num_experts * sizeof(Expert));
        for (int e = 0; e < config->num_experts; e++)
        {
            Expert *expert = &moe->experts[e];

            expert->w1 = malloc(config->embed_dim * config->hidden_dim * sizeof(float));
            expert->b1 = malloc(config->hidden_dim * sizeof(float));
            expert->w2 = malloc(config->hidden_dim * config->embed_dim * sizeof(float));
            expert->b2 = malloc(config->embed_dim * sizeof(float));

            fread(expert->w1, sizeof(float),
                  config->embed_dim * config->hidden_dim, file);
            fread(expert->b1, sizeof(float), config->hidden_dim, file);
            fread(expert->w2, sizeof(float),
                  config->hidden_dim * config->embed_dim, file);
            fread(expert->b2, sizeof(float), config->embed_dim, file);
        }
    }

    // Allocate and read final layer norm
    model->final_ln_gamma = malloc(config->embed_dim * sizeof(float));
    model->final_ln_beta = malloc(config->embed_dim * sizeof(float));
    fread(model->final_ln_gamma, sizeof(float), config->embed_dim, file);
    fread(model->final_ln_beta, sizeof(float), config->embed_dim, file);

    fclose(file);
    printf("Model loaded successfully from %s\n", filename);
    return 1;
}

// ============================================================================
// ## 9. TEXT GENERATION
// ============================================================================

int sample_from_logits(float *logits, int vocab_size, float temperature, float top_p)
{
    // Apply temperature
    for (int i = 0; i < vocab_size; i++)
    {
        logits[i] /= temperature;
    }

    // Apply softmax
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++)
    {
        if (logits[i] > max_logit)
            max_logit = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++)
    {
        logits[i] = exp(logits[i] - max_logit);
        sum += logits[i];
    }

    // Normalize probabilities
    for (int i = 0; i < vocab_size; i++)
    {
        logits[i] /= sum;
    }

    // Top-p sampling implementation
    if (top_p < 1.0f)
    {
        // Create array of (probability, index) pairs
        typedef struct
        {
            float prob;
            int index;
        } ProbIndex;

        ProbIndex *prob_indices = malloc(vocab_size * sizeof(ProbIndex));
        for (int i = 0; i < vocab_size; i++)
        {
            prob_indices[i].prob = logits[i];
            prob_indices[i].index = i;
        }

        // Sort by probability in descending order
        for (int i = 0; i < vocab_size - 1; i++)
        {
            for (int j = 0; j < vocab_size - i - 1; j++)
            {
                if (prob_indices[j].prob < prob_indices[j + 1].prob)
                {
                    ProbIndex temp = prob_indices[j];
                    prob_indices[j] = prob_indices[j + 1];
                    prob_indices[j + 1] = temp;
                }
            }
        }

        // Find the cutoff point for top-p
        float cumulative_prob = 0.0f;
        int cutoff_index = 0;
        for (int i = 0; i < vocab_size; i++)
        {
            cumulative_prob += prob_indices[i].prob;
            if (cumulative_prob >= top_p)
            {
                cutoff_index = i;
                break;
            }
        }

        // Sample from the top-p subset
        float r = (float)rand() / RAND_MAX;
        float cumsum = 0.0f;
        for (int i = 0; i <= cutoff_index; i++)
        {
            cumsum += prob_indices[i].prob;
            if (r <= cumsum)
            {
                int result = prob_indices[i].index;
                free(prob_indices);
                return result;
            }
        }

        // Fallback
        free(prob_indices);
        return prob_indices[cutoff_index].index;
    }
    else
    {
        // Standard sampling (when top_p >= 1.0)
        float r = (float)rand() / RAND_MAX;
        float cumsum = 0.0f;
        for (int i = 0; i < vocab_size; i++)
        {
            cumsum += logits[i];
            if (r <= cumsum)
            {
                return i;
            }
        }
        return vocab_size - 1; // fallback
    }
}

void generate_text(GPT2_MoE_Model *model, RunState *state, Dataset *dataset,
                   const char *prompt, int max_tokens, float temperature, float top_p)
{
    Config *config = &model->config;
    int seq_len = config->seq_len;

    // Start timing
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    printf("Starting text generation...\n");
    printf("Prompt: \"%s\"\n", prompt);
    printf("Max tokens: %d, Temperature: %.2f, Top-p: %.2f\n", max_tokens, temperature, top_p);
    fflush(stdout);

    // Tokenize prompt
    int prompt_len = strlen(prompt);
    int *tokens = malloc(seq_len * sizeof(int));
    memset(tokens, 0, seq_len * sizeof(int));

    printf("Tokenizing prompt (%d characters)...\n", prompt_len);
    fflush(stdout);

    // Simple character-level tokenization
    int valid_tokens = 0;
    for (int i = 0; i < prompt_len && i < seq_len; i++)
    {
        char c = prompt[i];
        int found = 0;
        for (int v = 0; v < dataset->vocab_size; v++)
        {
            if (dataset->vocab[v][0] == c)
            {
                tokens[i] = v;
                found = 1;
                valid_tokens++;
                break;
            }
        }
        if (!found)
        {
            tokens[i] = 0; // Use default token if character not found
        }
    }

    printf("Tokenized %d valid tokens from prompt\n", valid_tokens);
    printf("Generated: \"");
    fflush(stdout);

    // Initialize the token buffer with prompt tokens
    int tokens_generated = 0;

    printf("\nGenerating tokens...\n");
    fflush(stdout);

    // Generate tokens
    for (int gen = 0; gen < max_tokens; gen++)
    {
        if (gen % 75 == 0 && gen > 0)
        {
            printf("\n[Progress: %d/%d tokens]\n", gen, max_tokens);
            fflush(stdout);
        }

        // Forward pass
        forward_pass(model, state, tokens);

        // Get logits for the last position in the sequence
        int pred_pos = (prompt_len + tokens_generated) % seq_len;

        float *last_logits = &state->logits[pred_pos * config->vocab_size];

        // Sample next token with temperature and top-p
        int next_token = sample_from_logits(last_logits, config->vocab_size, temperature, top_p);

        // Print the generated character
        if (next_token < dataset->vocab_size && dataset->vocab[next_token][0] != '\0')
        {
            printf("%c", dataset->vocab[next_token][0]);
            fflush(stdout);
        }
        else
        {
            printf("?"); // Print ? for unknown tokens
            fflush(stdout);
        }

        // Update token buffer - shift all tokens left and add new token at the end
        if (prompt_len + tokens_generated < seq_len)
        {
            // Still filling the initial buffer
            tokens[prompt_len + tokens_generated] = next_token;
        }
        else
        {
            // Sliding window: shift all tokens left by 1 and add new token at the end
            for (int i = 0; i < seq_len - 1; i++)
            {
                tokens[i] = tokens[i + 1];
            }
            tokens[seq_len - 1] = next_token;
        }

        tokens_generated++;

        // Small delay to make generation visible (but not too slow)
        if (gen % 25 == 0)
        {
            usleep(1000); // 1ms delay every 25 tokens
        }
    }

    // End timing
    gettimeofday(&end_time, NULL);

    // Calculate elapsed time in seconds
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                          (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

    // Calculate tokens per second
    double tokens_per_second = tokens_generated / elapsed_time;

    printf("\"\n\n");
    printf("=== Generation Complete ===\n");
    printf("Tokens generated: %d\n", tokens_generated);
    printf("Time taken: %.2f seconds\n", elapsed_time);
    printf("Speed: %.2f tokens/second\n", tokens_per_second);
    printf("Average time per token: %.4f seconds\n", elapsed_time / tokens_generated);

    free(tokens);
}

// ============================================================================
// ## 10. MEMORY CLEANUP
// ============================================================================

void free_model(GPT2_MoE_Model *model)
{
    Config *config = &model->config;

    free(model->token_embeddings);
    free(model->pos_embeddings);

    for (int l = 0; l < config->num_layers; l++)
    {
        TransformerBlock *layer = &model->layers[l];

        free(layer->attn_qkv_w);
        free(layer->attn_qkv_b);
        free(layer->attn_proj_w);
        free(layer->attn_proj_b);
        free(layer->ln1_gamma);
        free(layer->ln1_beta);
        free(layer->ln2_gamma);
        free(layer->ln2_beta);

        MoELayer *moe = &layer->moe_layer;
        free(moe->gating_w);
        free(moe->gating_b);

        for (int e = 0; e < config->num_experts; e++)
        {
            Expert *expert = &moe->experts[e];
            free(expert->w1);
            free(expert->b1);
            free(expert->w2);
            free(expert->b2);
        }
        free(moe->experts);
    }

    free(model->layers);
    free(model->final_ln_gamma);
    free(model->final_ln_beta);
}

void free_state(RunState *state, Config *config)
{

    // Free forward pass buffers
    free(state->embedding_out);
    free(state->layer_outputs);
    free(state->attn_out);
    free(state->moe_out);
    free(state->logits);
    free(state->gating_logits);
    free(state->expert_indices);
    free(state->expert_weights);
    free(state->expert_outputs);
    free(state->temp_buffer);
    free(state->temp_buffer2);

    // Free pre-allocated buffers
    free(state->gating_scores_buffer);
    free(state->hidden_buffer);
    free(state->expert_out_buffer);
    free(state->attention_scores_buffer);
    free(state->qkv_buffer);

    // Free gradient buffers
    free(state->d_embedding_out);
    free(state->d_layer_outputs);
    free(state->d_attn_out);
    free(state->d_moe_out);
    free(state->d_temp_buffer);
    free(state->d_temp_buffer2);
    free(state->d_token_embeddings);
    free(state->d_pos_embeddings);
    free(state->d_final_ln_gamma);
    free(state->d_final_ln_beta);

    free(state->d_ln1_out);
    free(state->d_ln2_out);
    free(state->d_attn_residual_out);
    free(state->d_attn_concat);
    free(state->d_attention_scores);
    free(state->d_query);
    free(state->d_key);
    free(state->d_value);
    free(state->d_softmax_scores);

    free(state->attention_probs);

    // Free layer gradients
    for (int l = 0; l < config->num_layers; l++)
    {
        free(state->d_attn_qkv_w[l]);
        free(state->d_attn_qkv_b[l]);
        free(state->d_attn_proj_w[l]);
        free(state->d_attn_proj_b[l]);
        free(state->d_ln1_gamma[l]);
        free(state->d_ln1_beta[l]);
        free(state->d_ln2_gamma[l]);
        free(state->d_ln2_beta[l]);
        free(state->d_gating_w[l]);
        free(state->d_gating_b[l]);

        // Free expert gradients
        for (int e = 0; e < config->num_experts; e++)
        {
            free(state->d_expert_w1[l][e]);
            free(state->d_expert_b1[l][e]);
            free(state->d_expert_w2[l][e]);
            free(state->d_expert_b2[l][e]);
        }
        free(state->d_expert_w1[l]);
        free(state->d_expert_b1[l]);
        free(state->d_expert_w2[l]);
        free(state->d_expert_b2[l]);
    }

    free(state->d_attn_qkv_w);
    free(state->d_attn_qkv_b);
    free(state->d_attn_proj_w);
    free(state->d_attn_proj_b);
    free(state->d_ln1_gamma);
    free(state->d_ln1_beta);
    free(state->d_ln2_gamma);
    free(state->d_ln2_beta);
    free(state->d_gating_w);
    free(state->d_gating_b);
    free(state->d_expert_w1);
    free(state->d_expert_b1);
    free(state->d_expert_w2);
    free(state->d_expert_b2);

    free(state->ln1_outputs);
    free(state->ln2_outputs);
    free(state->attn_outputs);
    free(state->attn_residual);
    free(state->moe_outputs);
    free(state->gating_scores);
    free(state->expert_hiddens);
    free(state->attn_concat);
}

void free_dataset(Dataset *dataset)
{
    free(dataset->data);
    free(dataset->tokens);
    for (int i = 0; i < dataset->vocab_size; i++)
    {
        free(dataset->vocab[i]);
    }
}

void free_optimizer(Optimizer *opt, Config *config)
{
    free(opt->m_token_embeddings);
    free(opt->v_token_embeddings);
    free(opt->m_pos_embeddings);
    free(opt->v_pos_embeddings);
    free(opt->m_final_ln_gamma);
    free(opt->v_final_ln_gamma);
    free(opt->m_final_ln_beta);
    free(opt->v_final_ln_beta);

    for (int l = 0; l < config->num_layers; l++)
    {
        free(opt->m_attn_qkv_w[l]);
        free(opt->v_attn_qkv_w[l]);
        free(opt->m_attn_qkv_b[l]);
        free(opt->v_attn_qkv_b[l]);
        free(opt->m_attn_proj_w[l]);
        free(opt->v_attn_proj_w[l]);
        free(opt->m_attn_proj_b[l]);
        free(opt->v_attn_proj_b[l]);
        free(opt->m_ln1_gamma[l]);
        free(opt->v_ln1_gamma[l]);
        free(opt->m_ln1_beta[l]);
        free(opt->v_ln1_beta[l]);
        free(opt->m_ln2_gamma[l]);
        free(opt->v_ln2_gamma[l]);
        free(opt->m_ln2_beta[l]);
        free(opt->v_ln2_beta[l]);
        free(opt->m_gating_w[l]);
        free(opt->v_gating_w[l]);
        free(opt->m_gating_b[l]);
        free(opt->v_gating_b[l]);

        for (int e = 0; e < config->num_experts; e++)
        {
            free(opt->m_expert_w1[l][e]);
            free(opt->v_expert_w1[l][e]);
            free(opt->m_expert_b1[l][e]);
            free(opt->v_expert_b1[l][e]);
            free(opt->m_expert_w2[l][e]);
            free(opt->v_expert_w2[l][e]);
            free(opt->m_expert_b2[l][e]);
            free(opt->v_expert_b2[l][e]);
        }

        free(opt->m_expert_w1[l]);
        free(opt->v_expert_w1[l]);
        free(opt->m_expert_b1[l]);
        free(opt->v_expert_b1[l]);
        free(opt->m_expert_w2[l]);
        free(opt->v_expert_w2[l]);
        free(opt->m_expert_b2[l]);
        free(opt->v_expert_b2[l]);
    }

    free(opt->m_attn_qkv_w);
    free(opt->v_attn_qkv_w);
    free(opt->m_attn_qkv_b);
    free(opt->v_attn_qkv_b);
    free(opt->m_attn_proj_w);
    free(opt->v_attn_proj_w);
    free(opt->m_attn_proj_b);
    free(opt->v_attn_proj_b);
    free(opt->m_ln1_gamma);
    free(opt->v_ln1_gamma);
    free(opt->m_ln1_beta);
    free(opt->v_ln1_beta);
    free(opt->m_ln2_gamma);
    free(opt->v_ln2_gamma);
    free(opt->m_ln2_beta);
    free(opt->v_ln2_beta);
    free(opt->m_gating_w);
    free(opt->v_gating_w);
    free(opt->m_gating_b);
    free(opt->v_gating_b);
    free(opt->m_expert_w1);
    free(opt->v_expert_w1);
    free(opt->m_expert_b1);
    free(opt->v_expert_b1);
    free(opt->m_expert_w2);
    free(opt->v_expert_w2);
    free(opt->m_expert_b2);
    free(opt->v_expert_b2);
}

// ============================================================================
// ## 11. EXPERT UTILIZATION ANALYSIS
// ============================================================================

void analyze_expert_usage(RunState *state, Config *config, int num_steps)
{
    printf("\n=== Expert Utilization Analysis ===\n");

    int *expert_counts = calloc(config->num_experts, sizeof(int));
    int total_selections = 0;

    // Analyze expert usage across all layers and tokens
    for (int l = 0; l < config->num_layers; l++)
    {
        for (int t = 0; t < config->seq_len; t++)
        {
            for (int k = 0; k < config->top_k; k++)
            {
                int idx = (l * config->seq_len + t) * config->top_k + k;
                int expert_idx = state->expert_indices[idx];
                if (expert_idx >= 0 && expert_idx < config->num_experts)
                {
                    expert_counts[expert_idx]++;
                    total_selections++;
                }
            }
        }
    }

    printf("Expert usage distribution (all layers):\n");
    for (int e = 0; e < config->num_experts; e++)
    {
        float usage_percent = total_selections > 0 ? (float)expert_counts[e] / total_selections * 100.0f : 0.0f;
        printf("Expert %d: %d uses (%.1f%%)\n", e, expert_counts[e], usage_percent);
    }

    free(expert_counts);
}