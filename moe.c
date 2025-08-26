#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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
void save_model(GPT2_MoE_Model *model, const char *filename);
void generate_text(GPT2_MoE_Model *model, RunState *state, Dataset *dataset, const char *prompt, int max_tokens);
void free_model(GPT2_MoE_Model *model);
void free_state(RunState *state, Config *config);
void free_dataset(Dataset *dataset);

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
    float max_val = x[0];
    for (int i = 1; i < size; i++)
    {
        if (x[i] > max_val)
            max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < size; i++)
    {
        x[i] /= sum;
    }
}

// Layer normalization
void layer_norm(float *out, float *x, float *gamma, float *beta, int size)
{
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

    float std = sqrt(var + 1e-5f);
    for (int i = 0; i < size; i++)
    {
        out[i] = (x[i] - mean) / std * gamma[i] + beta[i];
    }
}

// Matrix multiplication: C = A * B^T
void matmul(float *c, float *a, float *b, int n, int d, int k)
{
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
    for (int i = 0; i < size; i++)
    {
        if (x[i] < 0)
            x[i] = 0;
    }
}

// Top-K selection for MoE gating
void topk_indices(int *indices, float *values, float *scores, int num_experts, int k)
{
    // Simple selection sort for top-k
    for (int i = 0; i < k; i++)
    {
        int max_idx = i;
        for (int j = i + 1; j < num_experts; j++)
        {
            if (scores[j] > scores[max_idx])
            {
                max_idx = j;
            }
        }
        // Swap
        float temp_score = scores[i];
        scores[i] = scores[max_idx];
        scores[max_idx] = temp_score;

        indices[i] = (i == max_idx) ? i : max_idx;
        values[i] = scores[i];
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
    state->embedding_out = malloc(config->seq_len * config->embed_dim * sizeof(float));
    state->layer_outputs = malloc(config->num_layers * config->seq_len * config->embed_dim * sizeof(float));
    state->attn_out = malloc(config->seq_len * config->embed_dim * sizeof(float));
    state->moe_out = malloc(config->seq_len * config->embed_dim * sizeof(float));
    state->logits = malloc(config->seq_len * config->vocab_size * sizeof(float));

    state->gating_logits = malloc(config->seq_len * config->num_experts * sizeof(float));
    state->expert_indices = malloc(config->seq_len * config->top_k * sizeof(int));
    state->expert_weights = malloc(config->seq_len * config->top_k * sizeof(float));
    state->expert_outputs = malloc(config->seq_len * config->top_k * config->embed_dim * sizeof(float));

    state->temp_buffer = malloc(config->seq_len * config->embed_dim * sizeof(float));
    state->temp_buffer2 = malloc(config->seq_len * config->embed_dim * sizeof(float));

    state->gating_scores_buffer = malloc(config->num_experts * sizeof(float));
    state->hidden_buffer = malloc(config->hidden_dim * sizeof(float));
    state->expert_out_buffer = malloc(config->embed_dim * sizeof(float));
    state->attention_scores_buffer = malloc(config->seq_len * config->seq_len * sizeof(float));
    state->qkv_buffer = malloc(config->seq_len * 3 * config->embed_dim * sizeof(float));

    // Allocate intermediate storage
    state->ln1_outputs = malloc(config->num_layers * config->seq_len * config->embed_dim * sizeof(float));
    state->ln2_outputs = malloc(config->num_layers * config->seq_len * config->embed_dim * sizeof(float));
    state->attn_outputs = malloc(config->num_layers * config->seq_len * config->embed_dim * sizeof(float));
    state->attn_residual = malloc(config->num_layers * config->seq_len * config->embed_dim * sizeof(float));
    state->moe_outputs = malloc(config->num_layers * config->seq_len * config->embed_dim * sizeof(float));
    state->gating_scores = malloc(config->num_layers * config->seq_len * config->num_experts * sizeof(float));
    state->expert_hiddens = malloc(config->num_layers * config->num_experts * config->seq_len * config->hidden_dim * sizeof(float));
    state->attn_concat = malloc(config->num_layers * config->seq_len * config->embed_dim * sizeof(float));

    // Gradient buffers
    state->d_embedding_out = malloc(config->seq_len * config->embed_dim * sizeof(float));
    state->d_layer_outputs = malloc(config->num_layers * config->seq_len * config->embed_dim * sizeof(float));
    state->d_attn_out = malloc(config->seq_len * config->embed_dim * sizeof(float));
    state->d_moe_out = malloc(config->seq_len * config->embed_dim * sizeof(float));
    state->d_temp_buffer = malloc(config->seq_len * config->embed_dim * sizeof(float));
    state->d_temp_buffer2 = malloc(config->seq_len * config->embed_dim * sizeof(float));

    // Model parameter gradients
    state->d_token_embeddings = malloc(config->vocab_size * config->embed_dim * sizeof(float));
    state->d_pos_embeddings = malloc(config->seq_len * config->embed_dim * sizeof(float));
    state->d_final_ln_gamma = malloc(config->embed_dim * sizeof(float));
    state->d_final_ln_beta = malloc(config->embed_dim * sizeof(float));

    // Layer gradients
    state->d_attn_qkv_w = malloc(config->num_layers * sizeof(float *));
    state->d_attn_qkv_b = malloc(config->num_layers * sizeof(float *));
    state->d_attn_proj_w = malloc(config->num_layers * sizeof(float *));
    state->d_attn_proj_b = malloc(config->num_layers * sizeof(float *));
    state->d_ln1_gamma = malloc(config->num_layers * sizeof(float *));
    state->d_ln1_beta = malloc(config->num_layers * sizeof(float *));
    state->d_ln2_gamma = malloc(config->num_layers * sizeof(float *));
    state->d_ln2_beta = malloc(config->num_layers * sizeof(float *));
    state->d_gating_w = malloc(config->num_layers * sizeof(float *));
    state->d_gating_b = malloc(config->num_layers * sizeof(float *));

    // Expert gradients
    state->d_expert_w1 = malloc(config->num_layers * sizeof(float **));
    state->d_expert_b1 = malloc(config->num_layers * sizeof(float **));
    state->d_expert_w2 = malloc(config->num_layers * sizeof(float **));
    state->d_expert_b2 = malloc(config->num_layers * sizeof(float **));
    state->expert_outputs = malloc(config->num_layers * config->num_experts * config->seq_len * config->embed_dim * sizeof(float));

    for (int l = 0; l < config->num_layers; l++)
    {
        state->d_attn_qkv_w[l] = malloc(config->embed_dim * 3 * config->embed_dim * sizeof(float));
        state->d_attn_qkv_b[l] = malloc(3 * config->embed_dim * sizeof(float));
        state->d_attn_proj_w[l] = malloc(config->embed_dim * config->embed_dim * sizeof(float));
        state->d_attn_proj_b[l] = malloc(config->embed_dim * sizeof(float));
        state->d_ln1_gamma[l] = malloc(config->embed_dim * sizeof(float));
        state->d_ln1_beta[l] = malloc(config->embed_dim * sizeof(float));
        state->d_ln2_gamma[l] = malloc(config->embed_dim * sizeof(float));
        state->d_ln2_beta[l] = malloc(config->embed_dim * sizeof(float));
        state->d_gating_w[l] = malloc(config->embed_dim * config->num_experts * sizeof(float));
        state->d_gating_b[l] = malloc(config->num_experts * sizeof(float));

        state->d_expert_w1[l] = malloc(config->num_experts * sizeof(float *));
        state->d_expert_b1[l] = malloc(config->num_experts * sizeof(float *));
        state->d_expert_w2[l] = malloc(config->num_experts * sizeof(float *));
        state->d_expert_b2[l] = malloc(config->num_experts * sizeof(float *));

        for (int e = 0; e < config->num_experts; e++)
        {
            state->d_expert_w1[l][e] = malloc(config->embed_dim * config->hidden_dim * sizeof(float));
            state->d_expert_b1[l][e] = malloc(config->hidden_dim * sizeof(float));
            state->d_expert_w2[l][e] = malloc(config->hidden_dim * config->embed_dim * sizeof(float));
            state->d_expert_b2[l][e] = malloc(config->embed_dim * sizeof(float));
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

void multi_head_attention(float *out, float *x, TransformerBlock *layer, Config *config, RunState *state)
{
    int seq_len = config->seq_len;
    int embed_dim = config->embed_dim;
    int num_heads = config->num_heads;
    int head_dim = embed_dim / num_heads;

    // Use pre-allocated buffer
    float *qkv = state->qkv_buffer;
    matmul(qkv, x, layer->attn_qkv_w, seq_len, embed_dim, 3 * embed_dim);
    add_bias(qkv, layer->attn_qkv_b, seq_len, 3 * embed_dim);

    float *q = qkv;
    float *k = qkv + seq_len * embed_dim;
    float *v = qkv + 2 * seq_len * embed_dim;

    memset(out, 0, seq_len * embed_dim * sizeof(float));

    for (int h = 0; h < num_heads; h++)
    {
        int head_offset = h * head_dim;

        // Compute full attention matrix for this head
        for (int i = 0; i < seq_len; i++)
        {
            for (int j = 0; j < seq_len; j++)
            {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++)
                {
                    score += q[i * embed_dim + head_offset + d] * k[j * embed_dim + head_offset + d];
                }
                state->attention_scores_buffer[i * seq_len + j] = (j <= i) ? score / sqrt(head_dim) : -INFINITY;
            }

            // Apply softmax to row i
            softmax(&state->attention_scores_buffer[i * seq_len], seq_len);

            // Compute output for position i
            for (int d = 0; d < head_dim; d++)
            {
                for (int j = 0; j < seq_len; j++)
                {
                    out[i * embed_dim + head_offset + d] +=
                        state->attention_scores_buffer[i * seq_len + j] * v[j * embed_dim + head_offset + d];
                }
            }
        }
    }

    // Output projection (reuse qkv buffer)
    memcpy(state->qkv_buffer, out, seq_len * embed_dim * sizeof(float));
    matmul(out, state->qkv_buffer, layer->attn_proj_w, seq_len, embed_dim, embed_dim);
    add_bias(out, layer->attn_proj_b, seq_len, embed_dim);
}

void moe_forward(float *out, float *x, MoELayer *moe, RunState *state, Config *config)
{
    int seq_len = config->seq_len;
    int embed_dim = config->embed_dim;
    int num_experts = config->num_experts;
    int top_k = config->top_k;
    int hidden_dim = config->hidden_dim;

    memset(out, 0, seq_len * embed_dim * sizeof(float));

    // Process each token
    for (int t = 0; t < seq_len; t++)
    {
        float *token_x = &x[t * embed_dim];
        float *token_out = &out[t * embed_dim];

        // Use pre-allocated buffer
        float *gating_scores = state->gating_scores_buffer;
        for (int e = 0; e < num_experts; e++)
        {
            gating_scores[e] = 0.0f;
            for (int d = 0; d < embed_dim; d++)
            {
                gating_scores[e] += token_x[d] * moe->gating_w[e * embed_dim + d];
            }
            gating_scores[e] += moe->gating_b[e];
        }

        memcpy(&state->gating_logits[t * num_experts], gating_scores, num_experts * sizeof(float));
        softmax(gating_scores, num_experts);

        int *indices = &state->expert_indices[t * top_k];
        float *weights = &state->expert_weights[t * top_k];
        topk_indices(indices, weights, gating_scores, num_experts, top_k);

        // Renormalize
        float weight_sum = 0.0f;
        for (int k = 0; k < top_k; k++)
            weight_sum += weights[k];
        for (int k = 0; k < top_k; k++)
            weights[k] /= weight_sum;

        // Process experts using pre-allocated buffers
        for (int k = 0; k < top_k; k++)
        {
            Expert *expert = &moe->experts[indices[k]];

            // Use pre-allocated buffers
            float *hidden = state->hidden_buffer;
            float *expert_out = state->expert_out_buffer;

            // First layer
            for (int h = 0; h < hidden_dim; h++)
            {
                hidden[h] = expert->b1[h];
                for (int d = 0; d < embed_dim; d++)
                {
                    hidden[h] += token_x[d] * expert->w1[d * hidden_dim + h];
                }
            }
            relu(hidden, hidden_dim);

            // Second layer
            for (int d = 0; d < embed_dim; d++)
            {
                expert_out[d] = expert->b2[d];
                for (int h = 0; h < hidden_dim; h++)
                {
                    expert_out[d] += hidden[h] * expert->w2[h * embed_dim + d];
                }
            }

            // Accumulate weighted output
            for (int d = 0; d < embed_dim; d++)
            {
                token_out[d] += weights[k] * expert_out[d];
            }
        }
    }
}

void forward_pass(GPT2_MoE_Model *model, RunState *state, int *inputs)
{
    Config *config = &model->config;
    int seq_len = config->seq_len;
    int embed_dim = config->embed_dim;

    // 1. Embedding Layer
    for (int t = 0; t < seq_len; t++)
    {
        int token_id = inputs[t];
        for (int d = 0; d < embed_dim; d++)
        {
            state->embedding_out[t * embed_dim + d] =
                model->token_embeddings[token_id * embed_dim + d] +
                model->pos_embeddings[t * embed_dim + d];
        }
    }

    // Copy initial input
    memcpy(state->temp_buffer, state->embedding_out, seq_len * embed_dim * sizeof(float));

    // 2. Transformer Blocks
    for (int l = 0; l < config->num_layers; l++)
    {
        TransformerBlock *layer = &model->layers[l];

        // a. Layer Normalization 1
        layer_norm(state->temp_buffer2, state->temp_buffer, layer->ln1_gamma, layer->ln1_beta, seq_len * embed_dim);

        // STORE LN1 OUTPUT
        memcpy(&state->ln1_outputs[l * seq_len * embed_dim], state->temp_buffer2, seq_len * embed_dim * sizeof(float));

        // b. Multi-Head Attention
        multi_head_attention(state->attn_out, state->temp_buffer2, layer, config, state);

        // c. Residual Connection 1
        for (int i = 0; i < seq_len * embed_dim; i++)
        {
            state->temp_buffer[i] += state->attn_out[i];
        }

        // STORE ATTENTION RESIDUAL
        memcpy(&state->attn_residual[l * seq_len * embed_dim], state->temp_buffer, seq_len * embed_dim * sizeof(float));

        // d. Layer Normalization 2
        layer_norm(state->temp_buffer2, state->temp_buffer, layer->ln2_gamma, layer->ln2_beta, seq_len * embed_dim);

        // STORE LN2 OUTPUT (this line already exists but needs to be before MoE)
        memcpy(&state->ln2_outputs[l * seq_len * embed_dim], state->temp_buffer2, seq_len * embed_dim * sizeof(float));

        // e. Mixture of Experts Forward Pass
        moe_forward(state->moe_out, state->temp_buffer2, &layer->moe_layer, state, config);

        // f. Residual Connection 2
        for (int i = 0; i < seq_len * embed_dim; i++)
        {
            state->temp_buffer[i] += state->moe_out[i];
        }

        // Store layer output
        memcpy(&state->layer_outputs[l * seq_len * embed_dim], state->temp_buffer, seq_len * embed_dim * sizeof(float));
    }

    // 3. Final Layer Normalization
    layer_norm(state->temp_buffer2, state->temp_buffer, model->final_ln_gamma, model->final_ln_beta, seq_len * embed_dim);

    // 4. Classifier Head (tied weights)
    matmul(state->logits, state->temp_buffer2, model->token_embeddings, seq_len, embed_dim, config->vocab_size);
}

void backward_pass(GPT2_MoE_Model *model, RunState *state, int *targets, float total_loss)
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
    float *d_logits = malloc(seq_len * vocab_size * sizeof(float));
    for (int t = 0; t < seq_len; t++)
    {
        int target = targets[t];
        float *token_logits = &state->logits[t * vocab_size];
        float *d_token_logits = &d_logits[t * vocab_size];

        // Compute softmax probabilities (re-used from forward pass for correctness)
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
            d_token_logits[v] /= seq_len; // Average gradient over sequence
        }
    }

    // 2. Gradient through output projection (tied weights)
    float *final_ln_output = state->temp_buffer2;
    for (int t = 0; t < seq_len; t++)
    {
        for (int d = 0; d < embed_dim; d++)
        {
            state->d_temp_buffer[t * embed_dim + d] = 0.0f; // This will be d_final_ln_output
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
    float *d_final_ln_output = state->d_temp_buffer;      // Gradient from step 2
    float *final_transformer_output = state->temp_buffer; // Input to final LN from forward pass

    // The gradient for the input of the final LN will be accumulated in d_temp_buffer2
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
        // Get input to this layer for gradient calculations
        float *layer_input = (l == 0) ? state->embedding_out : &state->layer_outputs[(l - 1) * seq_len * embed_dim];

        // Current layer state pointers
        float *ln1_output = &state->ln1_outputs[l * seq_len * embed_dim];
        float *attn_output = &state->attn_outputs[l * seq_len * embed_dim];
        float *attn_residual = &state->attn_residual[l * seq_len * embed_dim];
        float *ln2_output = &state->ln2_outputs[l * seq_len * embed_dim];
        float *moe_output = &state->moe_outputs[l * seq_len * embed_dim];

        // 4a. Second residual connection: residual + MoE
        float *d_moe_out = state->d_moe_out;
        float *d_attn_residual_out = malloc(seq_len * embed_dim * sizeof(float));

        memcpy(d_moe_out, d_residual, seq_len * embed_dim * sizeof(float));
        memcpy(d_attn_residual_out, d_residual, seq_len * embed_dim * sizeof(float));

        // 4b. Backward through MoE Layer
        float *d_ln2_out = state->d_temp_buffer;
        memset(d_ln2_out, 0, seq_len * embed_dim * sizeof(float));

        for (int t = 0; t < seq_len; t++)
        {
            // Get expert assignments for this token
            float *gating_scores = &state->gating_scores[l * seq_len * config->num_experts + t * config->num_experts];
            int *expert_indices = &state->expert_indices[l * seq_len * config->top_k + t * config->top_k];
            float *expert_weights = &state->expert_weights[l * seq_len * config->top_k + t * config->top_k];
            float *ln2_token_out = &state->ln2_outputs[l * seq_len * embed_dim + t * embed_dim];

            // Gradient through expert mixing
            for (int k = 0; k < config->top_k; k++)
            {
                int expert_idx = expert_indices[k];
                float weight = expert_weights[k];

                // Get expert outputs for this token
                float *expert_out = &state->expert_outputs[l * config->num_experts * seq_len * embed_dim +
                                                           expert_idx * seq_len * embed_dim + t * embed_dim];

                // Gradient w.r.t. expert output
                for (int d = 0; d < embed_dim; d++)
                {
                    float d_expert_out = d_moe_out[t * embed_dim + d] * weight;

                    // Backward through expert FFN (expert_out = W2 * ReLU(W1 * x + b1) + b2)
                    float *expert_hidden = &state->expert_hiddens[l * config->num_experts * seq_len * config->hidden_dim +
                                                                  expert_idx * seq_len * config->hidden_dim + t * config->hidden_dim];

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

                // Gradient w.r.t. gating weights (simplified)
                for (int d = 0; d < embed_dim; d++)
                {
                    float d_weight = d_moe_out[t * embed_dim + d] * expert_out[d];

                    // Softmax gradient for gating network
                    for (int e = 0; e < config->num_experts; e++)
                    {
                        float grad_contribution = d_weight * gating_scores[e];
                        if (e == expert_idx)
                            grad_contribution -= d_weight;

                        state->d_gating_b[l][e] += grad_contribution / config->top_k;

                        for (int d2 = 0; d2 < embed_dim; d2++)
                        {
                            state->d_gating_w[l][e * config->embed_dim + d2] += grad_contribution * ln2_token_out[d2] / config->top_k;
                            d_ln2_out[t * embed_dim + d2] += grad_contribution * layer->moe_layer.gating_w[e * config->embed_dim + d2] / config->top_k;
                        }
                    }
                }
            }
        }

        // 4c. Backward through second LayerNorm
        float *d_attn_residual_out2 = malloc(seq_len * embed_dim * sizeof(float));
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

        // 4d. First residual connection: ln1_out + attn_out
        float *d_attn_out = malloc(seq_len * embed_dim * sizeof(float));
        float *d_ln1_out = malloc(seq_len * embed_dim * sizeof(float));

        memcpy(d_attn_out, d_attn_residual_out2, seq_len * embed_dim * sizeof(float));
        memcpy(d_ln1_out, d_attn_residual_out2, seq_len * embed_dim * sizeof(float));

        // 4e. Backward through Multi-Head Attention
        float *d_ln1_out2 = malloc(seq_len * embed_dim * sizeof(float));
        memset(d_ln1_out2, 0, seq_len * embed_dim * sizeof(float));

        // Simplified attention backward (full implementation would be very complex)
        for (int t = 0; t < seq_len; t++)
        {
            // Gradient through output projection
            for (int d = 0; d < embed_dim; d++)
            {
                state->d_attn_proj_b[l][d] += d_attn_out[t * embed_dim + d];

                for (int h = 0; h < embed_dim; h++)
                {
                    float *attn_concat = &state->attn_concat[l * seq_len * embed_dim + t * embed_dim];
                    state->d_attn_proj_w[l][h * embed_dim + d] += d_attn_out[t * embed_dim + d] * attn_concat[h];
                }
            }

            // Simplified QKV gradient (actual implementation needs attention scores)
            for (int d = 0; d < 3 * embed_dim; d++)
            {
                state->d_attn_qkv_b[l][d] += d_attn_out[t * embed_dim + d % embed_dim] * 0.1f; // Simplified

                for (int h = 0; h < embed_dim; h++)
                {
                    state->d_attn_qkv_w[l][h * 3 * embed_dim + d] += d_attn_out[t * embed_dim + d % embed_dim] * ln1_output[t * embed_dim + h] * 0.1f;
                    d_ln1_out2[t * embed_dim + h] += d_attn_out[t * embed_dim + d % embed_dim] * model->layers[l].attn_qkv_w[h * 3 * embed_dim + d] * 0.1f;
                }
            }
        }

        // Add attention input gradients
        for (int i = 0; i < seq_len * embed_dim; i++)
        {
            d_ln1_out2[i] += d_ln1_out[i];
        }

        // 4f. Backward through first LayerNorm
        float *d_layer_input = malloc(seq_len * embed_dim * sizeof(float));
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

        // Update gradient for next iteration
        d_residual = d_layer_input;

        // Cleanup
        free(d_moe_out);
        free(d_attn_residual_out);
        free(d_ln2_out);
        free(d_attn_residual_out2);
        free(d_attn_out);
        free(d_ln1_out);
        free(d_ln1_out2);
        if (l > 0)
            free(d_layer_input); // Don't free if it will be used for embedding gradients
    }

    // 5. Embedding gradients
    for (int t = 0; t < seq_len; t++)
    {
        int token_id = (t > 0) ? targets[t - 1] : 0; // Use previous target or first token
        for (int d = 0; d < embed_dim; d++)
        {
            state->d_pos_embeddings[t * embed_dim + d] += d_residual[t * embed_dim + d];
            state->d_token_embeddings[token_id * embed_dim + d] += d_residual[t * embed_dim + d];
        }
    }

    free(d_logits);
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
    // Simple character-level tokenization
    const char *sample_text = "To be or not to be, that is the question. "
                              "Whether 'tis nobler in the mind to suffer "
                              "the slings and arrows of outrageous fortune, "
                              "or to take arms against a sea of troubles "
                              "and by opposing end them.";

    dataset->size = strlen(sample_text);
    dataset->data = malloc(dataset->size + 1);
    strcpy(dataset->data, sample_text);

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

int main()
{
    srand(time(NULL));

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

    // 2. Load Data
    Dataset dataset;
    load_tiny_shakespeare(&dataset);
    config.vocab_size = dataset.vocab_size;

    printf("=== MoE Transformer Training ===\n");
    printf("Loaded dataset with %d tokens and vocabulary size %d\n",
           dataset.num_tokens, dataset.vocab_size);
    printf("Model configuration:\n");
    printf("- Sequence length: %d\n", config.seq_len);
    printf("- Embedding dimension: %d\n", config.embed_dim);
    printf("- Number of layers: %d\n", config.num_layers);
    printf("- Number of experts: %d\n", config.num_experts);
    printf("- Top-K experts: %d\n", config.top_k);
    printf("- Hidden dimension: %d\n", config.hidden_dim);

    // 3. Initialize Model and State
    GPT2_MoE_Model model;
    RunState state;
    build_model(&model, &config);
    build_state(&state, &config);

    int approx_params = config.vocab_size * config.embed_dim +
                        config.num_layers * config.num_experts * config.hidden_dim * config.embed_dim;
    printf("\nModel initialized with ~%d parameters\n", approx_params);

    // 4. Initialize Optimizer
    Optimizer optimizer = {
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .step = 0};

    // 5. Training Loop
    printf("\n=== Training Started ===\n");
    int num_steps = 1000;
    float learning_rate = 0.0005f;
    float best_loss = INFINITY;

    for (int step = 0; step < num_steps; step++)
    {
        // Get a batch of input tokens and target tokens
        int start_pos = rand() % (dataset.num_tokens - config.seq_len - 1);
        int *inputs = &dataset.tokens[start_pos];
        int *targets = &dataset.tokens[start_pos + 1];

        // Forward pass
        forward_pass(&model, &state, inputs);

        // Calculate losses
        float primary_loss = calculate_crossentropy_loss(state.logits, targets, &config);
        float aux_loss = calculate_moe_aux_loss(&state, &config);
        float total_loss = primary_loss + 0.01f * aux_loss;

        // Backward pass
        backward_pass(&model, &state, targets, total_loss);

        // Update weights
        update_weights(&model, &state, &optimizer, learning_rate);

        // Track best loss
        if (total_loss < best_loss)
        {
            best_loss = total_loss;
        }

        // Progress reporting
        if (step % 50 == 0)
        {
            printf("Step %4d | Loss: %.4f (Primary: %.4f, Aux: %.4f) | Best: %.4f\n",
                   step, total_loss, primary_loss, aux_loss, best_loss);
        }

        // Expert utilization analysis
        if (step % 200 == 0 && step > 0)
        {
            printf("\n--- Expert Utilization at Step %d ---\n", step);
            analyze_expert_usage(&state, &config, 1);
            printf("----------------------------------------\n");
        }

        // Learning rate decay
        if (step > 0 && step % 300 == 0)
        {
            learning_rate *= 0.9f;
            printf("Learning rate decayed to: %.6f\n", learning_rate);
        }
    }

    printf("\n=== Training Completed ===\n");
    printf("Final best loss: %.4f\n", best_loss);

    // 6. Final Expert Analysis
    printf("\n=== Final Expert Utilization ===\n");
    analyze_expert_usage(&state, &config, 10);

    // 7. Save Model
    printf("\n=== Saving Model ===\n");
    save_model(&model, "tinymoe.bin");

    // 8. Demonstrate Text Generation
    printf("\n=== Text Generation Demo ===\n");
    const char *prompts[] = {"To be", "the ", "of ", "and"};
    int num_prompts = sizeof(prompts) / sizeof(prompts[0]);

    for (int i = 0; i < num_prompts; i++)
    {
        printf("\nPrompt %d:\n", i + 1);
        generate_text(&model, &state, &dataset, prompts[i], 15);
    }

    // 9. Model Statistics
    printf("\n=== Model Statistics ===\n");
    printf("Training steps completed: %d\n", num_steps);
    printf("Final learning rate: %.6f\n", learning_rate);
    printf("Model file: tinymoe.bin\n");
    printf("Vocabulary size: %d characters\n", dataset.vocab_size);

    // Display some vocabulary
    printf("Vocabulary sample: ");
    for (int i = 0; i < min(10, dataset.vocab_size); i++)
    {
        if (dataset.vocab[i][0] == ' ')
        {
            printf("'SPACE' ");
        }
        else if (dataset.vocab[i][0] == '\n')
        {
            printf("'NEWLINE' ");
        }
        else
        {
            printf("'%c' ", dataset.vocab[i][0]);
        }
    }
    printf("\n");

    // 10. Cleanup
    printf("\n=== Cleaning Up ===\n");
    free_model(&model);
    free_state(&state, &config);
    free_dataset(&dataset);

    printf("Training and inference completed successfully!\n");
    printf("Check 'tinymoe.bin' for the trained model weights.\n");

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

int sample_from_logits(float *logits, int vocab_size, float temperature)
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

    for (int i = 0; i < vocab_size; i++)
    {
        logits[i] /= sum;
    }

    // Sample from the distribution
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

void generate_text(GPT2_MoE_Model *model, RunState *state, Dataset *dataset,
                   const char *prompt, int max_tokens)
{
    Config *config = &model->config;
    int seq_len = config->seq_len;
    float temperature = 0.8f;

    // Tokenize prompt
    int prompt_len = strlen(prompt);
    int *tokens = malloc(seq_len * sizeof(int));
    memset(tokens, 0, seq_len * sizeof(int));

    // Simple character-level tokenization
    for (int i = 0; i < prompt_len && i < seq_len; i++)
    {
        char c = prompt[i];
        for (int v = 0; v < dataset->vocab_size; v++)
        {
            if (dataset->vocab[v][0] == c)
            {
                tokens[i] = v;
                break;
            }
        }
    }

    printf("Prompt: \"%s\"\n", prompt);
    printf("Generated: \"");

    // Rolling window approach - track position instead of shifting
    int current_pos = prompt_len;

    // Generate tokens
    for (int gen = 0; gen < max_tokens; gen++)
    {
        // Forward pass
        forward_pass(model, state, tokens);

        // Get logits for the last valid position
        int pred_pos = (current_pos - 1) % seq_len;
        if (pred_pos < 0)
            pred_pos = seq_len - 1;

        float *last_logits = &state->logits[pred_pos * config->vocab_size];

        // Sample next token
        int next_token = sample_from_logits(last_logits, config->vocab_size, temperature);

        // Print the generated character
        if (next_token < dataset->vocab_size)
        {
            printf("%c", dataset->vocab[next_token][0]);
            fflush(stdout);
        }

        // Update rolling window - overwrite oldest position
        tokens[current_pos % seq_len] = next_token;
        current_pos++;
    }

    printf("\"\n\n");
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
    int total_tokens = num_steps * config->seq_len;

    // Count how many times each expert was used
    for (int step = 0; step < num_steps; step++)
    {
        for (int t = 0; t < config->seq_len; t++)
        {
            for (int k = 0; k < config->top_k; k++)
            {
                int idx = (step * config->seq_len + t) * config->top_k + k;
                if (idx < total_tokens * config->top_k)
                {
                    int expert_idx = state->expert_indices[idx];
                    if (expert_idx >= 0 && expert_idx < config->num_experts)
                    {
                        expert_counts[expert_idx]++;
                    }
                }
            }
        }
    }

    printf("Expert usage distribution:\n");
    for (int e = 0; e < config->num_experts; e++)
    {
        float usage_percent = (float)expert_counts[e] / (total_tokens * config->top_k) * 100.0f;
        printf("Expert %d: %d uses (%.1f%%)\n", e, expert_counts[e], usage_percent);
    }

    free(expert_counts);
}