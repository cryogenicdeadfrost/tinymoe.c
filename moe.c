#include <stdio.h>
#include <cblas.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>
#include <sys/time.h>
#define DEBUG 0

// ============================================================================
// ## 1. ARCHITECTURE DEFINITION
// ============================================================================

typedef enum
{
    DATASET_TINYSHAKESPEARE,
    DATASET_TINYSTORIES,
    DATASET_CUSTOM
} DatasetType;

typedef struct
{
    float repetition_penalty;
    int repetition_window;
    float length_penalty;
    int max_repetitions;
    int min_tokens;
    int stop_on_repeat;
} GenerationConfig;

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

typedef struct
{
    float *losses;
    float *val_losses;
    float *perplexities;
    int *steps;
    int num_records;
    int capacity;
} TrainingHistory;

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

typedef struct
{
    char timestamp[32];
    float best_loss;
    int training_steps;
    DatasetType dataset_type;
    Config config;
    float validation_loss;
    float perplexity;
} ModelMetadata;


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
    char **vocab;
    int vocab_size;
    DatasetType type;
    char *name;

    // Training/validation split
    int train_tokens;
    int val_tokens;
    int *train_data;
    int *val_data;

    // Tokenization type
    int use_word_level; // 0 for char, 1 for word
} Dataset;

GenerationConfig create_generation_config(void);
TrainingHistory *create_training_history(int capacity);
char *generate_model_filename(const char *base_name, int step, float loss);
void save_model_with_metadata(GPT2_MoE_Model *model, const char *filename, ModelMetadata *metadata);
int load_model_with_metadata(GPT2_MoE_Model *model, const char *filename, ModelMetadata *metadata);
void free_model(GPT2_MoE_Model *model);
void free_state(RunState *state, Config *config);
void free_dataset(Dataset *dataset);
void free_optimizer(Optimizer *opt, Config *config);
float validate_model(GPT2_MoE_Model *model, RunState *state, Dataset *dataset, Config *config);
float calculate_perplexity(float *logits, int *targets, Config *config, int normalize);
float get_adaptive_learning_rate(int step, int warmup, int total, float max_lr, float min_lr);
void generate_training_sample(GPT2_MoE_Model *model, RunState *state, Dataset *dataset, Config *config, int step);
void record_training_step(TrainingHistory *history, int step, float loss, float val_loss, float perplexity);

// ============================================================================
// ## FORWARD DECLARATIONS
// ============================================================================
void analyze_expert_usage(RunState *state, Config *config, int num_steps);
void generate_text_enhanced(GPT2_MoE_Model *model, RunState *state, Dataset *dataset,
                            const char *prompt, int max_tokens, float temperature,
                            float top_p, GenerationConfig *gen_config);
int load_model(GPT2_MoE_Model *model, const char *filename);
int load_model_with_metadata(GPT2_MoE_Model *model, const char *filename,
                             ModelMetadata *metadata);
void save_model(GPT2_MoE_Model *model, const char *filename);
void generate_text(GPT2_MoE_Model *model, RunState *state, Dataset *dataset,
                   const char *prompt, int max_tokens, float temperature, float top_p)
{
    GenerationConfig gen_config = create_generation_config();
    generate_text_enhanced(model, state, dataset, prompt, max_tokens, temperature, top_p, &gen_config);
}
void record_training_step(TrainingHistory *history, int step, float loss, float val_loss, float perplexity);
void free_state(RunState *state, Config *config);
void free_dataset(Dataset *dataset);
void free_optimizer(Optimizer *opt, Config *config);

// ============================================================================
// ## 2. UTILITY FUNCTIONS
// ============================================================================

typedef struct
{
    float prob;
    int index;
} ProbIndex;
static int cmp_prob_desc(const void *a, const void *b)
{
    float pa = ((const ProbIndex *)a)->prob;
    float pb = ((const ProbIndex *)b)->prob;
    return (pa > pb) ? -1 : (pa < pb);
}

char *generate_model_filename(const char *base_name, int step, float loss)
{
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);

    char *filename = malloc(256);
    snprintf(filename, 256, "%s_step%d_loss%.4f_%04d%02d%02d_%02d%02d%02d.bin",
             base_name,
             step,
             loss,
             tm_info->tm_year + 1900,
             tm_info->tm_mon + 1,
             tm_info->tm_mday,
             tm_info->tm_hour,
             tm_info->tm_min,
             tm_info->tm_sec);

    return filename;
}

GenerationConfig create_generation_config()
{
    GenerationConfig config;
    config.repetition_penalty = 1.1f;
    config.repetition_window = 20;
    config.length_penalty = 1.0f;
    config.max_repetitions = 3;
    config.min_tokens = 10;
    config.stop_on_repeat = 1;
    return config;
}

void download_dataset(const char *url, const char *filename)
{
    char command[512];
    snprintf(command, sizeof(command), "curl -o %s %s", filename, url);
    printf("Downloading %s...\n", filename);
    if (system(command) != 0)
    {
        printf("Failed to download. Trying wget...\n");
        snprintf(command, sizeof(command), "wget -O %s %s", filename, url);
        if (system(command) != 0)
        {
            printf("Download failed. Please download manually.\n");
            exit(1);
        }
    }
    printf("✓ Downloaded %s\n", filename);
}

int file_exists(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file)
    {
        fclose(file);
        return 1;
    }
    return 0;
}

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

    /* 1.  Global max (single-threaded is faster for small arrays) */
    float max_val = x[0];
    for (int i = 1; i < size; ++i)
        if (x[i] > max_val)
            max_val = x[i];

    /* 2.  Clamp inputs *before* expf */
    float sum = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        float z = fminf(fmaxf(x[i] - max_val, -50.0f), 50.0f); /* clamp */
        float e = expf(z);
        x[i] = e;
        sum += e;
    }

    /* 3.  Normalize safely */
    if (sum < 1e-8f)
    {
        float uniform = 1.0f / size;
        for (int i = 0; i < size; ++i)
            x[i] = uniform;
    }
    else
    {
        for (int i = 0; i < size; ++i)
            x[i] /= sum;
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
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n, k, d, 1.0f, a, d, b, d, 0.0f, c, k);
}

TrainingHistory *create_training_history(int capacity)
{
    TrainingHistory *history = malloc(sizeof(TrainingHistory));
    if (!history)
    {
        fprintf(stderr, "Failed to allocate TrainingHistory\n");
        exit(1);
    }
    history->losses = malloc(capacity * sizeof(float));
    history->val_losses = malloc(capacity * sizeof(float));
    history->perplexities = malloc(capacity * sizeof(float));
    history->steps = malloc(capacity * sizeof(int));
    if (!history->losses || !history->val_losses || !history->perplexities || !history->steps)
    {
        fprintf(stderr, "Failed to allocate TrainingHistory buffers\n");
        exit(1);
    }
    history->num_records = 0;
    history->capacity = capacity;
    return history;
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
    int seq_len = config->seq_len;
    int embed_dim = config->embed_dim;
    int num_layers = config->num_layers;
    int num_experts = config->num_experts;
    int top_k = config->top_k;
    int hidden_dim = config->hidden_dim;

    /* -----  forward & backward buffers allocated once  ----- */
    state->embedding_out = calloc(seq_len * embed_dim, sizeof(float));
    state->layer_outputs = calloc(num_layers * seq_len * embed_dim, sizeof(float));
    state->attn_out = calloc(seq_len * embed_dim, sizeof(float));
    state->moe_out = calloc(seq_len * embed_dim, sizeof(float));
    state->logits = calloc(seq_len * config->vocab_size, sizeof(float));
    state->gating_logits = calloc(num_layers * seq_len * num_experts, sizeof(float));
    state->expert_indices = calloc(num_layers * seq_len * top_k, sizeof(int));
    state->expert_weights = calloc(num_layers * seq_len * top_k, sizeof(float));
    state->expert_outputs = calloc(num_layers * num_experts * seq_len * embed_dim, sizeof(float));
    state->temp_buffer = calloc(seq_len * embed_dim, sizeof(float));
    state->temp_buffer2 = calloc(seq_len * embed_dim, sizeof(float));
    state->gating_scores_buffer = calloc(num_experts, sizeof(float));
    state->hidden_buffer = calloc(hidden_dim, sizeof(float));
    state->expert_out_buffer = calloc(embed_dim, sizeof(float));
    state->attention_scores_buffer = calloc(seq_len * seq_len, sizeof(float));
    state->qkv_buffer = calloc(seq_len * 3 * embed_dim, sizeof(float));
    state->ln1_outputs = calloc(num_layers * seq_len * embed_dim, sizeof(float));
    state->ln2_outputs = calloc(num_layers * seq_len * embed_dim, sizeof(float));
    state->attn_outputs = calloc(num_layers * seq_len * embed_dim, sizeof(float));
    state->attn_residual = calloc(num_layers * seq_len * embed_dim, sizeof(float));
    state->moe_outputs = calloc(num_layers * seq_len * embed_dim, sizeof(float));
    state->gating_scores = calloc(num_layers * seq_len * num_experts, sizeof(float));
    state->expert_hiddens = calloc(num_layers * num_experts * seq_len * hidden_dim, sizeof(float));
    state->attn_concat = calloc(num_layers * seq_len * embed_dim, sizeof(float));
    state->attention_probs = calloc(num_layers * seq_len * seq_len, sizeof(float));

    /* -----  gradient buffers allocated once  ----- */
    state->d_embedding_out = calloc(seq_len * embed_dim, sizeof(float));
    state->d_layer_outputs = calloc(num_layers * seq_len * embed_dim, sizeof(float));
    state->d_attn_out = calloc(seq_len * embed_dim, sizeof(float));
    state->d_moe_out = calloc(seq_len * embed_dim, sizeof(float));
    state->d_temp_buffer = calloc(seq_len * embed_dim, sizeof(float));
    state->d_temp_buffer2 = calloc(seq_len * embed_dim, sizeof(float));
    state->d_token_embeddings = calloc(config->vocab_size * embed_dim, sizeof(float));
    state->d_pos_embeddings = calloc(seq_len * embed_dim, sizeof(float));
    state->d_final_ln_gamma = calloc(embed_dim, sizeof(float));
    state->d_final_ln_beta = calloc(embed_dim, sizeof(float));

    /* -----  per-layer gradient arrays  ----- */
    state->d_attn_qkv_w = calloc(num_layers, sizeof(float *));
    state->d_attn_qkv_b = calloc(num_layers, sizeof(float *));
    state->d_attn_proj_w = calloc(num_layers, sizeof(float *));
    state->d_attn_proj_b = calloc(num_layers, sizeof(float *));
    state->d_ln1_gamma = calloc(num_layers, sizeof(float *));
    state->d_ln1_beta = calloc(num_layers, sizeof(float *));
    state->d_ln2_gamma = calloc(num_layers, sizeof(float *));
    state->d_ln2_beta = calloc(num_layers, sizeof(float *));
    state->d_gating_w = calloc(num_layers, sizeof(float *));
    state->d_gating_b = calloc(num_layers, sizeof(float *));
    state->d_expert_w1 = calloc(num_layers, sizeof(float **));
    state->d_expert_b1 = calloc(num_layers, sizeof(float **));
    state->d_expert_w2 = calloc(num_layers, sizeof(float **));
    state->d_expert_b2 = calloc(num_layers, sizeof(float **));

    /* -----  scratch buffers reused by every layer  ----- */
    state->d_ln1_out = calloc(seq_len * embed_dim, sizeof(float));
    state->d_ln2_out = calloc(seq_len * embed_dim, sizeof(float));
    state->d_attn_residual_out = calloc(seq_len * embed_dim, sizeof(float));
    state->d_attn_concat = calloc(seq_len * embed_dim, sizeof(float));
    state->d_attention_scores = calloc(seq_len * seq_len, sizeof(float));
    state->d_query = calloc(seq_len * (embed_dim / config->num_heads), sizeof(float));
    state->d_key = calloc(seq_len * (embed_dim / config->num_heads), sizeof(float));
    state->d_value = calloc(seq_len * (embed_dim / config->num_heads), sizeof(float));
    state->d_softmax_scores = calloc(seq_len * seq_len, sizeof(float));

    /* -----  allocate per-layer parameter gradients  ----- */
    for (int l = 0; l < num_layers; l++)
    {
        state->d_attn_qkv_w[l] = calloc(embed_dim * 3 * embed_dim, sizeof(float));
        state->d_attn_qkv_b[l] = calloc(3 * embed_dim, sizeof(float));
        state->d_attn_proj_w[l] = calloc(embed_dim * embed_dim, sizeof(float));
        state->d_attn_proj_b[l] = calloc(embed_dim, sizeof(float));
        state->d_ln1_gamma[l] = calloc(embed_dim, sizeof(float));
        state->d_ln1_beta[l] = calloc(embed_dim, sizeof(float));
        state->d_ln2_gamma[l] = calloc(embed_dim, sizeof(float));
        state->d_ln2_beta[l] = calloc(embed_dim, sizeof(float));
        state->d_gating_w[l] = calloc(embed_dim * num_experts, sizeof(float));
        state->d_gating_b[l] = calloc(num_experts, sizeof(float));

        state->d_expert_w1[l] = calloc(num_experts, sizeof(float *));
        state->d_expert_b1[l] = calloc(num_experts, sizeof(float *));
        state->d_expert_w2[l] = calloc(num_experts, sizeof(float *));
        state->d_expert_b2[l] = calloc(num_experts, sizeof(float *));

        for (int e = 0; e < num_experts; e++)
        {
            state->d_expert_w1[l][e] = calloc(embed_dim * hidden_dim, sizeof(float));
            state->d_expert_b1[l][e] = calloc(hidden_dim, sizeof(float));
            state->d_expert_w2[l][e] = calloc(hidden_dim * embed_dim, sizeof(float));
            state->d_expert_b2[l][e] = calloc(embed_dim, sizeof(float));
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

void multi_head_attention(float *out, float *x, TransformerBlock *layer,
                          Config *config, RunState *state, int layer_idx)
{
    if (DEBUG)
        printf("  Multi-head attention\n");

    int seq_len = config->seq_len;
    int embed_dim = config->embed_dim;
    int num_heads = config->num_heads;
    int head_dim = embed_dim / num_heads;

    float *qkv = state->qkv_buffer;
    memset(qkv, 0, seq_len * 3 * embed_dim * sizeof(float));

    /* ----  FUSED: LayerNorm + QKV projection  ---- */
#pragma omp parallel for
    for (int t = 0; t < seq_len; t++)
    {
        /* LayerNorm in-register */
        float mean = 0.0f, var = 0.0f;
        for (int d = 0; d < embed_dim; d++)
            mean += x[t * embed_dim + d];
        mean /= embed_dim;
        for (int d = 0; d < embed_dim; d++)
        {
            float diff = x[t * embed_dim + d] - mean;
            var += diff * diff;
        }
        var /= embed_dim;
        float rstd = 1.0f / sqrtf(var + 1e-5f);

        /* QKV projection using normalized x */
        for (int d = 0; d < 3 * embed_dim; d++)
        {
            float sum = layer->attn_qkv_b[d];
            for (int e = 0; e < embed_dim; e++)
            {
                float val = (x[t * embed_dim + e] - mean) * rstd *
                                layer->ln1_gamma[e] +
                            layer->ln1_beta[e];
                sum += val * layer->attn_qkv_w[e * 3 * embed_dim + d];
            }
            qkv[t * 3 * embed_dim + d] = sum;
        }
    }

    /* ----  rest of attention unchanged  ---- */
    memset(out, 0, seq_len * embed_dim * sizeof(float));

    for (int h = 0; h < num_heads; h++)
    {
        for (int i = 0; i < seq_len; i++)
        {
            for (int j = 0; j < seq_len; j++)
            {
                if (j > i)
                {
                    state->attention_scores_buffer[i * seq_len + j] = -1e9f;
                }
                else
                {
                    float score = 0.0f;
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
            float *row = &state->attention_scores_buffer[i * seq_len];
            softmax(row, seq_len);
            int prob_offset = (layer_idx * seq_len * seq_len) + (i * seq_len);
            memcpy(&state->attention_probs[prob_offset], row, seq_len * sizeof(float));

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

    /* output projection (unchanged) */
    for (int t = 0; t < seq_len; t++)
    {
        for (int d = 0; d < embed_dim; d++)
        {
            out[t * embed_dim + d] = layer->attn_proj_b[d];
            for (int e = 0; e < embed_dim; e++)
                out[t * embed_dim + d] +=
                    state->temp_buffer[t * embed_dim + e] * layer->attn_proj_w[e * embed_dim + d];
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
                /* ----------  exact Jacobian-vector product for softmax  ---------- */
                float *probs = state->gating_scores_buffer; // already softmax-normalised
                float sum = 0.0f;
                for (int e = 0; e < config->num_experts; e++)
                    sum += probs[e] * d_weight;

                for (int e = 0; e < config->num_experts; e++)
                {
                    float grad = probs[e] * (d_weight - sum);
                    state->d_gating_b[l][e] += grad;
                    for (int d2 = 0; d2 < embed_dim; d2++)
                    {
                        state->d_gating_w[l][e * embed_dim + d2] += grad * ln2_token_out[d2];
                        d_ln2_out[t * embed_dim + d2] += grad * layer->moe_layer.gating_w[e * embed_dim + d2];
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

        // 4e. Backward through Multi-Head Attention (Jacobian-vector product)
        float *d_attn_concat = state->d_temp_buffer; // gradient w.r.t concatenated heads
        float *d_ln1_out2 = state->d_temp_buffer2;   // gradient w.r.t ln1 output
        memset(d_attn_concat, 0, seq_len * embed_dim * sizeof(float));
        memset(d_ln1_out2, 0, seq_len * embed_dim * sizeof(float));

        // --- 1. Gradient through output projection
        for (int t = 0; t < seq_len; t++)
        {
            for (int d = 0; d < embed_dim; d++)
            {
                state->d_attn_proj_b[l][d] += d_attn_out[t * embed_dim + d];
                for (int e = 0; e < embed_dim; e++)
                {
                    state->d_attn_proj_w[l][e * embed_dim + d] +=
                        d_attn_out[t * embed_dim + d] * state->attn_concat[t * embed_dim + e];
                    d_attn_concat[t * embed_dim + e] +=
                        d_attn_out[t * embed_dim + d] * layer->attn_proj_w[e * embed_dim + d];
                }
            }
        }

        // --- 2. Split gradient across heads
        for (int h = 0; h < num_heads; h++)
        {
            int head_offset = h * head_dim;

            // --- 3. Compute gradient w.r.t attention probabilities (d_probs)
            float *d_probs = state->d_temp_buffer2; // seq_len * seq_len
            memset(d_probs, 0, seq_len * seq_len * sizeof(float));

            for (int i = 0; i < seq_len; i++)
            {
                float *probs = &state->attention_probs[(l * seq_len + i) * seq_len];
                for (int j = 0; j <= i; j++)
                {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++)
                    {
                        int concat_idx = i * embed_dim + head_offset + d;
                        float v_j = state->qkv_buffer[j * 3 * embed_dim + 2 * embed_dim + head_offset + d];
                        dot += d_attn_concat[concat_idx] * v_j;
                    }
                    d_probs[i * seq_len + j] = dot;
                }
            }

            // --- 4. Backward through softmax (Jacobian-vector product)
            float *d_scores = state->d_temp_buffer; // seq_len * seq_len
            memset(d_scores, 0, seq_len * seq_len * sizeof(float));

            for (int i = 0; i < seq_len; i++)
            {
                float *probs = &state->attention_probs[(l * seq_len + i) * seq_len];
                float sum = 0.0f;
                for (int j = 0; j <= i; j++)
                {
                    sum += probs[j] * d_probs[i * seq_len + j];
                }
                for (int j = 0; j <= i; j++)
                {
                    d_scores[i * seq_len + j] = probs[j] * (d_probs[i * seq_len + j] - sum);
                }
            }

            // --- 5. Backward through Q*K^T / scale
            float scale = 1.0f / sqrtf((float)head_dim);
            float *d_query = state->d_temp_buffer2;
            float *d_key = state->d_moe_out;
            memset(d_query, 0, seq_len * head_dim * sizeof(float));
            memset(d_key, 0, seq_len * head_dim * sizeof(float));

            for (int i = 0; i < seq_len; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    float d_s = d_scores[i * seq_len + j] * scale;
                    for (int d = 0; d < head_dim; d++)
                    {
                        float qi = state->qkv_buffer[i * 3 * embed_dim + head_offset + d];
                        float kj = state->qkv_buffer[j * 3 * embed_dim + embed_dim + head_offset + d];
                        d_query[i * head_dim + d] += d_s * kj;
                        d_key[j * head_dim + d] += d_s * qi;
                    }
                }
            }

            // --- 6. Backward through V
            float *d_value = state->d_temp_buffer;
            memset(d_value, 0, seq_len * head_dim * sizeof(float));
            for (int i = 0; i < seq_len; i++)
            {
                float *probs = &state->attention_probs[(l * seq_len + i) * seq_len];
                for (int j = 0; j <= i; j++)
                {
                    for (int d = 0; d < head_dim; d++)
                    {
                        int concat_idx = i * embed_dim + head_offset + d;
                        d_value[j * head_dim + d] += probs[j] * d_attn_concat[concat_idx];
                    }
                }
            }

            // --- 7. Backward through QKV projection
            for (int t = 0; t < seq_len; t++)
            {
                for (int d = 0; d < head_dim; d++)
                {
                    // biases
                    state->d_attn_qkv_b[l][head_offset + d] += d_query[t * head_dim + d];
                    state->d_attn_qkv_b[l][embed_dim + head_offset + d] += d_key[t * head_dim + d];
                    state->d_attn_qkv_b[l][2 * embed_dim + head_offset + d] += d_value[t * head_dim + d];

                    // weights & input
                    for (int e = 0; e < embed_dim; e++)
                    {
                        float x = state->ln1_outputs[l * seq_len * embed_dim + t * embed_dim + e];

                        state->d_attn_qkv_w[l][e * 3 * embed_dim + head_offset + d] +=
                            d_query[t * head_dim + d] * x;
                        state->d_attn_qkv_w[l][e * 3 * embed_dim + embed_dim + head_offset + d] +=
                            d_key[t * head_dim + d] * x;
                        state->d_attn_qkv_w[l][e * 3 * embed_dim + 2 * embed_dim + head_offset + d] +=
                            d_value[t * head_dim + d] * x;

                        d_ln1_out2[t * embed_dim + e] +=
                            d_query[t * head_dim + d] * layer->attn_qkv_w[e * 3 * embed_dim + head_offset + d] +
                            d_key[t * head_dim + d] * layer->attn_qkv_w[e * 3 * embed_dim + embed_dim + head_offset + d] +
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

void record_training_step(TrainingHistory *history, int step, float loss, float val_loss, float perplexity)
{
    if (history->num_records < history->capacity)
    {
        history->steps[history->num_records] = step;
        history->losses[history->num_records] = loss;
        history->val_losses[history->num_records] = val_loss;
        history->perplexities[history->num_records] = perplexity;
        history->num_records++;
    }
}

static char *preprocess_text(const char *text)
{
    int len = strlen(text);
    char *out = malloc(len * 3 + 1); // worst-case expansion
    int j = 0;

    for (int i = 0; i < len; i++)
    {
        unsigned char c = text[i];

        if (isupper(c))
        {
            out[j++] = (char)tolower(c);
        }
        else if (ispunct(c))
        {
            if (j > 0 && out[j - 1] != ' ')
                out[j++] = ' ';
            out[j++] = c;
            out[j++] = ' ';
        }
        else if (isspace(c))
        {
            if (j > 0 && out[j - 1] != ' ')
                out[j++] = ' ';
        }
        else
        { // digits or other chars
            out[j++] = c;
        }
    }
    out[j] = '\0';
    return out; // caller frees
}

void build_char_vocabulary(Dataset *dataset)
{
    // Character-level vocabulary
    int char_count[256] = {0};
    for (int i = 0; i < dataset->size; i++)
    {
        char_count[(unsigned char)dataset->data[i]]++;
    }

    dataset->vocab_size = 0;
    dataset->vocab = malloc(256 * sizeof(char *));

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

void build_word_vocabulary(Dataset *dataset)
{
    // Simple word-level vocabulary (space-separated)
    // This is a basic implementation - could be enhanced with BPE

    int max_vocab = 10000;
    dataset->vocab = malloc(max_vocab * sizeof(char *));
    int *word_counts = calloc(max_vocab, sizeof(int));
    dataset->vocab_size = 0;

    // Add special tokens
    dataset->vocab[dataset->vocab_size++] = strdup("<UNK>");
    dataset->vocab[dataset->vocab_size++] = strdup("<PAD>");

    /* ----------  NEW PRE-PROCESSING  ---------- */
    char *processed = preprocess_text(dataset->data); // malloc'd
    free(dataset->data);                              // free old buffer
    dataset->data = processed;                        // adopt new buffer
    /* ------------------------------------------ */

    // Tokenize by spaces and build vocab
    char *data_copy = strdup(dataset->data);
    char *token = strtok(data_copy, " \n\t\r");

    // First pass: build vocabulary
    while (token != NULL && dataset->vocab_size < max_vocab - 1)
    {
        // Check if token already exists
        int found = -1;
        for (int i = 0; i < dataset->vocab_size; i++)
        {
            if (strcmp(dataset->vocab[i], token) == 0)
            {
                found = i;
                break;
            }
        }

        if (found >= 0)
        {
            word_counts[found]++;
        }
        else
        {
            dataset->vocab[dataset->vocab_size] = strdup(token);
            word_counts[dataset->vocab_size] = 1;
            dataset->vocab_size++;
        }

        token = strtok(NULL, " \n\t\r");
    }

    free(data_copy);

    // Second pass: tokenize
    data_copy = strdup(dataset->data);
    token = strtok(data_copy, " \n\t\r");

    // Count tokens first
    int token_count = 0;
    char *temp_copy = strdup(dataset->data);
    char *temp_token = strtok(temp_copy, " \n\t\r");
    while (temp_token != NULL)
    {
        token_count++;
        temp_token = strtok(NULL, " \n\t\r");
    }
    free(temp_copy);

    dataset->num_tokens = token_count;
    dataset->tokens = malloc(dataset->num_tokens * sizeof(int));

    int token_idx = 0;
    while (token != NULL && token_idx < dataset->num_tokens)
    {
        int found = 0; // UNK token
        for (int i = 0; i < dataset->vocab_size; i++)
        {
            if (strcmp(dataset->vocab[i], token) == 0)
            {
                found = i;
                break;
            }
        }
        dataset->tokens[token_idx++] = found;
        token = strtok(NULL, " \n\t\r");
    }

    free(data_copy);
    free(word_counts);
}

void load_dataset(Dataset *dataset, DatasetType type, const char *custom_path)
{
    const char *filename;
    const char *url;

    switch (type)
    {
    case DATASET_TINYSHAKESPEARE:
        filename = "tinyshakespeare.txt";
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";
        dataset->name = strdup("TinyShakespeare");
        dataset->use_word_level = 0; // Character level for Shakespeare
        break;

    case DATASET_TINYSTORIES:
        filename = "tinystories.txt";
        url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt";
        dataset->name = strdup("TinyStories");
        dataset->use_word_level = 1; // Word level for stories
        break;

    case DATASET_CUSTOM:
        filename = custom_path;
        url = NULL;
        dataset->name = strdup("Custom");
        dataset->use_word_level = 0; // Default to char level
        break;
    }

    dataset->type = type;

    // Download if doesn't exist (except for custom)
    if (type != DATASET_CUSTOM && !file_exists(filename))
    {
        download_dataset(url, filename);
    }

    // Load file
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        printf("Error: Could not open %s\n", filename);
        if (type == DATASET_TINYSHAKESPEARE)
        {
            printf("Please download from: %s\n", url);
        }
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

    printf("✓ Loaded %s: %d characters\n", dataset->name, dataset->size);

    // Build vocabulary and tokenize
    if (dataset->use_word_level)
    {
        build_word_vocabulary(dataset);
    }
    else
    {
        build_char_vocabulary(dataset);
    }

    // Create train/val split (90/10)
    dataset->train_tokens = (int)(dataset->num_tokens * 0.9);
    dataset->val_tokens = dataset->num_tokens - dataset->train_tokens;

    dataset->train_data = dataset->tokens;
    dataset->val_data = &dataset->tokens[dataset->train_tokens];

    printf("✓ Vocabulary size: %d\n", dataset->vocab_size);
    printf("✓ Train tokens: %d, Validation tokens: %d\n",
           dataset->train_tokens, dataset->val_tokens);
}

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

float validate_model(GPT2_MoE_Model *model, RunState *state, Dataset *dataset, Config *config)
{
    int val_steps = 10;
    float total_loss = 0.0f;
    for (int i = 0; i < val_steps; i++)
    {
        int start = rand() % (dataset->val_tokens - config->seq_len - 1);
        int *inputs = &dataset->val_data[start];
        int *targets = &dataset->val_data[start + 1];
        forward_pass(model, state, inputs);
        float loss = calculate_crossentropy_loss(state->logits, targets, config);
        total_loss += loss;
    }
    return total_loss / val_steps;
}

float get_adaptive_learning_rate(int step, int warmup, int total, float max_lr, float min_lr)
{
    if (step < warmup)
    {
        return max_lr * (step / (float)warmup);
    }
    float progress = (step - warmup) / (float)(total - warmup);
    return max_lr - (max_lr - min_lr) * progress;
}

float calculate_perplexity(float *logits, int *targets, Config *config, int normalize)
{
    float loss = calculate_crossentropy_loss(logits, targets, config);
    return expf(loss);
}

void generate_training_sample(GPT2_MoE_Model *model, RunState *state, Dataset *dataset, Config *config, int step)
{
    const char *prompt = (dataset->type == DATASET_TINYSHAKESPEARE) ? "To be or not to be" : "Once upon a time";
    GenerationConfig gen_config = create_generation_config();
    generate_text_enhanced(model, state, dataset, prompt, 50, 0.8f, 0.9f, &gen_config);
}

// ============================================================================
// ## 7. MAIN TRAINING LOOP
// ============================================================================

int min(int a, int b)
{
    return (a < b) ? a : b;
}

void print_usage(const char *program_name)
{
    printf("=== GPT-2 MoE Transformer - Enhanced Edition ===\n\n");
    printf("Usage: %s [MODE] [OPTIONS]\n\n", program_name);

    printf("MODES:\n");
    printf("  train          Train a new model (default)\n");
    printf("  generate       Generate text using trained model\n");
    printf("  analyze        Analyze expert usage and model statistics\n\n");

    printf("DATASET OPTIONS:\n");
    printf("  --dataset shakespeare    Use TinyShakespeare dataset (default)\n");
    printf("  --dataset stories        Use TinyStories dataset\n");
    printf("  --dataset <path>         Use custom dataset file\n\n");

    printf("GENERATION OPTIONS:\n");
    printf("  --prompt \"text\"          Starting prompt for generation\n");
    printf("  --length N               Number of tokens to generate (default: 100)\n");
    printf("  --temperature T          Sampling temperature 0.1-2.0 (default: 0.8)\n");
    printf("  --top_p P                Nucleus sampling threshold 0.1-1.0 (default: 0.9)\n\n");

    printf("EXAMPLES:\n");
    printf("  # Train on Shakespeare\n");
    printf("  %s train --dataset shakespeare\n\n", program_name);

    printf("  # Train on TinyStories\n");
    printf("  %s train --dataset stories\n\n", program_name);

    printf("  # Generate Shakespearean text\n");
    printf("  %s generate --dataset shakespeare --prompt \"To be or not to be\" --length 200\n\n", program_name);

    printf("  # Generate a story\n");
    printf("  %s generate --dataset stories --prompt \"Once upon a time\" --temperature 0.7\n\n", program_name);

    printf("  # Analyze model\n");
    printf("  %s analyze --dataset shakespeare\n\n", program_name);

    printf("NOTES:\n");
    printf("  - Models are automatically saved with timestamps and loss values\n");
    printf("  - Best models are saved as 'moe_model_best.bin'\n");
    printf("  - Training includes validation and early stopping\n");
    printf("  - Generation includes repetition penalty and quality control\n");
    printf("  - Datasets are automatically downloaded if not present\n\n");
}

int main(int argc, char *argv[])
{
    srand(time(NULL));
    printf("=== GPT-2 MoE Transformer - Enhanced Edition ===\n");

    // Parse command line arguments for dataset selection
    DatasetType dataset_type = DATASET_TINYSHAKESPEARE;
    const char *custom_dataset_path = NULL;

    // Enhanced CLI parsing
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--dataset") == 0 && i + 1 < argc)
        {
            if (strcmp(argv[i + 1], "shakespeare") == 0)
            {
                dataset_type = DATASET_TINYSHAKESPEARE;
            }
            else if (strcmp(argv[i + 1], "stories") == 0)
            {
                dataset_type = DATASET_TINYSTORIES;
            }
            else
            {
                dataset_type = DATASET_CUSTOM;
                custom_dataset_path = argv[i + 1];
            }
            i++;
        }
    }

    // Handle different modes
    if (argc > 1)
    {
        if (strcmp(argv[1], "generate") == 0)
        {
            printf("[MODE] Enhanced Text Generation\n");

            GPT2_MoE_Model model;
            RunState state;
            ModelMetadata metadata;

            // Try to load the best model
            if (!load_model_with_metadata(&model, "moe_model_best.bin", &metadata))
            {
                printf("No best model found. Trying latest...\n");
                if (!load_model(&model, "moe_model.bin"))
                {
                    printf("ERROR: No trained model found. Please train first.\n");
                    return 1;
                }
            }

            build_state(&state, &model.config);

            Dataset dataset;
            load_dataset(&dataset, dataset_type, custom_dataset_path);

            // Parse generation parameters
            const char *prompt = "To be or not to be";
            int max_tokens = 100;
            float temperature = 0.8f;
            float top_p = 0.9f;

            for (int i = 2; i < argc; i++)
            {
                if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc)
                {
                    prompt = argv[i + 1];
                    i++;
                }
                else if (strcmp(argv[i], "--length") == 0 && i + 1 < argc)
                {
                    max_tokens = atoi(argv[i + 1]);
                    i++;
                }
                else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc)
                {
                    temperature = atof(argv[i + 1]);
                    i++;
                }
                else if (strcmp(argv[i], "--top_p") == 0 && i + 1 < argc)
                {
                    top_p = atof(argv[i + 1]);
                    i++;
                }
            }

            GenerationConfig gen_config = create_generation_config();
            generate_text_enhanced(&model, &state, &dataset, prompt, max_tokens,
                                   temperature, top_p, &gen_config);

            free_state(&state, &model.config);
            free_model(&model);
            free_dataset(&dataset);
            return 0;
        }
        else if (strcmp(argv[1], "analyze") == 0)
        {
            printf("[MODE] Enhanced Expert Analysis\n");

            GPT2_MoE_Model model;
            RunState state;
            ModelMetadata metadata;

            if (!load_model_with_metadata(&model, "moe_model_best.bin", &metadata))
            {
                if (!load_model(&model, "moe_model.bin"))
                {
                    printf("ERROR: No trained model found.\n");
                    return 1;
                }
            }

            build_state(&state, &model.config);

            Dataset dataset;
            load_dataset(&dataset, dataset_type, custom_dataset_path);

            // Comprehensive analysis
            printf("\n=== Model Analysis ===\n");
            printf("Training metadata:\n");
            printf("  Steps: %d\n", metadata.training_steps);
            printf("  Best loss: %.4f\n", metadata.best_loss);
            printf("  Validation loss: %.4f\n", metadata.validation_loss);
            printf("  Perplexity: %.2f\n", metadata.perplexity);

            // Run multiple forward passes for analysis
            for (int i = 0; i < 10; i++)
            {
                int max_start = dataset.train_tokens - model.config.seq_len - 1;
                if (max_start > 0)
                {
                    int start_pos = rand() % max_start;
                    forward_pass(&model, &state, &dataset.tokens[start_pos]);
                }
            }

            analyze_expert_usage(&state, &model.config, 10);

            free_state(&state, &model.config);
            free_model(&model);
            free_dataset(&dataset);
            return 0;
        }
        else if (strcmp(argv[1], "train") != 0)
        {
            printf("Usage: %s [train|generate|analyze] [options]\n", argv[0]);
            printf("\nDataset Options:\n");
            printf("  --dataset shakespeare    Use TinyShakespeare (default)\n");
            printf("  --dataset stories        Use TinyStories\n");
            printf("  --dataset <path>         Use custom dataset\n");
            printf("\nGeneration Options:\n");
            printf("  --prompt \"text\"          Set prompt\n");
            printf("  --length N               Set generation length\n");
            printf("  --temperature T          Set sampling temperature\n");
            printf("  --top_p P                Set nucleus sampling\n");
            return 1;
        }
    }

    // Enhanced Training Mode
    printf("[MODE] Enhanced Training\n");

    // Load dataset
    Dataset dataset;
    load_dataset(&dataset, dataset_type, custom_dataset_path);

    // Configure model based on dataset
    Config config;
    if (dataset.type == DATASET_TINYSHAKESPEARE)
    {
        config = (Config){
            .vocab_size = dataset.vocab_size,
            .seq_len = 128,   // Longer for Shakespeare
            .embed_dim = 128, // Larger embedding
            .num_layers = 4,  // More layers
            .num_heads = 8,   // More heads
            .num_experts = 8, // More experts
            .top_k = 2,
            .hidden_dim = 256 // Larger hidden dim
        };
    }
    else if (dataset.type == DATASET_TINYSTORIES)
    {
        config = (Config){
            .vocab_size = dataset.vocab_size,
            .seq_len = 64,    // Shorter for simple stories
            .embed_dim = 64,  // Smaller embedding
            .num_layers = 2,  // Fewer layers
            .num_heads = 4,   // Fewer heads
            .num_experts = 4, // Fewer experts
            .top_k = 2,
            .hidden_dim = 128 // Smaller hidden dim
        };
    }
    else
    {
        // Default config for custom datasets
        config = (Config){
            .vocab_size = dataset.vocab_size,
            .seq_len = 64,
            .embed_dim = 64,
            .num_layers = 2,
            .num_heads = 4,
            .num_experts = 4,
            .top_k = 2,
            .hidden_dim = 128};
    }

    printf("\n=== Enhanced Training Configuration ===\n");
    printf("Dataset: %s\n", dataset.name);
    printf("Vocabulary size: %d\n", config.vocab_size);
    printf("Sequence length: %d\n", config.seq_len);
    printf("Embedding dimension: %d\n", config.embed_dim);
    printf("Transformer layers: %d\n", config.num_layers);
    printf("Attention heads: %d\n", config.num_heads);
    printf("MoE experts: %d (top-%d)\n", config.num_experts, config.top_k);
    printf("Expert hidden dimension: %d\n", config.hidden_dim);

    // Initialize model and training components
    GPT2_MoE_Model model;
    RunState state;
    Optimizer optimizer;

    build_model(&model, &config);
    build_state(&state, &config);
    init_optimizer(&optimizer, &config);

    TrainingHistory *history = create_training_history(10000);

    // Enhanced training parameters
    int num_steps = 3000;   // Much more training
    int warmup_steps = 200; // Warmup period
    float max_learning_rate = 0.0002f;
    float min_learning_rate = 0.0001f;
    int validation_interval = 100; // Validate every 100 steps
    int sample_interval = 200;     // Generate samples every 200 steps
    int save_interval = 500;       // Save checkpoint every 500 steps

    float best_val_loss = INFINITY;
    int patience = 1000; // Early stopping patience
    int steps_without_improvement = 0;

    printf("\n=== Enhanced Training Started ===\n");
    printf("Training steps: %d\n", num_steps);
    printf("Warmup steps: %d\n", warmup_steps);
    printf("Learning rate: %.4f -> %.4f\n", max_learning_rate, min_learning_rate);
    printf("Validation interval: %d\n", validation_interval);
    printf("Sample generation interval: %d\n", sample_interval);

    time_t start_time = time(NULL);

    for (int step = 0; step < num_steps; step++)
    {
        // Adaptive learning rate
        float learning_rate = get_adaptive_learning_rate(step, warmup_steps, num_steps,
                                                         max_learning_rate, min_learning_rate);

        // Get training batch
        int max_start = dataset.train_tokens - config.seq_len - 1;
        if (max_start <= 0)
        {
            printf("ERROR: Training data too small\n");
            break;
        }

        int start_pos = rand() % max_start;
        int *inputs = &dataset.train_data[start_pos];
        int *targets = &dataset.train_data[start_pos + 1];

        // Forward pass
        forward_pass(&model, &state, inputs);

        // Calculate loss
        float train_loss = calculate_crossentropy_loss(state.logits, targets, &config);
        float aux_loss = calculate_moe_aux_loss(&state, &config);
        float total_loss = train_loss + 0.01f * aux_loss;

        if (!isfinite(total_loss))
        {
            printf("ERROR: NaN/Inf loss at step %d\n", step);
            break;
        }

        // Backward pass and optimization
        backward_pass(&model, &state, inputs, targets, total_loss);
        update_weights(&model, &state, &optimizer, learning_rate);

        // Progress reporting
        if (step % 1 == 0)
        {
            printf("[%4d/%4d] Loss: %.4f (train: %.4f, aux: %.4f) LR: %.6f\n",
                   step, num_steps, total_loss, train_loss, aux_loss, learning_rate);
        }
        else if (step % 10 == 0)
        {
            putchar('.');
            fflush(stdout);
        }

        // Validation
        float val_loss = INFINITY;
        float perplexity = INFINITY;
        if (step % validation_interval == 0 && step > 0)
        {
            val_loss = validate_model(&model, &state, &dataset, &config);
            perplexity = calculate_perplexity(state.logits, targets, &config, 1);

            printf("  Validation - Loss: %.4f, Perplexity: %.2f\n", val_loss, perplexity);

            // Check for improvement
            if (val_loss < best_val_loss)
            {
                best_val_loss = val_loss;
                steps_without_improvement = 0;

                // Save best model
                ModelMetadata metadata = {0};
                strncpy(metadata.timestamp, "best", sizeof(metadata.timestamp));
                metadata.best_loss = best_val_loss;
                metadata.training_steps = step;
                metadata.dataset_type = dataset.type;
                metadata.config = config;
                metadata.validation_loss = val_loss;
                metadata.perplexity = perplexity;

                save_model_with_metadata(&model, "moe_model_best.bin", &metadata);
                printf("  ✓ New best model saved!\n");
            }
            else
            {
                steps_without_improvement += validation_interval;
            }

            record_training_step(history, step, train_loss, val_loss, perplexity);
        }

        // Generate training samples
        if (step % sample_interval == 0 && step > 0)
        {
            generate_training_sample(&model, &state, &dataset, &config, step);
        }

        // Save periodic checkpoints
        if (step % save_interval == 0 && step > 0)
        {
            char *filename = generate_model_filename("moe_model_checkpoint",
                                                     step, train_loss);
            save_model(&model, filename);
            printf("  ✓ Checkpoint saved: %s\n", filename);
            free(filename);
        }

        // Early stopping check
        if (steps_without_improvement >= patience)
        {
            printf("Early stopping: no improvement for %d steps\n", patience);
            break;
        }
    }

    time_t end_time = time(NULL);
    double training_duration = difftime(end_time, start_time);

    printf("\n=== Enhanced Training Complete ===\n");
    printf("Training duration: %.0f seconds (%.2f minutes)\n",
           training_duration, training_duration / 60.0);
    printf("Best validation loss: %.4f\n", best_val_loss);
    printf("Total steps completed: %d\n", optimizer.step);

    // Save final model
    char *final_filename = generate_model_filename("moe_model_final",
                                                   optimizer.step, best_val_loss);

    ModelMetadata final_metadata = {0};
    time_t now = time(NULL);
    strftime(final_metadata.timestamp, sizeof(final_metadata.timestamp),
             "%Y-%m-%d %H:%M:%S", localtime(&now));
    final_metadata.best_loss = best_val_loss;
    final_metadata.training_steps = optimizer.step;
    final_metadata.dataset_type = dataset.type;
    final_metadata.config = config;
    final_metadata.validation_loss = best_val_loss;

    save_model_with_metadata(&model, final_filename, &final_metadata);
    printf("✓ Final model saved: %s\n", final_filename);

    // Final analysis
    printf("\n=== Final Model Analysis ===\n");
    analyze_expert_usage(&state, &config, 1);

    // Generate final sample
    printf("\n=== Final Generation Test ===\n");
    GenerationConfig gen_config = create_generation_config();
    const char *test_prompt = (dataset.type == DATASET_TINYSHAKESPEARE) ? "To be or not to be" : "Once upon a time";
    generate_text_enhanced(&model, &state, &dataset, test_prompt, 100,
                           0.8f, 0.9f, &gen_config);

    // Cleanup
    free(final_filename);
    free(history->losses);
    free(history->val_losses);
    free(history->perplexities);
    free(history->steps);
    free(history);

    free_model(&model);
    free_state(&state, &config);
    free_dataset(&dataset);
    free_optimizer(&optimizer, &config);

    printf("\n=== Training Success! ===\n");
    printf("Next steps:\n");
    printf("  %s generate --dataset %s --prompt \"Your prompt here\"\n",
           argv[0], (dataset.type == DATASET_TINYSHAKESPEARE) ? "shakespeare" : "stories");
    printf("  %s analyze --dataset %s\n", argv[0],
           (dataset.type == DATASET_TINYSHAKESPEARE) ? "shakespeare" : "stories");

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

void save_model_with_metadata(GPT2_MoE_Model *model, const char *filename,
                              ModelMetadata *metadata)
{
    FILE *file = fopen(filename, "wb");
    if (!file)
    {
        printf("Error: Could not open file %s for writing\n", filename);
        return;
    }

    // Write metadata first
    fwrite(metadata, sizeof(ModelMetadata), 1, file);

    // Write config
    fwrite(&model->config, sizeof(Config), 1, file);

    Config *config = &model->config;

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
    printf("✓ Model with metadata saved to %s\n", filename);
}

int load_model_with_metadata(GPT2_MoE_Model *model, const char *filename,
                             ModelMetadata *metadata)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        printf("Error: Could not open file %s for reading\n", filename);
        return 0;
    }

    // Read metadata first
    if (fread(metadata, sizeof(ModelMetadata), 1, file) != 1)
    {
        printf("Error reading metadata from file\n");
        fclose(file);
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
    printf("✓ Model with metadata loaded from %s\n", filename);
    printf("  Trained for %d steps, best loss: %.4f, perplexity: %.2f\n",
           metadata->training_steps, metadata->best_loss, metadata->perplexity);
    return 1;
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

int check_repetition(int *tokens, int current_pos, int seq_len, int window_size)
{
    if (current_pos < window_size * 2)
        return 0;

    int matches = 0;
    for (int i = 1; i <= window_size && current_pos - i >= 0; i++)
    {
        if (tokens[current_pos - i] == tokens[current_pos - window_size - i])
        {
            matches++;
        }
    }

    return matches > window_size * 0.7; // 70% match threshold
}

void apply_repetition_penalty(float *logits, int *recent_tokens, int recent_count,
                              int vocab_size, float penalty)
{
    for (int i = 0; i < recent_count; i++)
    {
        int token = recent_tokens[i];
        if (token >= 0 && token < vocab_size)
        {
            if (logits[token] > 0)
            {
                logits[token] /= penalty;
            }
            else
            {
                logits[token] *= penalty;
            }
        }
    }
}

int sample_with_quality_control(float *logits, int vocab_size, float temperature,
                                float top_p, int *recent_tokens, int recent_count,
                                GenerationConfig *gen_config)
{
    // Apply repetition penalty
    apply_repetition_penalty(logits, recent_tokens, recent_count,
                             vocab_size, gen_config->repetition_penalty);

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

    // Normalize
    for (int i = 0; i < vocab_size; i++)
    {
        logits[i] /= sum;
    }

    // Top-p sampling
    if (top_p < 1.0f)
    {
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

        // Sort by probability (simple bubble sort for small vocab)
        qsort(prob_indices, vocab_size, sizeof(ProbIndex), cmp_prob_desc);

        float cumsum = 0.0f;
        int cutoff = vocab_size - 1;
        for (int i = 0; i < vocab_size; i++)
        {
            cumsum += prob_indices[i].prob;
            if (cumsum >= top_p)
            {
                cutoff = i;
                break;
            }
        }

        // Sample from top-p subset
        float r = (float)rand() / RAND_MAX;
        cumsum = 0.0f;
        for (int i = 0; i <= cutoff; i++)
        {
            cumsum += prob_indices[i].prob;
            if (r <= cumsum)
            {
                int result = prob_indices[i].index;
                free(prob_indices);
                return result;
            }
        }

        free(prob_indices);
        return prob_indices[cutoff].index;
    }
    else
    {
        // Standard sampling
        float r = (float)rand() / RAND_MAX;
        float cumsum = 0.0f;
        for (int i = 0; i < vocab_size; i++)
        {
            cumsum += logits[i];
            if (r <= cumsum)
                return i;
        }
        return vocab_size - 1;
    }
}

void generate_text_enhanced(GPT2_MoE_Model *model, RunState *state, Dataset *dataset,
                            const char *prompt, int max_tokens, float temperature,
                            float top_p, GenerationConfig *gen_config)
{
    Config *config = &model->config;
    int context_size = config->seq_len;

    printf("\n=== Enhanced Text Generation ===\n");
    printf("Dataset: %s\n", dataset->name);
    printf("Prompt: \"%s\"\n", prompt);
    printf("Parameters: temp=%.2f, top_p=%.2f, max_tokens=%d\n",
           temperature, top_p, max_tokens);
    printf("Repetition penalty: %.2f, window: %d\n",
           gen_config->repetition_penalty, gen_config->repetition_window);

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    // Tokenize prompt
    int *context = malloc(context_size * sizeof(int));
    int *recent_tokens = malloc(gen_config->repetition_window * sizeof(int));
    memset(context, 0, context_size * sizeof(int));
    memset(recent_tokens, -1, gen_config->repetition_window * sizeof(int));

    int prompt_len = strlen(prompt);
    int context_pos = 0;

    // Better prompt tokenization
    if (dataset->use_word_level)
    {
        // Word-level tokenization
        char *prompt_copy = strdup(prompt);
        char *token = strtok(prompt_copy, " ");

        while (token != NULL && context_pos < context_size - 1)
        {
            int found = 0; // UNK token
            for (int v = 0; v < dataset->vocab_size; v++)
            {
                if (strcmp(dataset->vocab[v], token) == 0)
                {
                    found = v;
                    break;
                }
            }
            context[context_pos++] = found;
            token = strtok(NULL, " ");
        }
        free(prompt_copy);
    }
    else
    {
        // Character-level tokenization
        for (int i = 0; i < prompt_len && context_pos < context_size - 1; i++)
        {
            char c = prompt[i];
            for (int v = 0; v < dataset->vocab_size; v++)
            {
                if (dataset->vocab[v][0] == c)
                {
                    context[context_pos++] = v;
                    break;
                }
            }
        }
    }

    printf("\nGenerated text:\n\"%s", prompt);
    fflush(stdout);

    int generated_count = 0;
    int consecutive_repeats = 0;
    int recent_pos = 0;

    // Generation loop
    for (int gen = 0; gen < max_tokens; gen++)
    {
        // Forward pass
        forward_pass(model, state, context);

        // Get logits for the last position
        int pred_pos = (context_pos - 1) % context_size;
        float *logits = &state->logits[pred_pos * config->vocab_size];

        // Copy logits for modification
        float *logits_copy = malloc(config->vocab_size * sizeof(float));
        memcpy(logits_copy, logits, config->vocab_size * sizeof(float));

        // Sample next token with quality control
        int next_token = sample_with_quality_control(logits_copy, config->vocab_size,
                                                     temperature, top_p, recent_tokens,
                                                     gen_config->repetition_window, gen_config);
        free(logits_copy);

        // Check for repetition
        if (check_repetition(context, context_pos, context_size,
                             gen_config->repetition_window))
        {
            consecutive_repeats++;
            if (consecutive_repeats >= gen_config->max_repetitions &&
                generated_count >= gen_config->min_tokens)
            {
                printf(" [stopped: repetition detected]");
                break;
            }
        }
        else
        {
            consecutive_repeats = 0;
        }

        // Update recent tokens for repetition penalty
        recent_tokens[recent_pos] = next_token;
        recent_pos = (recent_pos + 1) % gen_config->repetition_window;

        // Print generated token
        if (next_token < dataset->vocab_size)
        {
            if (dataset->use_word_level)
            {
                printf(" %s", dataset->vocab[next_token]);
            }
            else
            {
                printf("%c", dataset->vocab[next_token][0]);
            }
            fflush(stdout);
        }

        // Update context with sliding window
        if (context_pos >= context_size)
        {
            // Shift context left and add new token
            for (int i = 0; i < context_size - 1; i++)
            {
                context[i] = context[i + 1];
            }
            context[context_size - 1] = next_token;
        }
        else
        {
            context[context_pos++] = next_token;
        }

        generated_count++;
    }

    gettimeofday(&end_time, NULL);
    double elapsed = (end_time.tv_sec - start_time.tv_sec) +
                     (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

    printf("\"\n\n=== Generation Statistics ===\n");
    printf("Tokens generated: %d\n", generated_count);
    printf("Time taken: %.3f seconds\n", elapsed);
    printf("Speed: %.2f tokens/second\n", generated_count / elapsed);
    printf("Repetition penalty applied: %s\n",
           consecutive_repeats > 0 ? "Yes" : "No");

    free(context);
    free(recent_tokens);
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
