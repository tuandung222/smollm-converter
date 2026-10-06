/*
SmolLM2-135M Pure C Inference Engine implemented from scratch.
Zero external library dependencies: only standard C library (stdio, stdlib, math, string).
Direct, transparent port of torch_impl/modeling_llama.py to C.
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

typedef struct {
    int dim;           // 576
    int hidden_dim;    // 1536
    int n_layers;      // 30
    int n_heads;       // 9
    int n_kv_heads;    // 3
    int vocab_size;    // 49152
    int seq_len;       // 2048
} Config;

typedef struct {
    float* token_embedding_table; // [vocab_size, dim]
    float* rms_att_weight;        // [n_layers, dim]
    float* wq;                    // [n_layers, dim, dim]
    float* wk;                    // [n_layers, kv_dim, dim]
    float* wv;                    // [n_layers, kv_dim, dim]
    float* wo;                    // [n_layers, dim, dim]
    float* rms_ffn_weight;        // [n_layers, dim]
    float* w1;                    // [n_layers, hidden_dim, dim] (gate)
    float* w2;                    // [n_layers, dim, hidden_dim] (down)
    float* w3;                    // [n_layers, hidden_dim, dim] (up)
    float* rms_final_weight;      // [dim]
    float* wcls;                  // [vocab_size, dim] (lm_head)
} TransformerWeights;

typedef struct {
    float *x;           // [dim]
    float *xb;          // [dim]
    float *hb;          // [hidden_dim]
    float *hb2;         // [hidden_dim]
    float *q;           // [dim]
    float *k;           // [kv_dim]
    float *v;           // [kv_dim]
    float *att;         // [dim]
    float *logits;      // [vocab_size]
    float *key_cache;   // [n_layers, seq_len, kv_dim]
    float *value_cache; // [n_layers, seq_len, kv_dim]
} RunState;

typedef struct {
    char** vocab;
    float* vocab_scores;
    int vocab_size;
} Tokenizer;

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = (float*)calloc(p->dim, sizeof(float));
    s->xb = (float*)calloc(p->dim, sizeof(float));
    s->hb = (float*)calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float*)calloc(p->hidden_dim, sizeof(float));
    s->q = (float*)calloc(p->dim, sizeof(float));
    s->k = (float*)calloc(kv_dim, sizeof(float));
    s->v = (float*)calloc(kv_dim, sizeof(float));
    s->att = (float*)calloc(p->dim, sizeof(float));
    s->logits = (float*)calloc(p->vocab_size, sizeof(float));
    s->key_cache = (float*)calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float*)calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
}

// 1. RMSNorm: Direct C translation of LlamaRMSNorm
void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    float inv = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = x[j] * inv * weight[j];
    }
}

// 2. Matrix Multiplication: y = W * x
void matmul(float* xout, float* x, float* w, int n, int d) {
    // W is [d, n], x is [n], xout is [d]
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        float* w_row = w + i * n;
        for (int j = 0; j < n; j++) {
            val += w_row[j] * x[j];
        }
        xout[i] = val;
    }
}

// 3. Softmax
void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

// 4. RoPE: Direct C translation of apply_rotary_pos_emb with rotate_half
void apply_rope(float* vec, int n_heads, int head_dim, int pos) {
    int half_dim = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float* head_vec = vec + h * head_dim;
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / powf(100000.0f, (float)(2 * i) / (float)head_dim);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);

            float v0 = head_vec[i];
            float v1 = head_vec[i + half_dim];
            // rotate_half: [-v1, v0] -> v * cos + rotate_half * sin
            head_vec[i]            = v0 * fcr - v1 * fci;
            head_vec[i + half_dim] = v1 * fcr + v0 * fci;
        }
    }
}

// 5. Transformer Forward Pass (1 Token at pos)
void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w) {
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // 9 / 3 = 3
    int hidden_dim = p->hidden_dim;
    int head_dim = dim / p->n_heads; // 64

    // Token embedding lookup
    memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));

    // Iterate through 30 layers
    for (int l = 0; l < p->n_layers; l++) {
        // --- 1. Attention pre-norm ---
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // Linear projections
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * kv_dim * dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * kv_dim * dim, dim, kv_dim);

        // Apply RoPE
        apply_rope(s->q, p->n_heads, head_dim, pos);
        apply_rope(s->k, p->n_kv_heads, head_dim, pos);

        // Store into KV-Cache at position pos
        size_t cache_offset = ((size_t)l * p->seq_len + pos) * kv_dim;
        memcpy(s->key_cache + cache_offset, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + cache_offset, s->v, kv_dim * sizeof(float));

        // Grouped-Query Multi-Head Attention
        float scale = 1.0f / sqrtf((float)head_dim);
        for (int h = 0; h < p->n_heads; h++) {
            float* q_head = s->q + h * head_dim;
            int kv_h = h / kv_mul;
            float* att_scores = (float*)malloc((pos + 1) * sizeof(float));

            // Dot product Q * K for all tokens 0..pos
            for (int t = 0; t <= pos; t++) {
                size_t t_offset = ((size_t)l * p->seq_len + t) * kv_dim + kv_h * head_dim;
                float* k_head = s->key_cache + t_offset;
                float score = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    score += q_head[i] * k_head[i];
                }
                att_scores[t] = score * scale;
            }

            softmax(att_scores, pos + 1);

            // Weighted sum over V
            float* out_head = s->att + h * head_dim;
            memset(out_head, 0, head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                size_t t_offset = ((size_t)l * p->seq_len + t) * kv_dim + kv_h * head_dim;
                float* v_head = s->value_cache + t_offset;
                float a = att_scores[t];
                for (int i = 0; i < head_dim; i++) {
                    out_head[i] += a * v_head[i];
                }
            }
            free(att_scores);
        }

        // Attention output projection
        matmul(s->xb, s->att, w->wo + l * dim * dim, dim, dim);

        // Residual connection
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];

        // --- 2. Feed-Forward SwiGLU pre-norm ---
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        matmul(s->hb, s->xb, w->w1 + l * hidden_dim * dim, dim, hidden_dim);  // gate
        matmul(s->hb2, s->xb, w->w3 + l * hidden_dim * dim, dim, hidden_dim); // up

        // SwiGLU activation: silu(hb) * hb2
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            float silu = val / (1.0f + expf(-val));
            s->hb[i] = silu * s->hb2[i];
        }

        // Down projection
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

        // Residual connection
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];
    }

    // Final RMSNorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // LM Head
    matmul(s->logits, x, w->wcls, dim, p->vocab_size);
}

// 6. Tokenizer loader
void load_tokenizer(Tokenizer* t, const char* path, int vocab_size) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("[-] Failed to open tokenizer: %s\n", path);
        exit(1);
    }
    int file_vocab_size = 0;
    fread(&file_vocab_size, sizeof(int), 1, f);
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));

    for (int i = 0; i < vocab_size; i++) {
        float score;
        int len;
        fread(&score, sizeof(float), 1, f);
        fread(&len, sizeof(int), 1, f);
        t->vocab_scores[i] = score;
        t->vocab[i] = (char*)malloc(len + 1);
        fread(t->vocab[i], 1, len, f);
        t->vocab[i][len] = '\0';
    }
    fclose(f);
}

int main(int argc, char* argv[]) {
    const char* model_path = (argc > 1) ? argv[1] : "c_scratch/smollm2_135m_raw.bin";
    const char* tok_path = (argc > 2) ? argv[2] : "c_scratch/tokenizer.bin";
    int n_predict = (argc > 3) ? atoi(argv[3]) : 20;

    printf("=================================================================\n");
    printf("         SmolLM2-135M PURE C INFERENCE (FROM SCRATCH)            \n");
    printf("=================================================================\n");
    printf("[*] Loading model from: %s\n", model_path);

    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        printf("[-] Error opening %s\n", model_path);
        return 1;
    }
    off_t file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);

    void* data = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
        printf("[-] mmap failed\n");
        return 1;
    }

    Config config;
    int* ptr_i = (int*)data;
    config.dim = ptr_i[0];
    config.hidden_dim = ptr_i[1];
    config.n_layers = ptr_i[2];
    config.n_heads = ptr_i[3];
    config.n_kv_heads = ptr_i[4];
    config.vocab_size = ptr_i[5];
    config.seq_len = ptr_i[6];

    printf("[+] Model Config: dim=%d, layers=%d, heads=%d, kv_heads=%d, vocab=%d\n",
           config.dim, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size);

    float* weights_ptr = (float*)(ptr_i + 7);
    TransformerWeights w;
    int dim = config.dim;
    int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    int hidden_dim = config.hidden_dim;
    int n_layers = config.n_layers;
    int vocab_size = config.vocab_size;

    w.token_embedding_table = weights_ptr; weights_ptr += (size_t)vocab_size * dim;
    w.rms_att_weight = weights_ptr; weights_ptr += (size_t)n_layers * dim;
    w.wq = weights_ptr; weights_ptr += (size_t)n_layers * dim * dim;
    w.wk = weights_ptr; weights_ptr += (size_t)n_layers * kv_dim * dim;
    w.wv = weights_ptr; weights_ptr += (size_t)n_layers * kv_dim * dim;
    w.wo = weights_ptr; weights_ptr += (size_t)n_layers * dim * dim;
    w.rms_ffn_weight = weights_ptr; weights_ptr += (size_t)n_layers * dim;
    w.w1 = weights_ptr; weights_ptr += (size_t)n_layers * hidden_dim * dim;
    w.w2 = weights_ptr; weights_ptr += (size_t)n_layers * dim * hidden_dim;
    w.w3 = weights_ptr; weights_ptr += (size_t)n_layers * hidden_dim * dim;
    w.rms_final_weight = weights_ptr; weights_ptr += dim;
    w.wcls = weights_ptr;

    RunState state;
    malloc_run_state(&state, &config);

    Tokenizer tokenizer;
    load_tokenizer(&tokenizer, tok_path, config.vocab_size);

    // Read prompt tokens from command-line arguments if provided
    int prompt_tokens[1024];
    int n_prompt = 0;

    if (argc > 4) {
        n_prompt = argc - 4;
        for (int i = 0; i < n_prompt; i++) {
            prompt_tokens[i] = atoi(argv[4 + i]);
        }
    } else {
        // Default prompt tokens: "The theory of relativity explains that"
        int default_tokens[] = {504, 3108, 282, 24581, 6285, 338};
        n_prompt = sizeof(default_tokens) / sizeof(default_tokens[0]);
        for (int i = 0; i < n_prompt; i++) prompt_tokens[i] = default_tokens[i];
    }

    printf("\n[*] Starting Autoregressive Token Generation in Pure C:\n--> ");
    for (int i = 0; i < n_prompt; i++) {
        printf("%s", tokenizer.vocab[prompt_tokens[i]]);
    }
    fflush(stdout);

    clock_t t0 = clock();
    int token = prompt_tokens[0];
    int pos = 0;

    // Prefill prompt tokens
    while (pos < n_prompt - 1) {
        transformer(token, pos, &config, &state, &w);
        pos++;
        token = prompt_tokens[pos];
    }

    // Autoregressive generation
    for (int step = 0; step < n_predict; step++) {
        transformer(token, pos, &config, &state, &w);

        // Greedy argmax
        int next_token = 0;
        float max_logit = state.logits[0];
        for (int v = 1; v < config.vocab_size; v++) {
            if (state.logits[v] > max_logit) {
                max_logit = state.logits[v];
                next_token = v;
            }
        }

        printf("%s", tokenizer.vocab[next_token]);
        fflush(stdout);

        token = next_token;
        pos++;
    }
    clock_t t1 = clock();

    double sec = (double)(t1 - t0) / CLOCKS_PER_SEC;
    double tps = n_predict / sec;

    printf("\n\n=================================================================\n");
    printf("Generated %d tokens in %.3f s (%.2f tokens/second in pure C)\n", n_predict, sec, tps);
    printf("=================================================================\n");

    return 0;
}
