#ifndef MODEL_H
#define MODEL_H

#include"./tensor.cuh"

class Embedding : public Module {

    public:
        std::string name;
        int block_size;
        int num_embeddings;
        int embedding_dim;
        Tensor* enc;
        Tensor* weights;

        Embedding() { }

        Embedding(std::string _name, int _block_size, int _num_embeddings, int _embedding_dim) {
            name = _name;
            block_size = _block_size;
            num_embeddings = _num_embeddings;
            embedding_dim = _embedding_dim;

            enc = new Tensor(name + "/one-hot", new Shape(2, _block_size, _num_embeddings));
            weights = new Tensor(name + "/weights", new Shape(2, _num_embeddings, _embedding_dim));
            weights->normal();
            weights->learnable = true;

            enc->toDevice();
            weights->toDevice();
        }

        ~Embedding() {
            delete weights;
            delete enc;
        }

        void operator() (int* dev_x, Tensor* y) { 
            algebra::onehot<<<enc->size()/128+1, 128>>>(dev_x, enc->dev_ptr, num_embeddings, block_size);
            cce(enc->name);
            Tensor::mm(enc, weights, y);
        }
};

class Linear : public Module {
    public:
        std::string name;
        Tensor* weights;
        Tensor* bias;

        Tensor* y0;

        Linear() { }
        
        Linear(std::string _name, int batches, int block_size, int in_features, int out_features) {
            name = _name;
            weights = new Tensor(name + "/weights", new Shape(2, in_features, out_features));
            bias = new Tensor(name + "/bias", new Shape(1, out_features));
            y0 = new Tensor(name + "/y0", new Shape(3, batches, block_size, out_features));

            float k = sqrtf(1.0f/in_features);
            weights->uniform(-k, k);
            bias->uniform(-k, k);
            weights->learnable = true;
            bias->learnable = true;

            weights->toDevice();
            bias->toDevice();
            y0->toDevice();
        }

        ~Linear() {
            delete weights;
            delete bias;
            delete y0;
        }

        void operator() (Tensor* x, Tensor* y) {
            Tensor::mm(x, weights, y0);
            Tensor::add(y0, bias, y);
        }
};

class RELU : public Module {
    public:
        std::string name; 

        RELU() { }
        RELU(std::string _name) {
            name = _name;
        }

        void operator() (Tensor* x, Tensor* y) {
            Tensor::relu(x, y);
        }
};

class Layernorm : public Module {
    public:
        std::string name;
        Shape* normshape;
        int layers;

        Tensor* weights;
        Tensor* bias;

        Tensor* y0;
        Tensor* y1;

        Layernorm() { }
        Layernorm(std::string _name, Shape* _normshape, int _layers) {
            name = _name;
            normshape = _normshape;
            layers = _layers;
            weights = new Tensor(name + "/weights", normshape);
            bias = new Tensor(name + "/bias", normshape);
            weights->learnable = true;
            bias->learnable = true;

            y0 = new Tensor(name + "/normalized", new Shape(2, layers, normshape->size()));
            y1 = new Tensor(name + "/scaled", new Shape(2, layers, normshape->size()));

            weights->toDevice();
            bias->toDevice();
            weights->set(1.0);
            bias->set(0.0);
            y0->toDevice();
            y1->toDevice();
        }

        void operator() (Tensor* x, Tensor* y) {
            Tensor::layernorm(x, y0, layers, normshape->size());
            Tensor::mul(y0, weights, y1);
            Tensor::add(y1, bias, y);
        }

};

class SelfAttention : public Module {

    public:
        int n_embed;
        int n_head;
        int block_size;
        int batches;

        Linear* proj_q;
        Linear* proj_k;
        Linear* proj_v;
        Linear* proj_out;

        Tensor* mask;

        Tensor* query;
        Tensor* key;
        Tensor* value;
        Tensor* q;
        Tensor* k;
        Tensor* v;

        Tensor* kt;

        Tensor* kq;
        Tensor* scaled_kq;
        Tensor* masked;
        Tensor* attn;

        Tensor* attention;
        Tensor* attentiont;
    
        SelfAttention(std::string name, int _n_embed, int _n_head, int _block_size, int _batches) {
            batches = _batches;
            block_size = _block_size;
            n_embed = _n_embed;
            n_head = _n_head;

            
            proj_q = new Linear(name + "/proj_q", batches, block_size, n_embed, n_embed);
            proj_k = new Linear(name + "/proj_k", batches, block_size, n_embed, n_embed);
            proj_v = new Linear(name + "/proj_v", batches, block_size, n_embed, n_embed);
            proj_out = new Linear(name + "/proj_out", batches, block_size, n_embed, n_embed);

            mask = Tensor::tri(block_size);
            mask->name = name + "/attention mask";

            query = new Tensor(name + "/query", new Shape(4, batches, block_size, n_head, n_embed / n_head));
            key = new Tensor(name + "/key", new Shape(4, batches, block_size, n_head, n_embed / n_head));
            value = new Tensor(name + "/value", new Shape(4, batches, block_size, n_head, n_embed / n_head));
            query->toDevice();
            key->toDevice();
            value->toDevice();

            q = new Tensor(name + "/q", new Shape(4, batches, n_head, block_size, n_embed / n_head));
            k = new Tensor(name + "/k", new Shape(4, batches, n_head, block_size, n_embed / n_head));
            v = new Tensor(name + "/v", new Shape(4, batches, n_head, block_size, n_embed / n_head));
            q->toDevice();
            k->toDevice();
            v->toDevice();

            kt = new Tensor(name + "/kt", new Shape(4, batches, n_head, n_embed / n_head, block_size));
            kt->toDevice();

            kq = new Tensor(name + "/kq", new Shape(4, batches, n_head, block_size, block_size));
            scaled_kq = new Tensor(name + "/scaled_kq", new Shape(4, batches, n_head, block_size, block_size));
            masked = new Tensor(name + "/masked", new Shape(4, batches, n_head, block_size, block_size));
            attn = new Tensor(name + "/attn", new Shape(4, batches, n_head, block_size, block_size));
            kq->toDevice();
            scaled_kq->toDevice();
            masked->toDevice();
            attn->toDevice();

            attention = new Tensor(name + "/attention", new Shape(4, batches, n_head, block_size, n_embed / n_head));
            attentiont = new Tensor(name + "/attentiont", new Shape(3, batches, block_size, n_embed));
            attention->toDevice();
            attentiont->toDevice();
        }

        void operator() (Tensor* x, Tensor* y) {

            // calculate q k v
            (*proj_q)(x, query);
            (*proj_k)(x, key);
            (*proj_v)(x, value);

            Tensor::transpose(query, q, 1, 2);
            Tensor::transpose(key, k, 1, 2);
            Tensor::transpose(value, v, 1, 2);
            // attn = q mm k / sqrt(k.size)
            Tensor::transpose(k, kt, 2, 3);
            
            Tensor::mm(q, kt, kq);
            Tensor::div(kq, sqrtf((float)((*(k->shape))[3])), scaled_kq);
            Tensor::mask(scaled_kq, mask, masked);
            Tensor::softmax(masked, attn, batches*block_size*n_head, block_size);

            // y = attn mm v
            Tensor::mm(attn, v, attention);

            Tensor::transpose(attention, attentiont, 1, 2);

            (*proj_out)(attentiont, y);
        }
};

class MLP : public Module {

    public:
        std::string name;
        int batches;
        int block_size;
        int n_embed;

        Linear* ln_in;
        RELU* re;
        Linear* ln_out;

        Tensor* lt_in;
        Tensor* activations;

        MLP(std::string _name, int _batches, int _block_size, int _n_embed) {
            name = _name;
            batches = _batches;
            block_size = _block_size;
            n_embed = _n_embed;

            ln_in = new Linear(name + "/ln_in", batches, block_size, n_embed, 4*n_embed);
            re = new RELU(name + "/re");
            ln_out = new Linear(name + "/ln_out", batches, block_size, 4*n_embed, n_embed);

            lt_in = new Tensor(name + "/lt_in", new Shape(3, batches, block_size, 4*n_embed));
            activations = new Tensor(name + "/activations", new Shape(3, batches, block_size, 4*n_embed));
        }

        void operator() (Tensor* x, Tensor* y) {
            (*ln_in)(x, lt_in);
            (*re)(lt_in, activations);
            (*ln_out)(activations, y);
        }  

};

class Block : public Module {

    public:
        std::string name;
        int batches;
        int block_size;
        int n_embed;
        int n_head;

        Layernorm* ln_in;
        SelfAttention* sa;

        Layernorm* ln_out;
        MLP* mlp;

        Tensor* x0;
        Tensor* x1;
        Tensor* x2;
        Tensor* x3;
        Tensor* x4;

        Block(std::string _name, int _batches, int _block_size, int _n_embed, int _n_head) {
            name = _name;
            batches = _batches;
            block_size = _block_size;
            n_embed = _n_embed;
            n_head = _n_head;

            ln_in = new Layernorm(name + "/ln_in", new Shape(1, n_embed), batches*block_size);
            sa = new SelfAttention(name + "/sa", n_embed, n_head, block_size, batches);
            ln_out = new Layernorm(name + "/ln_out", new Shape(1, n_embed), batches*block_size);
            mlp = new MLP(name + "/mlp", batches, block_size, n_embed);

            x0 = new Tensor(name + "/x0", new Shape(3, batches, block_size, n_embed));
            x1 = new Tensor(name + "/x1", new Shape(3, batches, block_size, n_embed));
            x2 = new Tensor(name + "/x2", new Shape(3, batches, block_size, n_embed));
            x3 = new Tensor(name + "/x3", new Shape(3, batches, block_size, n_embed));
            x4 = new Tensor(name + "/x4", new Shape(3, batches, block_size, n_embed));
            x0->toDevice();
            x1->toDevice();
            x2->toDevice();
            x3->toDevice();
            x4->toDevice();
        }

        void operator() (Tensor* x, Tensor* y) {
            // self attention
            (*ln_in)(x, x0);
            (*sa)(x0, x1);
            Tensor::add(x, x1, x2);

            (*ln_out)(x2, x3);
            (*mlp)(x3, x4);
            Tensor::add(x2, x4, y);
        }
};

class Bigram : public Module {

    public:
        int batches;
        int block_size;
        int vocab_size;
        int emb_dim;

        int* positions;

        Embedding* tok_emb;
        Embedding* pos_emb;

        Tensor* tok_enc;
        Tensor* pos_enc;
        Tensor* x_enc;

        Layernorm* ln0;
        Tensor* x0;
        SelfAttention* sa0;
        Tensor* x1;
        MLP* mlp0;
        Tensor* x2;
        Layernorm* ln1;
        Tensor* x3;
        Layernorm* ln_out;
        Tensor* x4;

        Linear* proj_out;

        Bigram() { }

        Bigram(std::string name, int _batches, int _vocab_size, int _block_size, int _emb_dim) {
            batches = _batches;
            vocab_size = _vocab_size;
            block_size = _block_size;
            emb_dim = _emb_dim;

            int *host_positions = (int*)malloc(sizeof(int)*block_size);
            for(int i = 0; i < block_size; i++) { host_positions[i] = i; }
            cce(cudaMalloc(&positions, sizeof(int)*_block_size));
            cce(cudaMemcpy(positions, host_positions, sizeof(int)*_block_size, cudaMemcpyHostToDevice));

            tok_enc = new Tensor(name + "/tok_enc", new Shape(3, batches, block_size, emb_dim));
            pos_enc = new Tensor(name + "/pos_enc", new Shape(3, batches, block_size, emb_dim));
            tok_emb = new Embedding(name + "/tok_emb", block_size, vocab_size, emb_dim);
            pos_emb = new Embedding(name + "/pos_emb", block_size, block_size, emb_dim);

            x_enc = new Tensor("bigram enc", new Shape(3, batches, block_size, emb_dim));

            ln0 = new Layernorm("ln_in", new Shape(1, emb_dim), batches*block_size);
            x0 = new Tensor(name + "/pos_enc", new Shape(3, batches, block_size, emb_dim));
            sa0 = new SelfAttention("sa0", emb_dim, 4, block_size, batches);
            x1 = new Tensor(name + "/pos_enc", new Shape(3, batches, block_size, emb_dim));
            ln1 = new Layernorm("ln_in", new Shape(1, emb_dim), batches*block_size);
            x2 = new Tensor(name + "/pos_enc", new Shape(3, batches, block_size, emb_dim));
            mlp0 = new MLP("mlp", batches, block_size, emb_dim);
            x3 = new Tensor(name + "/pos_enc", new Shape(3, batches, block_size, emb_dim));
            ln_out = new Layernorm("ln_in", new Shape(1, emb_dim), batches*block_size);
            x4 = new Tensor(name + "/pos_enc", new Shape(3, batches, block_size, emb_dim));
            proj_out = new Linear(name + "/proj_out", batches, block_size, emb_dim, vocab_size);

            tok_enc->toDevice();
            pos_enc->toDevice();
            x_enc->toDevice();
        }

        ~Bigram() {
            delete tok_emb;
            delete pos_emb;
            delete proj_out;

            delete tok_enc;
            delete pos_enc;
            delete x_enc;

            free(positions);
        }

        void operator() (int* dev_x, Tensor* y) { 
            (*tok_emb)(dev_x, tok_enc);
            (*pos_emb)(positions, pos_enc);
            
            Tensor::add(tok_enc, pos_enc, x_enc);

            (*ln0)(x_enc, x0);
            (*sa0)(x0, x1);
            (*ln1)(x1, x2);
            (*mlp0)(x2, x3);
            (*ln_out)(x3, x4);

            (*proj_out)(x4, y);
        }

        std::string generate(std::string starter, Tokenizer* tokenizer, int num) {
            std::string out = starter;

            std::cerr << "generating\n";
            int* tokens = tokenizer->encode(starter, block_size);
            int* dev_tokens;
            cce(cudaMalloc(&dev_tokens, sizeof(int)*block_size*batches));
            int next_token;

            Tensor* y = new Tensor("generator y", new Shape(3, batches, block_size, vocab_size));
            Tensor* logits = new Tensor("generator logits", new Shape(3, batches, block_size, vocab_size));
            y->toDevice();
            logits->toDevice();
            for(int t = 0; t < num; t++) {
                cce(cudaMemcpy(dev_tokens, tokens, sizeof(int)*block_size*batches, cudaMemcpyHostToDevice));
                (*this)(dev_tokens, y);
                Tensor::softmax(y, logits, batches*block_size, vocab_size);
                logits->toHost();
                float max = -1;
                next_token = 0;
                logits->toHost();
                //std::cerr << "logits: " << logits->pretty() << "\n";
                for(int i = 0; i < vocab_size; i++) { 
                    if(logits->host_ptr[(block_size-1)*vocab_size + i] > max) {
                        max = logits->host_ptr[(block_size-1)*vocab_size + i];
                        next_token = i;
                    }
                }
                logits->toDevice();

                for(int i = 0; i < block_size-1; i++) { tokens[i] = tokens[i+1]; }
                tokens[block_size-1] = next_token;
                out += tokenizer->decode(next_token);
            }
            std::cerr << "\n\ndone generating.\n";
            return out;
        }
};

class Transformer : public Module {

    public:
        int batches;
        int vocab_size;
        int block_size;
        int num_layers;
        int emb_dim;
        int num_heads;

        int* positions;

        Embedding* tok_emb;
        Embedding* pos_emb;

        Tensor* tok_enc;
        Tensor* pos_enc;
        Tensor* x_enc;

        Block** blocks; 
        Tensor** steps;

        Linear* proj_out;

        Transformer() { }

        Transformer(std::string name, int _batches, int _vocab_size,  int _block_size, int _num_layers, int _emb_dim, int _num_heads) {
            batches = _batches;
            vocab_size = _vocab_size;
            block_size = _block_size;
            num_layers = _num_layers;
            emb_dim = _emb_dim;
            num_heads = _num_heads;

            int *host_positions = (int*)malloc(sizeof(int)*block_size);
            for(int i = 0; i < block_size; i++) { host_positions[i] = i; }
            cce(cudaMalloc(&positions, sizeof(int)*_block_size));
            cce(cudaMemcpy(positions, host_positions, sizeof(int)*_block_size, cudaMemcpyHostToDevice));

            tok_enc = new Tensor(name + "/tok_enc", new Shape(3, batches, block_size, emb_dim));
            pos_enc = new Tensor(name + "/pos_enc", new Shape(3, batches, block_size, emb_dim));
            tok_emb = new Embedding(name + "/tok_emb", block_size, vocab_size, emb_dim);
            pos_emb = new Embedding(name + "/pos_emb", block_size, block_size, emb_dim);

            x_enc = new Tensor("token enc", new Shape(3, batches, block_size, emb_dim));

            tok_enc->toDevice();
            pos_enc->toDevice();
            x_enc->toDevice();

            blocks = (Block**)malloc(sizeof(Block*)*num_layers);
            steps = (Tensor**)malloc(sizeof(Tensor*)*(num_layers+1));
            steps[0] = x_enc;
            for(int i = 0; i < num_layers; i++) {
                blocks[i] = new Block(name + "/block" + std::to_string(i) ,batches, block_size, emb_dim, num_heads);
                steps[i+1] = new Tensor(name + "/step" + std::to_string(i), new Shape(3, batches, block_size, emb_dim));
                steps[i+1]->toDevice();
            }

            proj_out = new Linear(name + "/proj_out", batches, block_size, emb_dim, vocab_size);
        }

        ~Transformer() {
            delete tok_emb;
            delete pos_emb;
            delete proj_out;

            delete tok_enc;
            delete pos_enc;
            delete x_enc;

            free(positions);

            for(int i = 0; i < num_layers; i++) {
                delete blocks[i];
                delete steps[i+1];
            }
        }

        void operator() (int* dev_x, Tensor* y) { 
            (*tok_emb)(dev_x, tok_enc);
            (*pos_emb)(positions, pos_enc);
            
            Tensor::add(tok_enc, pos_enc, x_enc);

            for(int i = 0; i < num_layers; i++) {
                (*(blocks[i]))(steps[i], steps[i+1]);
            }

            (*proj_out)(steps[num_layers], y);
        }

        std::string generate(std::string starter, Tokenizer* tokenizer, int num) {
            std::string out = starter;

            std::cerr << "generating\n";
            int* tokens = tokenizer->encode(starter, block_size);
            int* dev_tokens;
            cce(cudaMalloc(&dev_tokens, sizeof(int)*block_size*batches));
            int next_token;

            Tensor* y = new Tensor("generator y", new Shape(3, batches, block_size, vocab_size));
            Tensor* logits = new Tensor("generator logits", new Shape(3, batches, block_size, vocab_size));
            y->toDevice();
            logits->toDevice();
            for(int t = 0; t < num; t++) {
                cce(cudaMemcpy(dev_tokens, tokens, sizeof(int)*block_size*batches, cudaMemcpyHostToDevice));
                (*this)(dev_tokens, y);
                Tensor::softmax(y, logits, batches*block_size, vocab_size);
                logits->toHost();
                float max = -1;
                next_token = 0;
                logits->toHost();
                //std::cerr << "logits: " << logits->pretty() << "\n";
                for(int i = 0; i < vocab_size; i++) { 
                    if(logits->host_ptr[(block_size-1)*vocab_size + i] > max) {
                        max = logits->host_ptr[(block_size-1)*vocab_size + i];
                        next_token = i;
                    }
                }
                logits->toDevice();

                for(int i = 0; i < block_size-1; i++) { tokens[i] = tokens[i+1]; }
                tokens[block_size-1] = next_token;
                out += tokenizer->decode(next_token);
            }
            std::cerr << "\n\ndone generating.\n";
            return out;
        }
};

#endif