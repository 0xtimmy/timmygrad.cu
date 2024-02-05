#include<string>
#include<list>
#include"./tokenizer.cuh"
#include"./model.cuh"
#include"./helpers.cuh"

int EPOCHS = 10;
int BLOCK_SIZE = 512;
int N_EMBED = 768;
int N_HEAD = 8;
int N_LAYERS = 12;
int BATCH_SIZE = 1;

bool generate = true;

const std::string FILENAME = "./tiny_shakespeare.txt";  

int main() {
    
    // read dataset
    std::string dataset = readFile("tiny_shakespeare.txt");

    std::set<char> chars(dataset.begin(), dataset.end());
    Tokenizer* tokenizer = new Tokenizer(chars);
    std::cerr << "initialized tokenizer\n";

    // Create training and validation set
    int slice_pos = dataset.size() / 10;
    Tokens* val_data = new Tokens(tokenizer, dataset.substr(0, slice_pos));
    Tokens* train_data = new Tokens(tokenizer, dataset.substr(slice_pos+1));

    std::cerr << train_data->size() << " tokens in training set\n";

    //Bigram model = Bigram("model", BATCH_SIZE, tokenizer->vocab_size(), BLOCK_SIZE, N_EMBED);

    Transformer model = Transformer("model", BATCH_SIZE, tokenizer->vocab_size(), BLOCK_SIZE, N_LAYERS, N_EMBED, N_HEAD);

    std::cerr << "initialized model\n";
    
    int iteration = 0;
    int total_iterations = EPOCHS * train_data->size() / BLOCK_SIZE / BATCH_SIZE;

    Tensor* logits = new Tensor("logits", new Shape(3, BATCH_SIZE, BLOCK_SIZE, tokenizer->vocab_size()));
    Tensor* L = new Tensor("loss", new Shape(1, BLOCK_SIZE*BATCH_SIZE));

    logits->toDevice();
    L->toDevice();

    std::list<float> losses = {};

    for(int i = 0; i < EPOCHS; i++) {
        for(int j = 0; j < train_data->size()-BLOCK_SIZE*BATCH_SIZE; j += BLOCK_SIZE) {
            int* x = train_data->sample(BATCH_SIZE, j, BLOCK_SIZE);
            std::cerr << "\n\n----------------------------------------\n\n";
            std::cerr << "\n\nTraining on sample: " << tokenizer->decode(x, BLOCK_SIZE) << "\n";
            int* devx;
            cudaMalloc(&devx, sizeof(int)*BLOCK_SIZE*BATCH_SIZE);
            cce(cudaMemcpy(devx, x, sizeof(int)*BLOCK_SIZE*BATCH_SIZE, cudaMemcpyHostToDevice));

            model(devx, logits);

            Tensor::cross_entropy(logits, L, devx, BLOCK_SIZE*BATCH_SIZE);

            L->backward();

            L->toHost();

            losses.push_front(L->host_ptr[0]);

            float avg_loss = 0;
            for (float l : losses) { avg_loss += l; }
            avg_loss /= losses.size();
            if(losses.size() >= 100) losses = {};

            iteration++;
            std::cerr << "[" << (100*iteration/total_iterations) << "%] ";
            std::cerr << "Loss: " << L->host_ptr[0] << " [" << avg_loss << "]" << "\n";
      

            std::cerr << "Sample: ";
            std::cerr << tokenizer->decode_logits(logits, BLOCK_SIZE) << "\n";  
            L->toDevice();
            L->optimize(0.000001f);

            free(x);
            cce(cudaFree(devx));
        }
        std::cerr << "Generating: " << model.generate(dataset.substr(0, BLOCK_SIZE), tokenizer, 16) << "\n";
    }
}