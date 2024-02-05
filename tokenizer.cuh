#ifndef TOKENIZER_H
#define TOKENIZER_H

#include<string>
#include<iostream>
#include<map>
#include<set>
#include"./tensor.cuh"

// Declarations
class Tokens;
class Tokenizer;

// Definitions

class Tokenizer {

    std::set<char> vocab; 
    std::map<char, int> stoi;
    std::map<int, char> itos;

    public:
        Tokenizer(std::set<char> _vocab) {
            vocab = _vocab;
            stoi = {};
            itos = {};
            int i = 0;
            for (const char& c : vocab) {
                stoi[c] = i;
                itos[i] = c;
                i++;
            }

        }

        ~Tokenizer() { }

        int vocab_size() {
            return vocab.size();
        }

        int* encode(std::string x) {
            int* y = (int*)malloc(sizeof(int) * x.size());
            for (int i = 0; i < x.size(); i++) {
                y[i] = stoi[x[i]];
            }
            return y;
        }

        int* encode(std::string x, int block_size) {
            int* y = (int*)malloc(sizeof(int) * block_size);
            for (int i = 1; i <= block_size; i++) {
                if(i > x.size()) { y[block_size-i] = stoi['\n']; }
                else y[block_size-i] = stoi[x[block_size-i]];
            }
            return y;
        }

        std::string decode(int* x, int size) {
            std::string y(size, ' ');
            for (int i = 0; i < size; i++) {
                y[i] = itos[x[i]];
            }
            return y;
        }

        char decode(int x) {
            return itos[x];
        }

        std::string decode_logits(Tensor* logits, int block_size) {
            int* y = (int*)malloc(sizeof(int)*block_size);
            int* devy;
            cudaMalloc(&devy, sizeof(int)*block_size);
            algebra::max_idx<<<(block_size/32+1)*32, 32>>>(logits->dev_ptr, devy, this->vocab_size(), block_size);
            cudaMemcpy(y, devy, sizeof(int)*block_size, cudaMemcpyDeviceToHost);

            return decode(y, block_size);
        }

        void print_vocab() {
            std::cout << "There are " << vocab.size() << " unique characters: \n";

            for (const char& c : vocab) {
                std::cout << c;
            }
            std::cout << "\n";
        }
};

class Tokens {

    int _size;
    int* _tokens;
    std::string _chars;

    public:
        Tokens(Tokenizer* tokenizer, std::string str) {
            _size = str.size();
            _tokens = tokenizer->encode(str);
            _chars = str;
        }

        ~Tokens() {
            free(_tokens);
        }

        int size() {
            return _size;
        }
        int* tokens() {
            return _tokens;
        }
        int tokens(int i) {
            return _tokens[i];
        }
        const std::string chars() const {
            return _chars;
        }
        const char chars(int i) const {
            return _chars[i];
        }

        int* sample(int num_batches, int start, int length) {
            if(start+length*num_batches >= _size) {
                std::cerr << "Cannot sample " << num_batches << " x " << length << " tokens starting at index: " << start << "! Only " << _size << " tokens\n";
                exit(EXIT_FAILURE);
            }

            int* out = (int*)malloc(sizeof(int)*length*num_batches);
            for(int i = 0; i < length*num_batches; i++) { out[i] = _tokens[i+start]; }
            return out; 
        }

        std::string pretty() {
            std::string y = "[";
            for(int i = 0; i < _size-1; i++) {
                y += std::to_string(_tokens[i]) + ", ";
            }
            if(_size > 0) y += std::to_string(_tokens[_size-1]);
            y += "]";
            return y;
        }
};

#endif