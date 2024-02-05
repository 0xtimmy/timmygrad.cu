#ifndef CUDAHELPERS_H
#define CUDAHELPERS_H

#include<string>
#include<iostream>
#include<fstream>

std::string readFile(char* filename) {
    std::ifstream file(filename);

    if(!file.is_open()) {
        std::cerr << "Could not open file: \"" << filename << "\"\n";
        throw std::runtime_error("");
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    return content;
}

void cce(cudaError_t error, std::string msg="") {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error -- " << msg << ":" << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void cce(std::string msg="") {
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error -- " << msg << ":" << cudaGetErrorString(error);
        exit(EXIT_FAILURE);
    }
}

#endif