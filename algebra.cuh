#ifndef ALGEBRA_H
#define ALGEBRA_H

namespace algebra {
    // arthmetic
    __global__ void add(float* a, float* b, float* y, int sizea, int sizeb) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < sizea) {
            y[idx] = a[idx] + b[idx % sizeb];
        }
    }

    __global__ void add_grad(float* t, float* a, int sizet, int sizea) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < sizea) {
            float grad = 0.0;
            for(int i = 0; i < sizet/sizea; i++) { grad += t[i*sizea + idx]; }
            a[idx] += grad;
        }
    }

    __global__ void sub(float* a, float* b, int sizea, int sizeb) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < sizea) {
            a[idx] = a[idx] - b[idx % sizeb];
        }
    }

    __global__ void mul(float* a, float* b, float* y, int sizea, int sizeb) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < sizea) {
            y[idx] = a[idx] * b[idx % sizeb];
        }
    }

    __global__ void mul_grad(float* t, float* a_grad, float* b, int sizea, int sizeb) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < sizea) {
            if(sizea >= sizeb) {
                a_grad[idx] += b[idx % sizeb] * t[idx];
            } else {
                float grad = 0.0;
                for(int i = 0; i < sizeb/sizea; i++) {
                    grad += b[i*sizea + idx] * t[i*sizea + idx];
                }
                a_grad[idx] += grad;
            }
        }
    }

    __global__ void mul(float* x, float a, float* y, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            y[idx] = x[idx] * a;
        }
    }

    __global__ void mul_grad(float* t, float* a, float b, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if(idx < size) {
            a[idx] += b*t[idx];
        }
    }

    __global__ void div(float* a, float b, float* y, int sizea) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < sizea) {
            y[idx] = a[idx] / b;
        }
    }

    __global__ void div_numerator_grad(float* t, float* a_grad, float b, int sizea) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < sizea) {
            a_grad[idx] += t[idx] / b;
        }
    }
    // 

    __global__ void mm(float* a, float* b, float* y, int m, int inner, int n, int batches, int output_size, bool broadcasting) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batchsize = output_size / batches;
        int batch = idx / batchsize;
        int a_batch_offset = batch * (m*inner);
        int b_batch_offset = broadcasting ? 0 : batch * (n*inner);

        int row = (idx % batchsize) / n;
        int col = idx % n;

        if (idx < output_size) {
            float dot = 0;
            for (int i = 0; i < inner; i++) {
                dot += a[row * inner + i + a_batch_offset] *  b[i * n + col + b_batch_offset];
            }
            y[idx] = dot;
        }
    }

    __global__ void mm_in_grad(float* t_grad, float* a_grad, float* b, int m, int inner, int n, int batches, int size, bool broadcasting) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batchsize = size / batches;
        int batch = idx / batchsize;
        int a_batch_offset = batch * (m*n);
        int b_batch_offset = broadcasting ? 0 : batch * (n*inner);

        int row = (idx % batchsize) / inner;
        int col = idx % inner;

        if (idx < size) {
            if(broadcasting) {
                float sum = 0;
                for(int i = 0; i < n; i++) {
                    //sum += b[col][i] * t_grad[row][i];
                    sum += b[col*n + i] * t_grad[row*n + i + a_batch_offset];
                }
                a_grad[idx] += sum;
            } else {
                float sum = 0;
                for(int i = 0; i < n; i++) {
                    //sum += b[col][i] * t_grad[row][i];
                    sum += b[col*n + i + b_batch_offset] * t_grad[row*n + i + a_batch_offset];
                }
                a_grad[idx] += sum;
            }
        }
    }

    __global__ void mm_out_grad(float* t_grad, float* b_grad, float* a, int m, int inner, int n, int batches, int size, bool broadcasting) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size) {
            if(broadcasting) {
                int batchsize = size / batches;
                int row = idx / n;
                int col = idx % n;

                float sum = 0;
                for(int batch = 0; batch < batches; batch++) {
                    int t_batch_offset = batch * (m*n);
                    int a_batch_offset = batch * (m*inner);
                    for(int i = 0; i < m; i++) {
                        //sum += a[i][row] * t_grad[i][col];
                        sum += a[i*inner + row + a_batch_offset] * t_grad[i*n + col + t_batch_offset];
                    }
                }
                b_grad[idx] += sum; 
            } else {
                int batchsize = size / batches;
                int row = (idx % batchsize) / n;
                int col = idx % n;
                int batch = idx / batchsize;
                int b_batch_offset = batch * (m*n);
                int a_batch_offset = batch * (m*inner);

                float sum = 0;
                for(int i = 0; i < m; i++) {
                    //sum += a[i][row] * t_grad[i][col];
                    sum += a[i*inner + row + a_batch_offset] * t_grad[i*n + col + b_batch_offset];
                }
                b_grad[idx] += sum; 
            }
        }
    }

    __global__ void transpose(float* x, float* y, int width, int height, int depth, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int local_size = width * height * depth;
        int local = idx / local_size;
        int local_idx = idx % local_size;
        int row = (local_idx / depth / width) % height;
        int col = (local_idx / depth) % width;
        int deep = idx % depth;

        if(idx < size) y[local * local_size + col * height * depth + row * depth + deep] = x[idx];
    }

    __global__ void split(float* x, float* y, int batches, int splits, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int splitsize = size / splits;
        int split = idx / splitsize;

        int rowsize = size / batches;
        int split_rowsize = splitsize / batches;
        int split_row_idx = idx % split_rowsize;
        int split_row = (idx / split_rowsize) % batches; 

        if(idx < size) y[idx] = x[split_row*rowsize + split*split_rowsize + split_row_idx];
    }

    __global__ void split_grad(float* t, float* x_grad, int batches, int splits, int split, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int splitsize = size;
        int totalsize = size * splits;
        int rowsize = totalsize / batches;

        int split_rowsize = size / batches;
        int split_row_idx = idx % split_rowsize;
        int split_row = (idx / split_rowsize) % batches; 

        if(idx < size) x_grad[split_row*rowsize + split*split_rowsize + split_row_idx] += t[idx];
        //if(idx < size) x_grad[split_row*rowsize + split*split_rowsize + split_row_idx] = split_row;
    }

    __global__ void softmax(float* x, float* y, int batchsize, int size) {
        //extern __shared__ float _sum[];

        // channles must be reduced to calculate the sum; 
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batch = idx / batchsize;

       if(idx < size) {
            float sum = 0.0f;
            for(int i = 0; i < batchsize; i++) {
                    sum += expf(x[batch*batchsize + i]);
            }

            y[idx] = expf(x[idx]) / (sum + 0.000001);
       }  
    }

    __global__ void softmax_grad(float* t, float* x_grad, float* x, int batchsize, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batch = idx / batchsize;
        int batch_idx = idx % batchsize;

        if(idx < size) {
            float exp_x = expf(x[idx]);
            float _sum = 0;
            for(int i = 0; i < batchsize; i++) {
                _sum += expf(x[batch*batchsize+i]);
            }

            float grad = 0.0;
            for(int i = 0; i < batchsize; i++) {
                if(i == batch_idx) {
                    grad += t[batch*batchsize+i] * ((_sum-exp_x)*exp_x)/(_sum*_sum + 0.000001);
                } else {
                    grad -= t[batch*batchsize+i] * (expf(x[batch*batchsize+i])*exp_x)/(_sum*_sum + 0.000001);
                }
            }
            x_grad[idx] += grad; 
        }
    }

    __global__ void cross_entropy(float* x, int* targets, float* y, int batchsize, int size) {
        int batch = blockIdx.x * blockDim.x + threadIdx.x;

        if(batch < size) {
            float sum = 0.0;
            for(int i = 0; i < batchsize; i++) {
                sum += expf(x[batch*batchsize + i]);
            }

            float p = expf(x[batch*batchsize + targets[batch]]) / sum;

            y[batch] = -logf(p);
        }
    }

    __global__ void cross_entropy_grad(float* t, float* x_grad, float* x, int* targets, int batchsize, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batch = idx / batchsize;
        int batch_idx = idx % batchsize;

        if(idx < size) {
            float sum = 0.0;
            for(int i = 0; i < batchsize; i++) {
                sum += expf(x[batch*batchsize + i]);
            }

            float p = expf(x[idx]) / sum;

            float correct = (targets[batch] == batch_idx) ? 1.0 : 0.0;

            x_grad[idx] += t[batch] * (p - correct);

        }
    }

    __global__ void set(float* x, float val, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < size) {
            x[idx] = val;
        }
    }

    __global__ void mask(float* x, float* _mask, float* y, int masksize, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < size) {
            y[idx] = (_mask[idx % masksize] > 0.0) ? x[idx] : -INFINITY;
        }
    }

    __global__ void mask_grad(float*t, float* x_grad, float* _mask, int masksize, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < size) {
            x_grad[idx] += (_mask[idx % masksize] > 0.0) ? t[idx] : 0.0;
        }
    }

    __global__ void tri(float* x, int block_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int col = idx % block_size;
        int row = idx / block_size;
        if(idx < block_size*block_size) {
            x[idx] = (col <= row) ? 1.0f : -1.0f; 
        }
    }

    __global__ void max_idx(float* x, int* y, int batchsize, int batches) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < batches) {
            float max = x[idx*batchsize];
            int max_idx = 0;
            for (int i = 1; i < batchsize; i++) { 
                if(x[idx*batchsize + i] > max) {
                    max = x[idx*batchsize + i];
                    max_idx = i;
                }
            }
            y[idx] = max_idx;
        }
    }

    __global__ void onehot(int* x, float* y, int num_classes, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; 

        if(idx < size) {
            for(int i = 0; i < num_classes; i++) {
                y[idx*num_classes + i] = (i == x[idx]) ? 1.0f : 0.0f;
            }
        }
    }

    __global__ void layernorm(float* x, float* y, int layersize, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int layer = idx / layersize;
        int layer_idx = idx % layersize;

        if(idx < size) {
            float _mean = 0.0;
            float _variance = 0.0;

            for(int i = 0; i < layersize; i++) {
                _mean += x[layer*layersize + i];
            }
            _mean /= layersize;

            for (int i = 0; i < layersize; ++i) {
                float diff = x[layer * layersize + i] - _mean;
                _variance += diff * diff;
            }
            _variance /= layersize;

            y[idx] = (x[idx] - _mean) / (sqrtf(_variance + 0.00001));         
        }
    }

     __global__ void layernorm_grad(float* t, float* x_grad, float* x, float* y, int layersize, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int layer = idx / layersize;
        int layer_idx = idx % layersize;

        if(idx < size) {
            float _mean = 0.0;
            float _variance = 0.0;

            for(int i = 0; i < layersize; i++) {
                _mean += x[layer*layersize + i];
            }
            _mean /= layersize;

            for (int i = 0; i < layersize; ++i) {
                float diff = x[layer * layersize + i] - _mean;
                _variance += diff * diff;
            }
            _variance /= layersize;

            //x_grad[idx] += t[idx] / (sqrtf(_variance + 0.00001));
            float grad = 0.0;
            for(int i = 0; i < layersize; i++) {
                grad += (1.0f/(1.0f*layersize + sqrtf(_variance)))
                     * ((i == layer_idx ? 1.0f*layersize : 0.0) - 1 - y[layer * layersize + i] * y[idx])
                     * t[layer * layersize + i];
            }
            x_grad[idx] += grad;
        }
    }

    __global__ void relu(float* x, float* y, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if(idx < size) {
            y[idx] = x[idx] > 0.0 ? x[idx] : 0.0;
        }
    }

    __global__ void relu_grad(float* t, float* x_grad, float* x, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if(idx < size) {
            x_grad[idx] = x[idx] > 0.0 ? t[idx] : 0.0;
        }
    }

}

#endif