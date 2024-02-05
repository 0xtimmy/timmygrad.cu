#ifndef TENSOR_H
#define TENSOR_H

#include<string>
#include<stdarg.h>
#include<map>
#include<set>
#include<list>
#include<iostream>
#include<functional>
#include<random>
#include"helpers.cuh"
#include"algebra.cuh"

const int NUMTHREADS = 32;

enum device_t { HOST, DEVICE };

class Shape {

    public:
        // fields
        int dim;
        int* shape;
        
        // constructors
        Shape() { }
        Shape(int _dim, ...) {
            va_list args;
            va_start(args, _dim);

            dim = _dim;
            shape = (int*)malloc(sizeof(int)*dim);

            for (int i = 0; i < dim; i++) {
                shape[i] = va_arg(args, int);
            }

            va_end(args);
        }

        ~Shape() {
            std::cerr << "destroying shape\n";
            free(shape);
        }

        // operators
        int& operator [] (int i) { return shape[i]; } 
        int size(int start=0, int end=-1) {
            if(end == -1) end = dim;
            int sz = 1;
            for(int i = start; i < end; i++) {
                sz *= shape[i];
            }
            return sz;
        }
};



class Tensor {

    std::set<Tensor*> _children;


    public:
        // fields
        std::string name;
        Shape* shape;
        
        std::function<void()> _backward;
        bool learnable;

        float* host_ptr;
        float* host_grad;
        float* dev_ptr;
        float* dev_grad;
        
        device_t device;
        
        // constructors
        Tensor() {}

        Tensor(std::string _name, Shape* _shape) {
            name = _name;
            shape = _shape;

            _children = std::set<Tensor*>();

            host_ptr = (float*)malloc(sizeof(float)*shape->size());
            host_grad = (float*)malloc(sizeof(float)*shape->size());
            cudaMalloc(&dev_ptr, sizeof(float)*shape->size());
            cudaMalloc(&dev_grad, sizeof(float)*shape->size());
            memset(host_ptr, 0, sizeof(float)*shape->size());
            memset(host_grad, 0, sizeof(float)*shape->size());

            device = HOST; 
            learnable = false;

            _backward = []() { };
        }

        Tensor(std::string _name, Shape* _shape, float* _host_ptr, float* _host_grad, float* _dev_ptr, float* _dev_grad) {
            name = _name;
            shape = _shape;

            _children = std::set<Tensor*>();

            host_ptr = _host_ptr;
            host_grad = _host_grad;
            dev_ptr = _dev_ptr;
            dev_grad = _dev_grad;

            device = HOST; 
            learnable = false;

            _backward = []() { };
        }

        ~Tensor() {
            free(host_ptr);
            free(host_grad);
            cudaFree(dev_ptr);
            cudaFree(dev_grad);
        }

        int size() { return shape->size(); }

        void view(Shape* _shape) {
            if(this->shape->size() != _shape->size()) {
                std::cerr << "shapes do not match in Tensor::view(): " << this->shape->size() << " and " << _shape->size() << "\n";
                exit(EXIT_FAILURE);
            }
            delete this->shape;
            this->shape = _shape;
        }

        void uniform(float min=0.0, float max=1.0) {
            if(device == HOST) {
                std::default_random_engine generator;
                std::uniform_real_distribution<double> distribution(min, max);

                for(int i = 0; i < size(); i++) {
                    host_ptr[i] = distribution(generator);
                }
            }
            else throw std::runtime_error("Tensor needs to be on the CPU");
        }

        void normal(float mu=0.0, float sigsquared=1.0) {
            if(device == HOST) {
                std::default_random_engine generator;
                std::normal_distribution<float> distribution(mu,sigsquared);
                for(int i = 0; i < size(); i++) {
                    host_ptr[i] = distribution(generator);
                }
            }
            else throw std::runtime_error("Tensor needs to be on the CPU");
        }

        void set(float val) {
            algebra::set<<<this->size()/NUMTHREADS+1, NUMTHREADS>>>(this->dev_ptr, val, this->size());
        }

        static Tensor* tri(int block_size) {
            Tensor* t = new Tensor("tri", new Shape(2, block_size, block_size));
            t->toDevice();
            algebra::tri<<<t->size()/NUMTHREADS+1, NUMTHREADS>>>(t->dev_ptr, block_size);
            return t;
        }

        // Device Management
        void toHost() {
            cce(cudaMemcpy(host_ptr, dev_ptr, sizeof(float)*shape->size(), cudaMemcpyDeviceToHost));
            cce(cudaMemcpy(host_grad, dev_grad, sizeof(float)*shape->size(), cudaMemcpyDeviceToHost));
        }

        void toDevice() {
            cce(cudaMemcpy(dev_ptr, host_ptr, sizeof(float)*shape->size(), cudaMemcpyHostToDevice));
            cce(cudaMemcpy(dev_grad, host_grad, sizeof(float)*shape->size(), cudaMemcpyHostToDevice));
        }

        static void build_graph(Tensor* t, std::set<Tensor*> *visited, std::list<Tensor*> *graph) {
            if(visited->find(t) == visited->end()) {
                visited->insert(t);
                for (Tensor* c : t->_children) {
                    build_graph(c, visited, graph);
                }
                graph->push_front(t);
            }
        }

        // Gradients
        void add_child(Tensor* t) {
            _children.insert(t);
        }

        void backward() {
            algebra::set<<<this->size()/NUMTHREADS+1, NUMTHREADS>>>(this->dev_grad, 1.0f, this->size());

            std::list<Tensor*> topo;
            std::set<Tensor*> visited;

            build_graph(this, &visited, &topo);
            for (Tensor* t : topo) {
                t->_backward();
            }
        }

        void optimize(float lp) {
            std::list<Tensor*> topo;
            std::set<Tensor*> visited;

            build_graph(this, &visited, &topo);
            for (Tensor* t : topo) {
                t->_optimize(lp);
            }
        }
        void _optimize(float lp) {
            if(learnable) {
                float* delta;
                cce(cudaMalloc(&delta, sizeof(float)*this->size())), "allocating optimization delta";
                algebra::mul<<<this->size()/NUMTHREADS+1, NUMTHREADS>>>(this->dev_grad, lp, delta, this->size());
                algebra::sub<<<this->size()/NUMTHREADS+1, NUMTHREADS>>>(this->dev_ptr, delta, this->size(), this->size());
                cce(cudaFree(delta));
            }
            cce(cudaMemset(this->dev_grad, 0, sizeof(float)*this->size()), "clearing gradients");
            this->_children.clear();
        }

        // Pretty
        std::string pretty() {
            toHost();
            int cursor = 0;
            return _pretty(&cursor, this->shape->dim, this->shape->shape, host_ptr);
            toDevice();
        }

        std::string pretty_shape() {
            std::string out = "[ ";
            //out += "shape ";
            for(int i = 0; i < this->shape->dim; i++) { out += std::to_string((*(this->shape))[i]) + " "; }
            out += "]";
            return out;
        }

        std::string pretty_graph(bool showgrad=false) {

            std::list<Tensor*> topo;
            std::set<Tensor*> visited;

            build_graph(this, &visited, &topo);

            std::string out = "";
            for (Tensor* t : topo) {
                std::cerr << t->name +"\n";
                t->toHost();
                out += t->name + "\n";
                if(showgrad) out += t->pretty() + "\n\n";
                t->toDevice();
            }
            return out;
        }

        std::string pretty_grad(bool showgrad=false) {
            std::string step = "\n<" + name + ">:\n";
            for(Tensor* c : this->_children) { step += "->" + c->name + "\n"; }
            step += ")\n";
            if(showgrad) {
                int cursor = 0;
                step += _pretty(&cursor, this->shape->dim, this->shape->shape, this->host_grad);
            }
            for(Tensor* c : this->_children) { 
                c->toHost();
                step += c->pretty_grad(showgrad); 
                c->toDevice();
            }
            return step;
        }

        // arithmetic ---------------------------------------------------------

        static void add(Tensor* a, Tensor* b, Tensor* y) {
            y->add_child(a);
            y->add_child(b);

            if(a->shape->size() >= b->shape->size()) {
                algebra::add<<<a->size()/NUMTHREADS+1, NUMTHREADS>>>(a->dev_ptr, b->dev_ptr, y->dev_ptr, a->size(), b->size());
            } else {
                algebra::add<<<b->size()/NUMTHREADS+1, NUMTHREADS>>>(b->dev_ptr, a->dev_ptr, y->dev_ptr, b->size(), a->size());
            }
            cce("add");

            y->_backward = [a, b, y]() {
                algebra::add_grad<<<a->size()/NUMTHREADS+1, NUMTHREADS>>>(y->dev_grad, a->dev_grad, y->size(), a->size());
                algebra::add_grad<<<b->size()/NUMTHREADS+1, NUMTHREADS>>>(y->dev_grad, b->dev_grad, y->size(), b->size());
                cce("add grad");
            };
        }

        static void mul(Tensor* a, Tensor* b, Tensor* y) {
            y->add_child(a);
            y->add_child(b);

            if(a->shape->size() >= b->shape->size()) {
                algebra::mul<<<a->size()/NUMTHREADS+1, NUMTHREADS>>>(a->dev_ptr, b->dev_ptr, y->dev_ptr, a->size(), b->size());
            } else {
                algebra::mul<<<b->size()/NUMTHREADS+1, NUMTHREADS>>>(b->dev_ptr, a->dev_ptr, y->dev_ptr, b->size(), a->size());
            }

            y->_backward = [a, b, y]() {
                algebra::mul_grad<<<a->size()/NUMTHREADS+1, NUMTHREADS>>>(y->dev_grad, a->dev_grad, b->dev_ptr, a->size(), b->size());
                algebra::mul_grad<<<b->size()/NUMTHREADS+1, NUMTHREADS>>>(y->dev_grad, b->dev_grad, a->dev_ptr, b->size(), a->size());
            };
        }

        static void div(Tensor* a, float b, Tensor* y) {
            y->add_child(a);

            algebra::div<<<a->size()/NUMTHREADS+1, NUMTHREADS>>>(a->dev_ptr, b, y->dev_ptr, a->size());

            y->_backward = [a, b, y]() {
                algebra::div_numerator_grad<<<a->size()/NUMTHREADS+1, NUMTHREADS>>>(y->dev_grad, a->dev_grad, b, a->size());
            };
        }

        static void mask(Tensor* x, Tensor* mask, Tensor* y) {
            y->add_child(x);
            y->add_child(mask);

            algebra::mask<<<x->size()/NUMTHREADS+1, NUMTHREADS>>>(x->dev_ptr, mask->dev_ptr, y->dev_ptr, mask->size(), x->size());

            y->_backward = [x, mask, y]() {
                algebra::mask_grad<<<x->size()/NUMTHREADS+1, NUMTHREADS>>>(y->dev_grad, x->dev_grad, mask->dev_ptr, mask->size(), x->size());
            };
        }

        // matrix
        
        static void mm(Tensor* a, Tensor* b, Tensor* y) {
            y->add_child(a);
            y->add_child(b);

            int m = (*(a->shape))[a->shape->dim-2];
            int n = (*(b->shape))[b->shape->dim-1];
            int inner = (*(a->shape))[a->shape->dim-1];

            if(inner != (*(b->shape))[b->shape->dim-2]) {
                std::cerr << "matmul must have matching inner dimentions: " << inner << " and " << (*(b->shape))[b->shape->dim-2] << " and shapes: ";
                std::cerr << "a: " << a->pretty_shape() << ", and b: " << b->pretty_shape() << "\n";
            }

            int a_batches = a->size()/inner/m;
            int b_batches = b->size()/inner/n;
            bool broadcasting;
            if (a_batches == b_batches) broadcasting = false;
            else if (b_batches == 1) broadcasting = true;
            else {
                std::cerr << "bad mm shapes, can't batch or broadcast: " << a->pretty_shape() << " @ " << b->pretty_shape() << "\n";
                exit(EXIT_FAILURE);
            }

            algebra::mm<<<y->size()/NUMTHREADS+1, NUMTHREADS>>>(a->dev_ptr, b->dev_ptr, y->dev_ptr, m, inner, n, a_batches, y->size(), broadcasting);
            cce("mm");


            y->_backward = [y, a, b, m, n, inner, a_batches, broadcasting]() {
                algebra::mm_in_grad<<<a->size()/NUMTHREADS+1, NUMTHREADS>>>(y->dev_grad, a->dev_grad, b->dev_ptr, m, inner, n, a_batches, a->size(), broadcasting);
                algebra::mm_out_grad<<<b->size()/NUMTHREADS+1, NUMTHREADS>>>(y->dev_grad, b->dev_grad, a->dev_ptr, m, inner, n, a_batches, b->size(), broadcasting);
                cce("mm grad");
            };
        }

        static void transpose(Tensor* x, Tensor* y, int dim1, int dim2) {
            y->add_child(x);

            int height = 0;
            int width = 0;
            int depth = 1;

            if(dim1 - dim2 == 1) {
                height = (*(x->shape))[dim2];
                width = (*(x->shape))[dim1];
                depth = 1;
                for(int i = dim1+1; i < x->shape->dim; i++) { depth *= (*(x->shape))[i]; }
            } else if (dim1 - dim2 == -1) {
                height = (*(x->shape))[dim1];
                width = (*(x->shape))[dim2];
                depth = 1;
                for(int i = dim2+1; i < x->shape->dim; i++) { depth *= (*(x->shape))[i]; }
            } else {
                std::cerr << "cry";
                exit(EXIT_FAILURE);
            }

            algebra::transpose<<<x->size()/NUMTHREADS+1, NUMTHREADS>>>(x->dev_ptr, y->dev_ptr, width, height, depth, x->size());
            cce("transpose");

            y->_backward = [y, x, width, height, depth]() {
                algebra::transpose<<<x->size()/NUMTHREADS+1, NUMTHREADS>>>(y->dev_grad, x->dev_grad, height, width, depth, x->size());
                cce("transpose grad");
            };
        }

        static void split(Tensor* x, Tensor** y, int splits, int dim=0) {
            if(dim > x->shape->dim) { 
                std::cerr << "Split dimension, " << dim << ", is greater than tensor dimension, " << x->shape->dim << "\n";
                exit(EXIT_FAILURE);
            }
            if((*(x->shape))[dim] < splits) {
                std::cerr << "There are more splits, " << splits << ", than there is space in dimension " << dim << ", " << (*(x->shape))[dim] << "\n";
                exit(EXIT_FAILURE);
            }
            int batches = 1;
            for (int i = 0; i < dim; i++) { batches *= (*(x->shape))[i]; }


            algebra::split<<<x->size()/NUMTHREADS+1, NUMTHREADS>>>(x->dev_ptr, y[0]->dev_ptr, batches, splits, x->size());
            cce("split");

            for(int i = 0; i < splits; i++) { 

                y[i]->_backward = [y, x, batches, splits, i]() {
                    algebra::split_grad<<<y[i]->size()/NUMTHREADS+1, NUMTHREADS>>>(y[i]->dev_grad, x->dev_grad, batches, splits, i, y[i]->size());
                };
                cce("split grad");
            }
        }

        // activation
        static void relu(Tensor* x, Tensor* y) {
            y->add_child(x);

            algebra::relu<<<x->size()/NUMTHREADS+1, NUMTHREADS>>>(x->dev_ptr, y->dev_ptr, x->size());
            cce("relu");

            y->_backward = [x, y]() {
                algebra::relu_grad<<<x->size()/NUMTHREADS+1, NUMTHREADS>>>(y->dev_grad, x->dev_grad, x->dev_ptr, x->size());
                cce("relu grad");
            };
        }

        // normalization

        static void softmax(Tensor* x, Tensor* y, int batches, int batchsize) {
            y->add_child(x);

            algebra::softmax<<<y->size()/NUMTHREADS+1, NUMTHREADS>>>(x->dev_ptr, y->dev_ptr, batchsize, batchsize*batches);
            cce("softmax");

            y->_backward = [x, y, batchsize, batches]() {
                algebra::softmax_grad<<<x->size()/NUMTHREADS+1, NUMTHREADS>>>(y->dev_grad, x->dev_grad, x->dev_ptr, batchsize, batchsize*batches);
                cce("softmax grad");
            };
        }

        static void layernorm(Tensor* x, Tensor* y, int layers, int layersize) {
            y->add_child(x);

            algebra::layernorm<<<y->size()/NUMTHREADS+1, NUMTHREADS>>>(x->dev_ptr, y->dev_ptr, layersize, layers*layersize);
            cce("layernorm");

            y->_backward = [x, y, layers, layersize]() {
                algebra::layernorm_grad<<<x->size()/NUMTHREADS+1, NUMTHREADS>>>(y->dev_grad, x->dev_grad, x->dev_ptr, y->dev_ptr, layersize, layers*layersize);
                cce("layernorm grad");
            };
        }


        // loss

        static void cross_entropy(Tensor* x, Tensor* y, int* dev_targets, int batches) {
            y->add_child(x);

            int batchsize = x->size() / batches;
            if(x->size() % batches != 0) {
                std::cerr << "uneven number of batches in cross entropy!!!";
                exit(EXIT_FAILURE);
            }

            algebra::cross_entropy<<<batches/NUMTHREADS+1,NUMTHREADS>>>(x->dev_ptr, dev_targets, y->dev_ptr, batchsize, batches);
            cce("cross entropy");

            y->_backward = [y, x, dev_targets, batchsize]() {
                algebra::cross_entropy_grad<<<x->size()/NUMTHREADS+1, NUMTHREADS>>>(y->dev_grad, x->dev_grad, x->dev_ptr, dev_targets, batchsize, x->size());
                cce("cross entropy grad");
            };
        }

        private:
            std::string _pretty(int *cursor, int dim, int* shape, float* data) {
                std::string out = ""; 
                if (dim == 1) {
                    out += "[";
                    for (int i = 0; i < shape[0] - 1; i++) {
                    out += std::to_string(data[*cursor + i]) + ", ";
                    }
                    out += std::to_string(data[*cursor + shape[0] - 1]) + "]\n";
                    *cursor += shape[0];
                    return out;
                }
                if (dim > 1) {
                    out += "[";
                    for(int i = 0; i < shape[0] - 1; i++) {
                        out += _pretty(cursor, dim-1, shape+1, data) + ", ";
                    }
                    out += _pretty(cursor, dim-1, shape+1, data) + "]";
                    return out;
                }
                return "";
            }
};

class Module {

    std::map<std::string, Tensor*> parameters;
    std::map<std::string, Module*> modules;
};

// Model Modules --------------------------------------------------------------


#endif