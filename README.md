# Timmygrad.cu

A simple autograd engine built in cuda c. Deisgned to be as simple as possible and expose how 1) control flow moves through the model and 2) how memory is handled.

![a console screenshot](https://github.com/0xtimmy/timmygrad.cu/blob/master/Screenshot%20from%202024-02-04%2022-52-51.png?raw=true)

## Getting started
Make sure you have a cuda-capable GPU before starting. Built on linux but it should compile on any platform.

Run to start training:
```
make run
```

## Code
`model.cuh` contains the code for setting up and runninh the model, including allocating tensors for intermediate values.

`tensor.uch` contains the code for executing operations and allocating memory. Each operation really only does three things: 1) run the operation, 2) add the operands as children of the result, and 3) set the backward operation. Adding the operands as children will assemble a computation graph through which we'll backpropogate. It also holds the `Tensor::backward()` function which will run through the computation graph we've built and compute each tensors gradient. The `Tensor::optimize()` function will use those gradients to train learnable tensors.

`algebra.cuh` contains the gpu kernels.