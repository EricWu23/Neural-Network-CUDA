#include "mse.h"
#include <iostream>


__global__
void mse_forward_gpu(float *inp, float *out, int sz_out){
    int ind = blockDim.x*blockIdx.x + threadIdx.x;

    if (ind < sz_out){
        atomicAdd(&out[sz_out], powf(inp[ind]-out[ind],2));//fdividef(,sz_out)
    }
}


__global__
void mse_backward_gpu(float *inp, float *out, int sz_out){
    int ind = blockDim.x*blockIdx.x + threadIdx.x;

    if (ind < sz_out){
        inp[ind] = 2*(inp[ind]-out[ind]) ;//fdividef(,sz_out)
    }
}


MSE_GPU::MSE_GPU(int _sz_out){
    sz_out = _sz_out;
    
    n_blocks = (sz_out + block_size - 1) / block_size;
}

/*
_inp --- pointer to prediction
_out --- pointer to label
*/
void MSE_GPU::forward(float *_inp, float *_out){
    inp = _inp;
    out = _out;
}

/*
    Description: This function calls the CUDA kernel mse_forward_gpu to compute the mean square error loss.

    Inputs:             
        _inp    ----- pointer to the 1-D array of size (bsxn_out) that stores the output layer's predictions
        _out    ----- pointer to the 1-D array of size (bsxn_out+1) that stores the target labels and with the last 
                      element being used to store the average square error loss.
 
*/
void MSE_GPU::_forward(float *_inp, float *_out){
    _out[sz_out] = 0.0f;
    //std::cout<<"n_blocks:"<< n_blocks<< std::endl;
    //std::cout<<"block_size:"<< block_size<< std::endl;
    mse_forward_gpu<<<n_blocks, block_size>>>(_inp, _out, sz_out);
    cudaDeviceSynchronize();
    //std::cout<<"debug:"<< _out[5]<< std::endl;
    //std::cout<<"debug:"<< _out[10]<< std::endl;

}


void MSE_GPU::backward(){
    mse_backward_gpu<<<n_blocks, block_size>>>(inp, out, sz_out);
    cudaDeviceSynchronize();
}
