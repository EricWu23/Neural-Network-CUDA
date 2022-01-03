#include "sequential.h"
#include "../utils/utils.h"


void sequential_forward_gpu(float *inp, std::vector<Module*> layers, float *out){
    int sz_out;
    float *curr_out;

    for (int i=0; i<layers.size(); i++){
        Module *layer = layers[i];

        sz_out = layer->sz_out;

        cudaMallocManaged(&curr_out, sz_out*sizeof(float));//temporary mem for storing the current layer's forward outpt.
        layer->forward(inp, curr_out);//linear.forward does not modify inp, it only stores inp inside linear.inp. Similarly, curr_out is stored in linear.out

        inp = curr_out;// the output of current layer (curr_out) is the input of next layer. Question: does modify inp changes the pointer beinng orignally pass in this function? No. It is just a copied pointer of the pointer that was passed in as argument.
    }
    // kill the curr_out pointer
    cudaMallocManaged(&curr_out, sizeof(float));
    cudaFree(curr_out);
}


void sequetial_update_gpu(std::vector<Module*> layers){
    for (int i=layers.size()-1; 0<=i; i--){
        Module *layer = layers[i];

        layer->update(); 
        layer->backward();
    }
}


Sequential_GPU::Sequential_GPU(std::vector<Module*> _layers){
    layers = _layers;
}


void Sequential_GPU::forward(float *inp, float *out){
    sequential_forward_gpu(inp, layers, out);
}


void Sequential_GPU::update(){
    sequetial_update_gpu(layers);
}
