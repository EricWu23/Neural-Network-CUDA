#include <iostream>

#include "mse.h"
#include "validate.h"
#include "../utils/utils.h"


void validate_gpu(Sequential_GPU & seq, float *inp, float *targ, int bs, int n_in,int n_out){
    int sz_out = bs*n_out;
    MSE_GPU mse(sz_out);
    float *out;

    seq.forward(inp, out);
    mse._forward(seq.layers.back()->out, targ);// compute the actual loss
    std::cout << "The final Testing loss is: " << targ[sz_out] << std::endl;
}
