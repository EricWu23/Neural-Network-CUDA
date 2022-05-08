#ifndef TEST_GPU_H
#define TEST_GPU_H


#include "sequential.h"

/*
  Descrition: 
    This function test the Neural Network defined by Sequential_GPU & seq. and print out the testing loss
    Inputs:
            Sequential_GPU & seq ----- sequential model defines the model of neural network under test
            inp                  ----- pointer to the bsxn_input array, which stores input data for a batch in the flat array format
            targ                 ----- pointer to the bsxn_out array, which stores labels for a batch in the flat array format
            bs                   ----- batch size
            n_in                 ----- number of input features into the Neural Network for per training sample
            n_out                ----- number of output of the Neural Network
*/
void validate_gpu(Sequential_GPU & seq, float *inp, float *targ, int bs, int n_in, int n_out);


#endif