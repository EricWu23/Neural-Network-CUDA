#include <chrono>

#include "linear.h"
#include "relu.h"
#include "train.h"
#include "../data/read_csv.h"


int main(){
    std::chrono::steady_clock::time_point begin, end;

    int tbs = 100000, n_in = 50, n_epochs = 3;
    int n_hidden = n_in/2;
    int n_out = 1;

    float *inp, *targ;  
    cudaMallocManaged(&inp, tbs*n_in*sizeof(float));
    cudaMallocManaged(&targ, (tbs+1)*sizeof(float));// declare 1 element more due to the fact that it is used to store loss computation result

    begin = std::chrono::steady_clock::now();
    read_csv(inp, "../data/x.csv");
    read_csv(targ, "../data/y.csv");
    end = std::chrono::steady_clock::now();
    std::cout << "Data reading time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;
    
    int bs=5;
    Linear_GPU* lin1 = new Linear_GPU(bs, n_in, n_hidden);
    ReLU_GPU* relu1 = new ReLU_GPU(bs*n_hidden);
    Linear_GPU* lin2 = new Linear_GPU(bs, n_hidden, n_out);
    std::vector<Module*> layers = {lin1, relu1, lin2};

    Sequential_GPU seq(layers);
    
    begin = std::chrono::steady_clock::now();
    train_gpu(seq,inp, targ, bs, n_in,n_out, n_epochs);
    end = std::chrono::steady_clock::now();
    std::cout << "Training time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;

    return 0;
}
