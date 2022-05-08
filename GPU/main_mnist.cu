#include <chrono>

#include "linear.h"
#include "relu.h"
#include "train.h"
#include "validate.h"
#include "../data/read_csv.h"

#define TOTAL_TRAINING_SAMPLE 10000
#define TRAIN_GBATCH_SIZE 1
#define NUM_OF_INPUT 784
#define NUM_OF_OUTPUT 10

#define TOTAL_TESTING_SAMPLE 1000
#define TEST_BATCH_SIZE 1000

int main(){
    std::chrono::steady_clock::time_point begin, end;

    int tbs = TOTAL_TRAINING_SAMPLE, n_in = NUM_OF_INPUT, n_epochs = 1;
    int n_hidden_1 = 512 ;
    int n_hidden_2 = 512;
    int n_out = NUM_OF_OUTPUT;

    float *inp, *targ;  
    cudaMallocManaged(&inp, tbs*n_in*sizeof(float));
    cudaMallocManaged(&targ, (TOTAL_TRAINING_SAMPLE*n_out+1)*sizeof(float));

    // reading training data
    begin = std::chrono::steady_clock::now();
    read_csv(inp, "../data/train_x.csv");
    read_csv(targ, "../data/train_y.csv");
    end = std::chrono::steady_clock::now();
    std::cout << "Data (Training) reading time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;
    
    int bs=TRAIN_GBATCH_SIZE;
    Linear_GPU* lin1 = new Linear_GPU(bs, n_in, n_hidden_1);
    ReLU_GPU* relu1 = new ReLU_GPU(bs*n_hidden_1);
    //Linear_GPU* lin2 = new Linear_GPU(bs, n_hidden_1, n_hidden_2);
    //ReLU_GPU* relu2 = new ReLU_GPU(bs*n_hidden_2);
    //Linear_GPU* lin3 = new Linear_GPU(bs, n_hidden_2, n_out);
    //std::vector<Module*> layers = {lin1, relu1, lin2, relu2, lin3};
    Linear_GPU* lin3 = new Linear_GPU(bs, n_hidden_1, n_out);
    std::vector<Module*> layers = {lin1, relu1,lin3};
    Sequential_GPU seq(layers);

    //Training:
    begin = std::chrono::steady_clock::now();
    train_gpu(seq,inp, targ, bs, n_in,n_out,n_epochs);
    end = std::chrono::steady_clock::now();
    std::cout << "Training time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;


 /*

    //Testing:
    int tbs_test = TOTAL_TESTING_SAMPLE;
    float *inp_test, *targ_test;  
    cudaMallocManaged(&inp_test, tbs_test*n_in*sizeof(float));
    cudaMallocManaged(&targ_test, (tbs_test*n_out+1)*sizeof(float));

    // read in testing data
    begin = std::chrono::steady_clock::now();
    read_csv(inp_test, "../data/test_x.csv");
    read_csv(targ_test, "../data/test_y.csv");
    end = std::chrono::steady_clock::now();
    std::cout << "Data (Validation) reading time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;

    int bs_test = TEST_BATCH_SIZE;

    begin = std::chrono::steady_clock::now();
    validate_gpu(seq,inp_test,targ_test,bs_test,n_in,n_out);
    end = std::chrono::steady_clock::now();
    std::cout << "Validation time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;
*/
    return 0;
}
