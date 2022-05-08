#include <iostream>

#include "mse.h"
#include "train.h"
#include "../utils/utils.h"

void debug(float *arrayname,int n_sample,int sz_out){

 if ((n_sample+9)<sz_out){
    for(int i=0; i<10;i++){
        std::cout<< arrayname[n_sample+i] <<' ';
        if(i == 9){
          std::cout<<'\n';
        }
    }
 }        
}

void train_gpu(Sequential_GPU & seq, float *inp, float *targ, int bs, int n_in, int n_out, int n_epochs){
    int sz_out = bs*n_out;
    MSE_GPU mse(sz_out);
    
    int sz_inp = bs*n_in;
    float *cp_inp, *out;
    cudaMallocManaged(&cp_inp, sz_inp*sizeof(float));

    for (int i=0; i<n_epochs; i++){
        set_eq(cp_inp, inp, sz_inp);// create a deep copy of the inp as cp_inp

        seq.forward(cp_inp, out);// after runing lin1.inp, lin1.out,relu1.inp,relu1.out,lin2.inp, and lin2.out will contain the results from forward propogation
        mse.forward(seq.layers.back()->out, targ);// dummy, store the argument passed in as mse.inp (y_hat), mse.out (targ, don't care)
        mse._forward(seq.layers.back()->out, targ);// compute the actual loss

        /*
        std::cout<<"label:"<< 0<<": ";
        debug(targ,0,sz_out);// label
        std::cout<<"prediction:"<<0<<": ";
        debug(seq.layers.back()->out,0,sz_out);// estimation

        std::cout<<"sz_out:"<< mse.sz_out << std::endl;
        */
        std::cout<<"epoch:"<< i << " Training loss: " << targ[sz_out] << std::endl;
        
        mse.backward();//update the mse.inp to be dJ/dy_hat. mse.out still stores the targ and the last digit as don't care
        seq.update();

        for (int i=0; i<seq.layers.size(); i++){
            Module *layer = seq.layers[i];
            cudaFree(layer->out);
        }
    }
    
    seq.forward(inp, out);
    mse._forward(seq.layers.back()->out, targ);// compute the actual loss
    std::cout << "The final Training loss is: " << targ[sz_out] << std::endl;
}
