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
inline void CUDAErrorCheck(cudaError_t err,const char * name){
 
    if(err!= cudaSuccess)
    {
      std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
      exit(-1);
    }
}
void train_gpu(Sequential_GPU & seq, float *inp, float *targ, int bs, int n_in,int n_out, int batch_idx,int epoch_idx,int log_interval,int tbs){

    float* inp_shift=inp+batch_idx*bs*n_in;
    float* targ_shft=targ+batch_idx*bs*n_out;

    int sz_out = bs*n_out;
    MSE_GPU mse(sz_out);
    
    int sz_inp = bs*n_in;
    float *cp_inp, *out;
    cudaError_t err=cudaMallocManaged(&cp_inp, sz_inp*sizeof(float));
    CUDAErrorCheck(err,"failed to allocate memory for input copy");

    set_eq(cp_inp, inp_shift, sz_inp);// create a deep copy of the inp as cp_inp

    seq.forward(cp_inp, out);// after runing lin1.inp, lin1.out,relu1.inp,relu1.out,lin2.inp, and lin2.out will contain the results from forward propogation
    mse.forward(seq.layers.back()->out, targ_shft);// dummy, store the argument passed in as mse.inp (y_hat), mse.out (targ, don't care)

    mse.backward();//update the mse.inp to be dJ/dy_hat. mse.out still stores the targ and the last digit as don't care
    seq.update();
    /* clean up temporary memory at the end of each batch*/
    seq.free();

    if(batch_idx%log_interval==0){
      seq.forward(cp_inp, out);
      mse._forward(seq.layers.back()->out, targ_shft);// compute the actual loss
      seq.free();
      std::cout << "Training Epoch:"<< epoch_idx << "| [finished size/traing size] : ["<< (batch_idx+1)*bs<<"/"<<tbs<< "] ("<<
      (int)((batch_idx+1)*bs*100.0/tbs)<<"%) | Training Loss:"<< targ_shft[sz_out] << std::endl;
    }
    cudaFree(cp_inp);

/* 
    std::cout<<"label:"<< 0<<": ";
    debug(targ,0,sz_out);// label
    std::cout<<"prediction:"<<0<<": ";
    debug(seq.layers.back()->out,0,sz_out);// estimation
*/  
}