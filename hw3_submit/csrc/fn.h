#include<ndarray.h>

/* NN-module layers */
void fc_fwd(NDArray& input, NDArray& weight, NDArray& bias, 
    NDArray& output);
void fc_bwd(NDArray& input, NDArray& weight, NDArray& bias, NDArray& grad_output, 
    NDArray& grad_input, NDArray& grad_weight, NDArray& grad_bias);

NDArray& im2col(NDArray& img);
void conv2d_k33p1s1_fwd(NDArray& input, NDArray& kernel, NDArray& bias, 
    NDArray& output);
void conv2d_k33p1s1_bwd(
    NDArray& input, 
    NDArray& kernel, 
    NDArray& grad_output, 
    NDArray& grad_input,
    NDArray& grad_kernel, 
    NDArray& grad_bias);

void maxpooling2d_k22s2_fwd(NDArray& input, NDArray& output);
void maxpooling2d_k22s2_bwd(NDArray& input, NDArray& output, NDArray& grad_output, 
    NDArray& grad_input);


void softmax_fwd(NDArray& input, 
    NDArray& output);
scalar_t celoss_fwd(NDArray& prob, NDArray& tgt);
void softmax_ce_bwd(NDArray& prob, NDArray& tgt, scalar_t grad_loss,
    NDArray& grad_input);

void relu_fwd(NDArray& input, NDArray& output);
void relu_bwd(NDArray& input, NDArray& grad_output, NDArray& grad_input);

void sigmoid_fwd(NDArray& input, NDArray& output);
void sigmoid_bwd(NDArray& input, NDArray& grad_output, NDArray& grad_input);



