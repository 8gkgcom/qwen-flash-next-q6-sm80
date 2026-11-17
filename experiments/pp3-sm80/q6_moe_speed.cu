// SPDX-License-Identifier: Apache-2.0
// Uses the pinned vllm-gguf-plugin kernels (derived from llama.cpp).
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "cuda_compat.h"
#include "ggml-common.h"
#include "vecdotq.cuh"
#include "mmvq.cuh"
#include "mmq.cuh"
#include "moe.cuh"
#include "moe_vec.cuh"
#include "quantize.cuh"

static void validate(const at::Tensor& X,const at::Tensor& W,int64_t type,int64_t row){
  TORCH_CHECK(X.is_cuda() && W.is_cuda() && X.get_device()==W.get_device());
  TORCH_CHECK(X.scalar_type()==at::kBFloat16 && W.scalar_type()==at::kByte);
  TORCH_CHECK(X.is_contiguous() && W.is_contiguous() && W.size(0)==512 && W.size(1)==row);
  TORCH_CHECK(type==8 || type==14);
  TORCH_CHECK(X.size(1)%(type==14?256:32)==0);
}
at::Tensor vector_moe(at::Tensor X,at::Tensor W,at::Tensor ids,int64_t top,int64_t type,int64_t row,int64_t tokens,int64_t rows){
  validate(X,W,type,row);TORCH_CHECK(ids.scalar_type()==at::kInt && ids.is_contiguous());
  TORCH_CHECK(rows==2 || rows==4 || rows==8 || rows==16);
  TORCH_CHECK(X.size(0)==tokens && ids.numel()==tokens*top);
  c10::cuda::CUDAGuard guard(X.device());auto stream=c10::cuda::getCurrentCUDAStream();
  int col=X.size(1),padded=(col+511)/512*512;
  auto Y=at::zeros({tokens*top,row},X.options());
  auto Q=at::empty({tokens,padded/32*9},X.options().dtype(at::kInt));
  quantize_row_q8_1_cuda<at::BFloat16>(X.data_ptr<at::BFloat16>(),Q.data_ptr(),col,tokens,stream);
  dim3 grid((row+rows-1)/rows,1,tokens*top),block(32,rows,1);
  if(type==8)
    moe_vec_q<at::BFloat16,QK8_0,QI8_0,block_q8_0,VDR_Q8_0_Q8_1_MMVQ,vec_dot_q8_0_q8_1><<<grid,block,0,stream>>>(W.data_ptr(),Q.data_ptr(),Y.data_ptr<at::BFloat16>(),ids.data_ptr<int>(),top,col,row,Q.stride(0));
  else
    moe_vec_q<at::BFloat16,QK_K,QI6_K,block_q6_K,VDR_Q6_K_Q8_1_MMVQ,vec_dot_q6_K_q8_1><<<grid,block,0,stream>>>(W.data_ptr(),Q.data_ptr(),Y.data_ptr<at::BFloat16>(),ids.data_ptr<int>(),top,col,row,Q.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return Y;
}
at::Tensor matrix_moe(at::Tensor X,at::Tensor W,at::Tensor sorted,at::Tensor experts,at::Tensor padded_tokens,int64_t type,int64_t row,int64_t top,int64_t tokens){
  validate(X,W,type,row);TORCH_CHECK(X.size(0)==tokens);
  TORCH_CHECK(sorted.scalar_type()==at::kInt && experts.scalar_type()==at::kInt && padded_tokens.scalar_type()==at::kInt);
  c10::cuda::CUDAGuard guard(X.device());auto stream=c10::cuda::getCurrentCUDAStream();
  int col=X.size(1),padded=(col+511)/512*512;
  auto Y=at::zeros({tokens*top,row},X.options());
  auto Q=at::empty({tokens,padded/32*9},X.options().dtype(at::kInt));
  quantize_row_q8_1_cuda<at::BFloat16>(X.data_ptr<at::BFloat16>(),Q.data_ptr(),col,tokens,stream);
  if(type==8)
    ggml_moe_q8_0_q8_1_cuda<at::BFloat16>(Q.data_ptr(),W.data_ptr(),Y.data_ptr<at::BFloat16>(),sorted.data_ptr<int>(),experts.data_ptr<int>(),padded_tokens.data_ptr<int>(),W.stride(0),col,row,tokens,padded,row,top,sorted.size(0),stream);
  else
    ggml_moe_q6_K_q8_1_cuda<at::BFloat16>(Q.data_ptr(),W.data_ptr(),Y.data_ptr<at::BFloat16>(),sorted.data_ptr<int>(),experts.data_ptr<int>(),padded_tokens.data_ptr<int>(),W.stride(0),col,row,tokens,padded,row,top,sorted.size(0),stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return Y;
}
TORCH_LIBRARY(q6_speed_cuda,m){
  m.def("vector_moe(Tensor X, Tensor W, Tensor ids, int top, int type, int row, int tokens, int rows) -> Tensor");
  m.def("matrix_moe(Tensor X, Tensor W, Tensor sorted, Tensor experts, Tensor padded_tokens, int type, int row, int top, int tokens) -> Tensor");
}
TORCH_LIBRARY_IMPL(q6_speed_cuda,CUDA,m){m.impl("vector_moe",&vector_moe);m.impl("matrix_moe",&matrix_moe);}
