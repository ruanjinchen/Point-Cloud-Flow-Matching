// third_party/PyTorchEMD/cuda/THC/THC.h
#pragma once
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// 兼容老接口的检查宏别名
#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif
#define AT_CHECK TORCH_CHECK

// 旧接口常传 state，我们给个占位即可
struct THCState {};

// 旧接口：获取当前 CUDA stream
static inline cudaStream_t THCState_getCurrentStream(THCState*) {
    return at::cuda::getCurrentCUDAStream();
}

// 旧接口：CUDA 错误检查
static inline void __thc_check(cudaError_t err, const char* file, int line) {
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel failed at ", file, ":", line,
                " code=", static_cast<int>(err), " ", cudaGetErrorString(err));
}
#define THCudaCheck(err) __thc_check((err), __FILE__, __LINE__)

// -------- 新增：老代码常用的 CHECK 宏族 --------
#define CHECK(cond) TORCH_CHECK((cond), "CHECK(", #cond, ") failed")

// 为了打印整型/size_t 更稳妥地强转到 long long
#define __TO_LL__(x) static_cast<long long>(x)

#define CHECK_EQ(a,b) TORCH_CHECK(((a) == (b)), \
    "CHECK_EQ failed: ", #a, "=", __TO_LL__(a), " vs ", #b, "=", __TO_LL__(b))
#define CHECK_NE(a,b) TORCH_CHECK(((a) != (b)), \
    "CHECK_NE failed: ", #a, " vs ", #b)
#define CHECK_LE(a,b) TORCH_CHECK(((a) <= (b)), \
    "CHECK_LE failed: ", #a, " vs ", #b)
#define CHECK_LT(a,b) TORCH_CHECK(((a) <  (b)), \
    "CHECK_LT failed: ", #a, " vs ", #b)
#define CHECK_GE(a,b) TORCH_CHECK(((a) >= (b)), \
    "CHECK_GE failed: ", #a, " vs ", #b)
#define CHECK_GT(a,b) TORCH_CHECK(((a) >  (b)), \
    "CHECK_GT failed: ", #a, " vs ", #b)

// 如老代码里有 LOG(FATAL) 也一并拦截
#define LOG_FATAL(msg) TORCH_CHECK(false, msg)
