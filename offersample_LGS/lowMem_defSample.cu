#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>



#define BLOCK_H 4
#define BLOCK_W 8
#define BLOCK_HW BLOCK_H * BLOCK_W
#define CHANNEL_STRIDE 32


__forceinline__ __device__
bool within_bounds(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

template <typename scalar_t>
__global__ void lowMem_defSample_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap1,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap2,
    const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> coords,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> offset,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> corr,
    int r)
{
  const int b = blockIdx.x;
  const int h0 = blockIdx.y * blockDim.x;
  const int w0 = blockIdx.z * blockDim.y;
  const int tid = threadIdx.x * blockDim.y + threadIdx.y;

  const int H1 = fmap1.size(1);
  const int W1 = fmap1.size(2);
  const int H2 = fmap2.size(1);
  const int W2 = fmap2.size(2);
  const int N = coords.size(1);
  const int C = fmap1.size(3);

  __shared__ scalar_t f1[CHANNEL_STRIDE][BLOCK_HW];
  __shared__ scalar_t f2_Q11[CHANNEL_STRIDE][BLOCK_HW];
  __shared__ scalar_t f2_Q21[CHANNEL_STRIDE][BLOCK_HW];
  __shared__ scalar_t f2_Q12[CHANNEL_STRIDE][BLOCK_HW];
  __shared__ scalar_t f2_Q22[CHANNEL_STRIDE][BLOCK_HW];
  
  __shared__ float x2s[BLOCK_HW];
  __shared__ float y2s[BLOCK_HW];
  // __shared__ float ofs_x[BLOCK_HW];
  // __shared__ float ofs_y[BLOCK_HW];

  for (int c=0; c<C; c+=CHANNEL_STRIDE) {
    for (int k=0; k<BLOCK_HW; k+=BLOCK_HW/CHANNEL_STRIDE) {
      int k1 = k + tid / CHANNEL_STRIDE;
      int h1 = h0 + k1 / BLOCK_W;
      int w1 = w0 + k1 % BLOCK_W;
      int c1 = tid % CHANNEL_STRIDE;

      if (within_bounds(h1, w1, H1, W1))
        f1[c1][k1] = fmap1[b][h1][w1][c+c1];
      
      else
        f1[c1][k1] = 0.0;
    }

    __syncthreads();

    for (int n=0; n<N; n++) {
      int h1 = h0 + threadIdx.x;
      int w1 = w0 + threadIdx.y;
      int rd = 2*r + 1;    
      
      for (int iy=0; iy<rd; iy++) {
        for (int ix=0; ix<rd; ix++) {
            
          if (within_bounds(h1, w1, H1, W1)) {
            offset[b*n][h1][w1][static_cast<int>(rd/2)][static_cast<int>(rd/2)][0] = 0.0f;
            offset[b*n][h1][w1][static_cast<int>(rd/2)][static_cast<int>(rd/2)][1] = 0.0f;
            x2s[tid] = coords[b][n][h1][w1][0]+offset[b*n][h1][w1][ix][iy][0];
            y2s[tid] = coords[b][n][h1][w1][1]+offset[b*n][h1][w1][ix][iy][1];
          
        }
        __syncthreads();
            float dx= x2s[tid] - floor(x2s[tid]);
            float dy= y2s[tid] - floor(y2s[tid]);

          for (int k=0; k<BLOCK_HW; k+=BLOCK_HW/CHANNEL_STRIDE) {
            
            int k1 = k + tid / CHANNEL_STRIDE;
           
            int h2 = static_cast<int>(floor(y2s[k1])) - r + iy;
            int h2_high = h2+1;
            int w2 = static_cast<int>(floor(x2s[k1])) - r + ix;
            int w2_high = w2+1;
            int c2 = tid % CHANNEL_STRIDE;

            if (within_bounds(h2, w2, H2, W2))
              f2_Q11[c2][k1] = fmap2[b][h2][w2][c+c2];
            
            else
                f2_Q11[c2][k1] = static_cast<scalar_t>(0.0);
            
            if (within_bounds(h2, w2_high, H2, W2))
              f2_Q21[c2][k1] = fmap2[b][h2][w2_high][c+c2];
            
            else
                f2_Q21[c2][k1] = static_cast<scalar_t>(0.0);

            if (within_bounds(h2_high, w2, H2, W2))
              f2_Q12[c2][k1] = fmap2[b][h2_high][w2][c+c2];
            
            else
              f2_Q12[c2][k1] = static_cast<scalar_t>(0.0);

            if (within_bounds(h2_high, w2_high, H2, W2))
              f2_Q22[c2][k1] = fmap2[b][h2_high][w2_high][c+c2];
            
            else
                f2_Q22[c2][k1] = static_cast<scalar_t>(0.0);
          }

          __syncthreads();
      
          scalar_t Q11 = 0.0;
          scalar_t Q21 = 0.0;
          scalar_t Q12 = 0.0;
          scalar_t Q22 = 0.0;
          for (int k=0; k<CHANNEL_STRIDE; k++)
            {
                Q11 += f1[k][tid] * f2_Q11[k][tid];
                Q21 += f1[k][tid] * f2_Q21[k][tid];
                Q12 += f1[k][tid] * f2_Q12[k][tid];
                Q22 += f1[k][tid] * f2_Q22[k][tid];

            }
            // int ix_se = H1*W1*(iy + rd*ix);

            // scalar_t* corr_ptr = &corr[b][n][0][0][h1][w1];

          if (iy < rd && ix < rd && within_bounds(h1, w1, H1, W1))
          {
            corr[b][n][ix][iy][h1][w1] = Q11 * scalar_t((1.0f - dy) * (1.0f - dx)) + 
          Q21 * scalar_t((1.0f - dy) * dx) + 
          Q12 * scalar_t(dy * (1.0f - dx)) + 
          Q22 * scalar_t(dy * dx);
        }

        }
      } 
    }
  }
}



std::vector<torch::Tensor> lowMem_defSample_cuda(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  torch::Tensor offset,
  int radius)
{
  const auto B = coords.size(0);
  const auto N = coords.size(1);
  const auto H = coords.size(2);
  const auto W = coords.size(3);

  const auto rd = 2 * radius + 1;
  auto opts = fmap1.options();
  auto corr = torch::zeros({B, N, rd,rd, H, W}, opts);
  
  const dim3 blocks(B, (H+BLOCK_H-1)/BLOCK_H, (W+BLOCK_W-1)/BLOCK_W);
  const dim3 threads(BLOCK_H, BLOCK_W);


  AT_DISPATCH_FLOATING_TYPES_AND_HALF(fmap1.type(), "altcorr_forward_kernel", ([&] {
    lowMem_defSample_kernel<scalar_t><<<blocks, threads>>>(
        fmap1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        fmap2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        coords.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
        offset.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        corr.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        radius);
  }));

  return {corr};
}