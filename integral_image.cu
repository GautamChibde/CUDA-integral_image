#include "utils.h"
#include <stdio.h>

/****************************************************************************************
 ***                          Matrix Transpose                                        *** 
 ***  https://devblogs.nvidia.com/parallelforall/efficient-matrix-transpose-cuda-cc/  ***
 ***                                                                                  ***
 ****************************************************************************************/

__global__ 
void transpose(float *A,float *B,int n,int m) {
	int x = threadIdx.x, y = threadIdx.y;
	
	int row = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;

	__shared__ float tile[BLOCK_SIZE_32][BLOCK_SIZE_32+1];
	
	if(row<n&&col<m){
		tile[y][x] = A[row*m+col];
	__syncthreads();
		B[row+col*n] = tile[y][x];
	}
}

/****************************************************************************************
 ***                            Prefix sum                                            *** 
 ***          http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html            ***
 ***                                                                                  ***
 ****************************************************************************************/
template<typename T>
__global__ void scan(T* input, float * output, float *aux, int len,int i,int M) {
    
    // put input into shared memory
    __shared__ float temp[BLOCK_SIZE_32*2];
    
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE_32;
    if (start + t < len)
        temp[t] = (float)input[start + t + i*M];    
    else
       temp[t] = 0;
    if (start + BLOCK_SIZE_32 + t < len)
        temp[BLOCK_SIZE_32 + t] = (float) input[start + BLOCK_SIZE_32 + t + i*M];
    else
       temp[BLOCK_SIZE_32 + t] = 0;
    __syncthreads();

    // Reduction
    int stride;
    for (stride = 1; stride <= BLOCK_SIZE_32; stride <<= 1) {
       int index = (t + 1) * stride * 2 - 1;
       if (index < 2 * BLOCK_SIZE_32)
          temp[index] += temp[index - stride];
       __syncthreads();
    }

    // Post reduction
    for (stride = BLOCK_SIZE_32 >> 1; stride; stride >>= 1) {
       int index = (t + 1) * stride * 2 - 1;
       if (index + stride < 2 * BLOCK_SIZE_32)
          temp[index + stride] += temp[index];
       __syncthreads();
    }

    if (start + t < len)
        output[start + t +i*M] = temp[t];
    if (start + BLOCK_SIZE_32 + t < len)
        output[start + BLOCK_SIZE_32 + t +i*M] = temp[BLOCK_SIZE_32 + t];

    if (aux && t == 0)
       aux[blockIdx.x] = temp[2 * BLOCK_SIZE_32 - 1];
}

__global__ void fixup(float* input, float *aux, int len,int i,int M) {
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE_32;
    if (blockIdx.x) {
       if (start + t < len)
            input[start + t +i*M] += aux[blockIdx.x - 1];
       if (start + BLOCK_SIZE_32 + t < len)
            input[start + BLOCK_SIZE_32 + t +i*M] += aux[blockIdx.x - 1];
    }
}

void compute_integral_image(unsigned char* const d_rgbaImage,
                       float* const d_integralImage, 
                       size_t numRows, 
                       size_t numCols){

	float *deviceAuxArray, *deviceAuxScannedArray,*d_temp;

	unsigned int N = numRows,M= numCols;

	unsigned int numElements = N*M;

	checkCudaErrors(cudaMalloc(&deviceAuxArray, (BLOCK_SIZE_32 << 1) * sizeof(float)));
  checkCudaErrors(cudaMalloc(&deviceAuxScannedArray, (BLOCK_SIZE_32 << 1) * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_temp, numElements * sizeof(float)));

  cudaMemset(d_integralImage, 0, numElements*sizeof(float));

 	const int 	blocksX = numRows/BLOCK_SIZE_32+1;
  const int   blocksY = numCols/BLOCK_SIZE_32+1; 

	const dim3 	dimblock(BLOCK_SIZE_32,BLOCK_SIZE_32);

	const dim3 	dimgrid(blocksX , blocksY);
	const dim3 	dimgrid2(blocksY , blocksX);

	int numBlocks = ceil((float)numElements/(BLOCK_SIZE_32*2));
  dim3 scanGrid(numBlocks, 1, 1);
  dim3 scanBlock(BLOCK_SIZE_32, 1, 1);
	
  //kernal scan
	forn(i,N){
        scan<unsigned char><<<scanGrid, scanBlock>>>(d_rgbaImage, d_integralImage, deviceAuxArray, M,i,M);
        cudaDeviceSynchronize();
        scan<float><<<dim3(1,1,1), scanBlock>>>(deviceAuxArray, deviceAuxScannedArray, NULL, BLOCK_SIZE_32 << 1,i,0);
        cudaDeviceSynchronize();
        fixup<<<scanGrid, scanBlock>>>(d_integralImage, deviceAuxScannedArray, M,i,M);
    }
  //Transpose
	transpose<<<dimgrid,dimblock>>>(d_integralImage,d_temp,N,M);

  //scan
	forn(i,M){
        scan<unsigned char><<<scanGrid, scanBlock>>>(d_rgbaImage, d_temp, deviceAuxArray, N,i,N);
        cudaDeviceSynchronize();
        scan<float><<<dim3(1,1,1), scanBlock>>>(deviceAuxArray, deviceAuxScannedArray, NULL, BLOCK_SIZE_32 << 1,i,0);
        cudaDeviceSynchronize();
        fixup<<<scanGrid, scanBlock>>>(d_temp, deviceAuxScannedArray, N,i,N);
    }
  //transpose
	transpose<<<dimgrid2,dimblock>>>(d_temp,d_integralImage,M,N);
  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());
}