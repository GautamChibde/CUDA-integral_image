#include "utils.h"
#include <stdio.h>

/***************************************************
 **   https://en.wikipedia.org/wiki/Grayscale     **
 **   For given input uchar4 RGBA image mapp each **
 **   output element of image as                  **
 **    I = .299f * R + .587f * G + .114f * B      **
 ***************************************************/

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols){
  int y = threadIdx.y+ blockIdx.y*BLOCK_SIZE_32;
  int x = threadIdx.x+ blockIdx.x*BLOCK_SIZE_32;
  if (y < numCols && x < numRows) {
  	int index = numRows*y +x;
    uchar4 color = rgbaImage[index];
    unsigned char grey = (unsigned char)(0.299f*color.x+ 0.587f*color.y + 0.114f*color.z);
    greyImage[index] = grey;
  }
}

void call_to_kernal_rgba_to_grey(uchar4 * const d_rgbaImage,
                                 unsigned char* const d_greyImage, 
                                 size_t numRows, 
                                 size_t numCols){
  const dim3 blockSize(BLOCK_SIZE_32, BLOCK_SIZE_32, 1);
  int   blocksX = numRows/BLOCK_SIZE_32+1;
  int   blocksY = numCols/BLOCK_SIZE_32+1; 
  const dim3 gridSize( blocksX, blocksY, 1);  
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}