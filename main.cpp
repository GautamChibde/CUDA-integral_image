#include <iostream>
#include <stdio.h>
#include <fstream>
#include <math.h> 
#include <string.h>
#include <ctime>
#include <cstdlib>

#include "gputimer.h"
#include "utils.h"
#include "image_pre_process.cpp"

using namespace std;

void call_to_kernal_rgba_to_grey(uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, 
                            size_t numRows, size_t numCols);

void compute_integral_image(unsigned char* const d_rgbaImage,
                       float* const d_integralImage, 
                       size_t numRows, 
                       size_t numCols);

int main(int argc, char **argv){

	uchar4 *h_rgbaImage,*d_rgbaImage;
	unsigned char *d_greyImage;
	float *d_integralImage;
	float *h_integralImage;

	string input_file;
	string output_file;
	input_file = string(argv[1]);

	preProcess(&h_rgbaImage,&d_rgbaImage,&d_greyImage,input_file);

	call_to_kernal_rgba_to_grey(d_rgbaImage,d_greyImage,numRows(),numCols());

	size_t numPixels = numRows()*numCols();

	checkCudaErrors(cudaMalloc(&d_integralImage, sizeof(float) * numPixels));

	compute_integral_image(d_greyImage,d_integralImage,numRows(),numCols());

	h_integralImage = new float[numPixels];	
	
 	checkCudaErrors(cudaMemcpy(h_integralImage, d_integralImage, sizeof(float) * numPixels, cudaMemcpyDeviceToHost));
  	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

 	checkCudaErrors(cudaFree(d_rgbaImage));
  	checkCudaErrors(cudaFree(d_greyImage));
	return 0;
}