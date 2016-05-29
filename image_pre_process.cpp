#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

cv::Mat imageRGBA;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }

void preProcess(uchar4 **inputImage,
                uchar4 **d_rgbaImage,
				        unsigned char **d_greyImage,
                const std::string &filename) {
  //make sure the context initializes ok
  checkCudaErrors(cudaFree(0));

  cv::Mat image;
  image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }
  cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

  if (!imageRGBA.isContinuous()) {
    std::cerr << "Images isn't continuous!! Exiting." << std::endl;
    exit(1);
  }
  *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);

  const size_t numPixels = numRows() * numCols();
  //allocate memory on the device for both input and output
  checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));
                         
  //copy input array to the GPU
  checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));
}

void display_output(unsigned char **d_greyImage){
  cv::namedWindow("grey Image", cv::WINDOW_AUTOSIZE );
  cv::Mat greyImage = cv::Mat(numRows(),numCols(), CV_8UC1, *d_greyImage);;
  cv::imshow("grey Image", greyImage);
  cv::waitKey(0);
}