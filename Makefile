###################################
# These are the default install   #
# locations on most linux distros #
###################################

####################################
# 			OPENCV				   #
####################################
OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include
OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui

####################################
# 			CUDA				   #
####################################

NVCC=nvcc
CUDA_INCLUDEPATH=/usr/local/cuda-7.5/include
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64
GCC_OPTS=-O3 -Wall -Wextra -m64

project: main.o rgb_to_grey.o integral_image.o Makefile
	$(NVCC) -o out main.o rgb_to_grey.o integral_image.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)

main.o: main.cpp gputimer.h utils.h image_pre_process.cpp
	g++ -c main.cpp $(GCC_OPTS) -I $(OPENCV_INCLUDEPATH) -I $(CUDA_INCLUDEPATH)

####################################
# 			KERNALS				   #
####################################

rgb_to_grey.o: rgb_to_grey.cu utils.h
	nvcc -c rgb_to_grey.cu $(NVCC_OPTS)

integral_image.o:integral_image.cu utils.h
	nvcc -c integral_image.cu $(NVCC_OPTS)

clean:
	rm -f *.o out
