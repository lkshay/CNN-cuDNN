#include <cudnn.h>
#include <cstdlib>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
using namespace std;
using namespace cv;

#define N 102400

// --- Define a function for cuDNN function status returns --- //

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

// --- Define a function for cuDNN function error returns --- //

#define checkCudaErrors(status) do {                         \
    std::stringstream _error;                                \
    if (status != 0) {                                       \
      _error << "Cuda failure: " << status;                  \
      FatalError(_error.str());                              \
    }                                                        \
} while(0)


__global__ void kernel1(int *a, int num){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(int i = idx; i < num; i += stride){

		a[i] = 2;
		//__syncthreads();
	}

}


int device = 0;

int main(int argc, const char* argv[]){

	int *a;
	
	cudaMallocManaged(&a, N*sizeof(int));

	cudaSetDevice(device);

	cudaGetDevice(&device);
	cout<<device<<endl;

	cudaMemPrefetchAsync(a, N*sizeof(int), device);

	int threads_per_block = 1024;
	int numBlocks = (N + threads_per_block - 1)/threads_per_block;

	kernel1<<<threads_per_block,numBlocks>>>(a,(int)N);

	cudaMemPrefetchAsync(a,N*sizeof(int),cudaCpuDeviceId);
	
	cudaDeviceSynchronize();

	int error = 0;
	for(int i = 0; i < N; i++){

		if(a[i] != 2){
			error = 1;
		}
		//cout<<a[i] << " "<< i<< endl;
	}

	cout << error << endl;
	cudaFree(a);
	return 0;
}