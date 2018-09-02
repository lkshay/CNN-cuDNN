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

