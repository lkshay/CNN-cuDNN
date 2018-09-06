#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <random>
#include <cmath>
using namespace std;
using namespace cv;
#define BATCH_SIZE 1
#define OVERLAP_POOLING 1
#define BIAS_INIT_VAL 0.001

int gpu_id;
int device;


int roundUp(int num, int den){

  return((num + den - 1 )/(den));

}

// a struct for outputs & inputs of convolutional layers (including max pool, if exists)

struct convDim_t{

  int Height;
  int Width;
  int Channels;
  int Batch;
};

// a struct to define pooling layers for a conv layer (if pool is true)

struct poolDim_t{

  int Height;
  int Width;
  int padHeight;
  int padWidth;
  int strideHeight;
  int strideWidth;
};

// a struct to define kernel dimensions for a conv operation in a layer

struct kernelDim_t{

  int kernelSize;
  int kernelHeight;
  int kernelWidth;
  int strideHeight;
  int strideWidth;
  int padHeight;
  int padWidth;
  int dilationHeight;
  int dilationWidth;
};

// a struct cintaining all the descriptor of an instance of the class conv layers (to be used later when network is made)

struct convLayerSpec_t{

  cudnnTensorDescriptor_t input_desc;
  cudnnFilterDescriptor_t kernel_desc;
  cudnnConvolutionDescriptor_t convolution_desc;
  cudnnTensorDescriptor_t bias_desc;
  cudnnTensorDescriptor_t output_desc;
  cudnnConvolutionFwdAlgo_t convolution_algo;
  cudnnActivationDescriptor_t activation_desc;
  cudnnPoolingDescriptor_t pooling_desc;
  cudnnTensorDescriptor_t poolTensor_desc;
  convDim_t outDim;
};

// a function to set the kernel params for a conv operation in a lyer

kernelDim_t setKernelSpecs(int size, int fheight, int fwidth, int sheight, int swidth, int pheight, int pwidth, int dheight, int dwidth){

  kernelDim_t layerKernel;
  layerKernel.kernelSize = size;
  layerKernel.kernelHeight = fheight;
  layerKernel.kernelWidth = fwidth;
  layerKernel.strideHeight = sheight;
  layerKernel.strideWidth = swidth;
  layerKernel.padHeight = pheight;
  layerKernel.padWidth = pwidth;
  layerKernel.dilationHeight = dheight;
  layerKernel.dilationWidth = dwidth;

  return layerKernel;
}

/*
int flagOverlap is a flag for setting dimensions ov poolDims. If it is 1, then F=3,S=2 otherwise F=2,S=2.
It is worth noting that there are only two commonly seen variations of the max pooling layer found in practice: 
A pooling layer with F=3,S=2 (also called overlapping pooling), and more commonly F=2,S=2. Pooling sizes with larger receptive fields are too destructive.
*/

// a function to set pool dimensions for a loyer operation, if pool is true

poolDim_t setPoolSpecs(bool flagOverlap){

  poolDim_t poolDims;

  if(flagOverlap){

    poolDims.Height = 3;
    poolDims.Width = 3;
    poolDims.padHeight = 1;
    poolDims.padWidth = 1;
    poolDims.strideHeight = 2;
    poolDims.strideWidth = 2;  
  }
  else{
    poolDims.Height = 2;
    poolDims.Width = 2;
    poolDims.padHeight = 1;
    poolDims.padWidth = 1;
    poolDims.strideHeight = 2;
    poolDims.strideWidth = 2;
  }

  return poolDims;
  
}


#define checkCUDNN(expression)                             \
{                                                          \
  cudnnStatus_t status = (expression);                     \
  if (status != CUDNN_STATUS_SUCCESS) {                    \
    std::cerr << "Error on line " << __LINE__ << ": "      \
              << cudnnGetErrorString(status) << std::endl; \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
}


// --- A function to convert the image to a array to be passed into the input conv layer --- //

float * image2array(Mat image){

  /*
  const int rows = image.rows;
  const int cols = image.cols;
  const int chans = image.channels();
  */

  float *imageArray = (float *)image.data;
  
  return imageArray;
}

void save_image(const string output_filename,
                float* buffer,
                int height,
                int width) {
  cv::Mat output_image(height, width, CV_32FC3, buffer);
  // Make negative values zero.
  cv::threshold(output_image,
                output_image,
                /*threshold=*/0,
                /*maxval=*/0,
                cv::THRESH_TOZERO);
  cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
  output_image.convertTo(output_image, CV_8UC3);
  cv::imwrite(output_filename, output_image);
  std::cerr << "Wrote output to " << output_filename << std::endl;
}

Mat load_image(const char* image_path) {
  cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
  image.convertTo(image, CV_32FC3);
  cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
  cerr << "Input Image: " << image.rows << " x " << image.cols << " x "
          << image.channels() << endl;
  return image;
}

/*

-Prepare the kernel and bias tensors before sending them to the next layer. Initialize the kernel with He. et. al., and bias with zero.
-The filter initializer is common to all the layers, so is the bias. The channel layout is also same except in some cases (will find out later)
-Input to the layer is now the dimension of kernel tensor, bias tensor, input tensor, and the initialized bias and kernel tensors and the
 and the input tensor either from a previous layer or the input image

*/

float alpha = 1.0;
float beta = 0.0;

class ConvLayers{

	// In addition to the inHeight, inWidth and inChannels pertaining to the output of the previous layer,
	// also have output object of the previous layer as a member of this class.
  public:

  float *inputTensor;                   // pointer to the input tensor. If this is an input layer, convert the cv::Mat image to a 3-D array first and then pass its pointer 
                                        // to the class constructor 
  float *kernelTensor;
  float *biasTensor;		
	int layerIndex;
  int alph, bet;
  cudnnHandle_t CUDNN;
  cudnnTensorFormat_t TensorFormat;
	cudnnDataType_t DataType;
	cudnnConvolutionMode_t ConvMode;
  cudnnActivationMode_t ActivationMode;
  cudnnPoolingMode_t PoolingMode;
  convDim_t outDims;
  convDim_t inDims;
  kernelDim_t kernelDims;
  poolDim_t poolDims;
  convLayerSpec_t layerSpecs;

  float* conv_output{nullptr};
  float* poolTensor{nullptr};
  void* d_workspace{nullptr};
  size_t workspaceBytes{0};

  int convOutDimHeight{0}, convOutDimWidth{0}, convOutDimChannels{0}, convOutDimBatchSize{0};
  int poolOutBatchSize{0}, poolOutChannels{0}, poolOutHeight{0}, poolOutWidth{0};

  bool POOL;  // True if pooling is to be done in this layer, otherwise False
  

  /*
  Constructor overloading for initialiing the class object with or without pooling mode. If the user wants to use pooling layer, use the second signature, otherwise first.  
  */

	ConvLayers( int index, float* inT, convDim_t inDim, kernelDim_t kdims, int a, int b, 
		cudnnTensorFormat_t t_format, cudnnDataType_t d_type, cudnnConvolutionMode_t c_mode, cudnnActivationMode_t ActMode, cudnnHandle_t cud){

    POOL = false;
    inputTensor = inT;
    inDims = inDim;
    kernelDims = kdims;
    
    layerIndex = index;
    alph = a; bet = b;
		TensorFormat = t_format;
		DataType = d_type;			
		ConvMode = c_mode;
    ActivationMode = ActMode;
		CUDNN = cud;	
	}


  ConvLayers( int index, float* inT, convDim_t inDim, kernelDim_t kdims, poolDim_t pdims, int a, int b, 
    cudnnTensorFormat_t t_format, cudnnDataType_t d_type, cudnnConvolutionMode_t c_mode, cudnnActivationMode_t ActMode,cudnnPoolingMode_t poolMode, cudnnHandle_t cud){

    POOL = true;
    inputTensor = inT;
    inDims = inDim;
    kernelDims = kdims;
    poolDims = pdims;
    layerIndex = index;
    alph = a; bet = b;
    TensorFormat = t_format;
    DataType = d_type;      
    ConvMode = c_mode;
    ActivationMode = ActMode;
    PoolingMode = poolMode;
    CUDNN = cud;  
  }

  void getConvLayerSpecs();

	void buildConvLayer();

  //void fwdProp();


};

  void ConvLayers::getConvLayerSpecs(){

  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/TensorFormat,
                                        /*dataType=*/DataType,
                                        /*batch_size=*/inDims.Batch,
                                        /*channels=*/inDims.Channels, 
                                        /*image_height=*/inDims.Height, 
                                        /*image_width=*/inDims.Width));

   
  // --- Build the Kernel which is going to convolve over the input ---//
  
  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                      /*dataType=*/DataType,
                                      /*format=*/TensorFormat,
                                      /*out_channels=*/kernelDims.kernelSize,
                                      /*in_channels=*/inDims.Channels,
                                      /*kernel_height=*/kernelDims.kernelHeight,
                                      /*kernel_width=*/kernelDims.kernelWidth));

  // --- Build the Convolution descriptor --- //

  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                           /*pad_height=*/kernelDims.padHeight,
                                           /*pad_width=*/kernelDims.padWidth,
                                           /*vertical_stride=*/kernelDims.strideHeight,
                                           /*horizontal_stride=*/kernelDims.strideWidth,
                                           /*dilation_height=*/kernelDims.dilationHeight,
                                           /*dilation_width=*/kernelDims.dilationWidth,
                                           /*mode=*/ConvMode,
                                           /*computeType=*/DataType));

  // --- This function returns the dimensions of the resulting 4D tensor of a 2D convolution,     //
  // ---given the convolution descriptor, the input tensor descriptor and the filter descriptor --- //

  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                 input_descriptor,
                                                 kernel_descriptor,
                                                 &convOutDimBatchSize,
                                                 &convOutDimChannels,
                                                 &convOutDimHeight,
                                                 &convOutDimWidth));
  
  outDims.Height = convOutDimHeight;
  outDims.Width = convOutDimWidth;
  outDims.Channels = convOutDimChannels;
  outDims.Batch = convOutDimBatchSize;
  

  cudnnTensorDescriptor_t bias_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&bias_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(bias_descriptor,
                                            TensorFormat,
                                            DataType,
                                            convOutDimBatchSize,
                                           convOutDimChannels,
                                           convOutDimHeight,
                                           convOutDimWidth));

  // ---Build the output Descriptor ---//

  cudnnTensorDescriptor_t convOutput_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&convOutput_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(convOutput_descriptor,
                                      /*format=*/TensorFormat,
                                      /*dataType=*/DataType,
                                      /*batch_size,Channels, Height, Width=*/ 
                                          convOutDimBatchSize,
                                                 convOutDimChannels,
                                                 convOutDimHeight,
                                                 convOutDimWidth));

  // -- Size references for next conv layer --- //

  

  // --- Determine the Convolution algorithm to be used in CNN layer ---//

  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  checkCUDNN(cudnnGetConvolutionForwardAlgorithm(CUDNN,
                                        input_descriptor,
                                        kernel_descriptor,
                                        convolution_descriptor,
                                        convOutput_descriptor,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        /*memoryLimitInBytes=*/0,
                                        &convolution_algorithm));

  cudnnActivationDescriptor_t activation_descriptor;
  checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
  checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
                                        ActivationMode,
                                        CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/0));


  /*
  Do some adjustment if the output dimension of pooling layer is not an integer (which will give an error) 
  Each dimension h and w of the output images is computed as followed:
  outputDim = 1 + (inputDim + 2*padding - windowDim)/poolingStride;

  */

  layerSpecs.input_desc = input_descriptor;
  layerSpecs.kernel_desc = kernel_descriptor;
  layerSpecs.convolution_desc = convolution_descriptor;
  layerSpecs.bias_desc = bias_descriptor;
  layerSpecs.output_desc = convOutput_descriptor;
  layerSpecs.convolution_algo = convolution_algorithm;
  layerSpecs.activation_desc = activation_descriptor;
  layerSpecs.outDim = outDims;

  // check if the user has asked to create a pooling layer for this conv layer
  if(POOL){

    if((outDims.Height - poolDims.Height)%2 != 0){
      poolDims.Height = (poolDims.Height == 2) ? 3 : 2;
    }

    if((outDims.Width - poolDims.Width)%2 != 0){
      poolDims.Width = (poolDims.Width == 2) ? 3 : 2;
    }

    cudnnPoolingDescriptor_t pooling_descriptor;
    checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_descriptor));
    checkCUDNN(cudnnSetPooling2dDescriptor(pooling_descriptor,
                                            PoolingMode,
                                            CUDNN_NOT_PROPAGATE_NAN,
                                            poolDims.Height,
                                            poolDims.Width,
                                            poolDims.padHeight,
                                            poolDims.padWidth,
                                            poolDims.strideHeight,
                                            poolDims.strideWidth));

    checkCUDNN(cudnnGetPooling2dForwardOutputDim(pooling_descriptor,
                                              convOutput_descriptor,
                                                  &poolOutBatchSize,
                                                  &poolOutChannels,
                                                  &poolOutHeight,
                                                  &poolOutWidth));

    cudnnTensorDescriptor_t poolTensor_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&poolTensor_descriptor));  
    checkCUDNN(cudnnSetTensor4dDescriptor(poolTensor_descriptor,
                                          TensorFormat,
                                          DataType,
                                          poolOutBatchSize,
                                          poolOutChannels,
                                          poolOutHeight,
                                          poolOutWidth));

    outDims.Batch = poolOutBatchSize;
    outDims.Channels = poolOutChannels;
    outDims.Height = poolOutHeight;
    outDims.Width = poolOutWidth;

    layerSpecs.pooling_desc = pooling_descriptor;
    layerSpecs.poolTensor_desc = poolTensor_descriptor;
    layerSpecs.outDim = outDims;
    }

  }

  void ConvLayers::buildConvLayer(){

  	// --- Set up the memory required for the convolution --- //
  
	  
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(CUDNN,
                                                     layerSpecs.input_desc,
                                                     layerSpecs.kernel_desc,
                                                     layerSpecs.convolution_desc,
                                                     layerSpecs.output_desc,
                                                     layerSpecs.convolution_algo,
                                                     &workspaceBytes));

    // Initialize bias and kernel tensors here //

    // Bias
    cudaMallocManaged(&biasTensor, layerSpecs.outDim.Channels * layerSpecs.outDim.Batch * sizeof(float));
    cudaMemset(biasTensor,(float)BIAS_INIT_VAL,layerSpecs.outDim.Channels * layerSpecs.outDim.Batch * sizeof(float)); //initializing all the bias units to BIAS_INIT_VAL

    //Kernel
    random_device rd{};
    mt19937 gen{rd()};

    // for initialization of weights with gassian distribution with zero mean and variance as one
    normal_distribution<> d{0,1}; 

    // callibrator to be multplied with the weights for scaling according to He. et. al. 
    float callibrator = (layerIndex != 1) ? sqrt(2 / (inDims.Channels * inDims.Height * inDims.Width)) : 1.0;

    float kernelTemplate[kernelDims.kernelHeight][kernelDims.kernelWidth];
    for(int i = 0; i < kernelDims.kernelHeight; i++){
      for(int j = 0; j < kernelDims.kernelWidth; j++){
        kernelTemplate[i][j] = d(gen) * callibrator;
      }
    }
    float hkernel[kernelDims.kernelSize][inDims.Channels][kernelDims.kernelHeight][kernelDims.kernelWidth];

    for(int i = 0; i < kernelDims.kernelSize; i++){
      for(int j = 0; j < inDims.Channels; j++){
        for(int k = 0; k < kernelDims.kernelHeight; k++){
          for(int l = 0; l < kernelDims.kernelWidth; l++){
            hkernel[i][j][k][l] = kernelTemplate[k][l];
          }
        }
      }
    }

    cudaMallocManaged(&kernelTensor, kernelDims.kernelSize * kernelDims.kernelHeight * kernelDims.kernelWidth * sizeof(float));
    cudaMemcpy(kernelTensor,hkernel,sizeof(hkernel),cudaMemcpyHostToDevice);
    
    // --- Allocate Memory in the GPU for layer operation --- //    
    cudaMallocManaged(&d_workspace, workspaceBytes);
    int convout_bytes = convOutDimBatchSize * convOutDimChannels * convOutDimHeight * convOutDimWidth * sizeof(float);    
 
    // memory required for storing output of the conv operation (after adding bias)
    cudaMallocManaged(&conv_output, convout_bytes);
    cudaMemset(conv_output, 0, convout_bytes);


    // set up memory for pool tensor if pool is true
    if(POOL){

      int poolSize =  layerSpecs.outDim.Batch * layerSpecs.outDim.Channels * layerSpecs.outDim.Height * layerSpecs.outDim.Width * sizeof(float);
      cudaMallocManaged(&poolTensor, poolSize); \
      cudaMemset(poolTensor, 0, poolSize);
    }

    /*
    cerr << "Workspace size: " << (workspaceBytes / 1048576.0) << "MB" << endl;
    */
}

/*
void ConvLayers::fwdProp(){

  checkCUDNN(cudnnConvolutionForward(CUDNN,
                                     &alph,
                                     layerSpecs.input_desc,
                                     inputTensor,
                                     layerSpecs.kernel_desc,
                                     kernelTensor,
                                     layerSpecs.convolution_desc,
                                     layerSpecs.convolution_algo,
                                     d_workspace,
                                     workspaceBytes,
                                     &bet,
                                     layerSpecs.output_desc,
                                     conv_output));

  checkCUDNN(cudnnActivationForward(CUDNN,
                                    layerSpecs.activation_desc,
                                    &alph,
                                    layerSpecs.output_desc,
                                    conv_output,
                                    &bet,
                                    layerSpecs.output_desc,
                                    conv_output));

  checkCUDNN(cudnnAddTensor(CUDNN, &alph, layerSpecs.bias_desc,
                                  biasTensor ,&bet, layerSpecs.output_desc, conv_output));
  
  if(POOL){
    checkCUDNN(cudnnPoolingForward(CUDNN, layerSpecs.pooling_desc, &alph, layerSpecs.output_desc,
                                       conv_output, &bet, layerSpecs.poolTensor_desc, poolTensor));
  }
  

}
*/


class FCLayers{

  int inDims;
  int ouDims;


};


int main(int argc, const char* argv[]){

	if (argc < 2) {
    cerr << "usage: conv <image> [gpu=0] [sigmoid=0]" << endl;
    exit(EXIT_FAILURE);
  }

  gpu_id = (argc > 2) ? std::atoi(argv[2]) : 0;
  std::cerr << "GPU: " << gpu_id << std::endl;

  cudaSetDevice(gpu_id);

	Mat image = load_image(argv[1]);

  float *inputImage = image2array(image);
	
	//--- Build the Handle for the present layer ---//
	//--- Common for one GPU Device, and all layers of CNN built on it ---//
	cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));
  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);


  convDim_t firstLayerInputDims; //dimensions of input to first layer
  //The input layer will be set here but will be given in each epoch. shift this for loop
  firstLayerInputDims.Height = image.rows;
  firstLayerInputDims.Width = image.cols;
  firstLayerInputDims.Channels = image.channels();
  firstLayerInputDims.Batch = BATCH_SIZE;
  
  float *input_layer1;
  cudaMallocManaged(&input_layer1,firstLayerInputDims.Height * firstLayerInputDims.Width * firstLayerInputDims.Channels * firstLayerInputDims.Batch * sizeof(float));
  cudaMemcpy(input_layer1,inputImage,sizeof(inputImage),cudaMemcpyHostToDevice);

  /*
  Declare the network architecture here
  Need to declare the input tensor, kernel tensor and bias tensor
  The input for the next layer is the output of current layer which is returned from the buildConvLayer function
  */

  // start with the kernel specs
  kernelDim_t layerKernel1 = setKernelSpecs(3,5,5,1,1,1,1,1,1);

  // set pooling specs like this, if there is a pooling layer after your conv layer
  // poolDim_t poolDim1 = setPoolSpecs((bool)OVERLAP_POOLING); //setting a overlapping pool layer

  /*
  This is the signature for conv layer without pooling layer in front it.

    ConvLayers( int index, float* inT, convDim_t inDim, kernelDim_t kdims, int a, int b, bool pool,
    cudnnTensorFormat_t t_format, cudnnDataType_t d_type, cudnnConvolutionMode_t c_mode, cudnnActivationMode_t ActMode,cudnnPoolingMode_t poolMode, cudnnHandle_t cud);

  This is the signature for conv layer with pooling layer in ffront of it

    ConvLayers( int index, float* inT, convDim_t inDim, kernelDim_t kdims, poolDim_t pdims, int a, int b, bool pool,
    cudnnTensorFormat_t t_format, cudnnDataType_t d_type, cudnnConvolutionMode_t c_mode, cudnnActivationMode_t ActMode,cudnnPoolingMode_t poolMode, cudnnHandle_t cud); 
  */
  
  // Make the architecture

  // This is the convolutional layer constructor (with no pooling layer)
  ConvLayers convlayer1(1, input_layer1, firstLayerInputDims, layerKernel1, alpha, beta, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, CUDNN_CROSS_CORRELATION,
                        CUDNN_ACTIVATION_SIGMOID, cudnn);
  // with these functions, we allocate the space in GPU for layer operation. 
  convlayer1.getConvLayerSpecs();
  convlayer1.buildConvLayer();

  // At this point, we have defined the layer, but we havent implemented forward or backward prop. That will be done while we start training, while this is just defination.

  cudaDeviceSynchronize();

	return 0;
}

