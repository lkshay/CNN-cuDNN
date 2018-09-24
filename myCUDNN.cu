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

// a function to set pool dimensions for a layer operation, if pool is true

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

  random_device rd{};
  mt19937 gen{rd()};  
  normal_distribution<> d{0,1}; 

  float* conv_output{nullptr}; // output of convolution
  float* poolTensor{nullptr};  // output of pooling layer, if exists
  void* d_workspace{nullptr};
  size_t workspaceBytes{0};

  int convOutDimHeight{0}, convOutDimWidth{0}, convOutDimChannels{0}, convOutDimBatchSize{0};
  int poolOutBatchSize{0}, poolOutChannels{0}, poolOutHeight{0}, poolOutWidth{0};

  bool POOL;  // True if pooling is to be done in this layer, otherwise False

  cudnnTensorDescriptor_t input_descriptor;
  cudnnFilterDescriptor_t kernel_descriptor;
  cudnnConvolutionDescriptor_t convolution_descriptor;
  cudnnTensorDescriptor_t bias_descriptor;
  cudnnTensorDescriptor_t convOutput_descriptor;
  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  cudnnActivationDescriptor_t activation_descriptor;
  cudnnPoolingDescriptor_t pooling_descriptor;
  cudnnTensorDescriptor_t poolTensor_descriptor;
	  
  /*
  Constructor overloading for initialiing the class object with or without pooling mode. If the user wants to use pooling layer, use the second signature, otherwise first.
  The ConvLayers class' objects behave differently when a pool layer is to be used and differently when pool isnt there!
  */

	ConvLayers( int index, float* inT, convDim_t inDim, kernelDim_t kdims, int a, int b, 
		cudnnTensorFormat_t t_format, cudnnDataType_t d_type, cudnnConvolutionMode_t c_mode, cudnnActivationMode_t ActMode, cudnnHandle_t cud){

    this->POOL = false;
    this->inputTensor = inT;
    this->inDims = inDim;
    this->kernelDims = kdims;
    
    this->layerIndex = index;
    this->alph = a; this->bet = b;
	this->TensorFormat = t_format;
	this->DataType = d_type;			
	this->ConvMode = c_mode;
    this->ActivationMode = ActMode;
	this->CUDNN = cud;	
	}

  ConvLayers( int index, float* inT, convDim_t inDim, kernelDim_t kdims, poolDim_t pdims, int a, int b, 
    cudnnTensorFormat_t t_format, cudnnDataType_t d_type, cudnnConvolutionMode_t c_mode, cudnnActivationMode_t ActMode,cudnnPoolingMode_t poolMode, cudnnHandle_t cud){

    this->POOL = true;
    this->inputTensor = inT;
    this->inDims = inDim;
    this->kernelDims = kdims;
    this->poolDims = pdims;
    this->layerIndex = index;
    this->alph = a; this->bet = b;
    this->TensorFormat = t_format;
    this->DataType = d_type;      
    this->ConvMode = c_mode;
    this->ActivationMode = ActMode;
    this->PoolingMode = poolMode;
    this->CUDNN = cud;  
  }

  	void getConvLayerSpecs();

	void buildConvLayer();

  	void fwdProp();

  	void bwdPropo();

};

  void ConvLayers::getConvLayerSpecs(){



  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                          TensorFormat,
                                          DataType,
                                          inDims.Batch,
                                          inDims.Channels, 
                                          inDims.Height, 
                                          inDims.Width));

   
  // --- Build the Kernel which is going to convolve over the input ---//
  
  
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        DataType,
                                        TensorFormat,
                                        kernelDims.kernelSize,
                                        inDims.Channels,
                                        kernelDims.kernelHeight,
                                        kernelDims.kernelWidth));

  // --- Build the Convolution descriptor --- //

  
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                            kernelDims.padHeight,
                                            kernelDims.padWidth,
                                            kernelDims.strideHeight,
                                            kernelDims.strideWidth,
                                            kernelDims.dilationHeight,
                                            kernelDims.dilationWidth,
                                            ConvMode,
                                            DataType));

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
  
  cout<<"Output image size "<<outDims.Batch<<" X "<<outDims.Height<<" X "<<outDims.Width<<" X "<<outDims.Channels<<endl;
  
  checkCUDNN(cudnnCreateTensorDescriptor(&bias_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(bias_descriptor,
                                            TensorFormat,
                                            DataType,
                                            convOutDimBatchSize,
                                           convOutDimChannels,
                                           convOutDimHeight,
                                           convOutDimWidth));

  // ---Build the output Descriptor ---//

  
  checkCUDNN(cudnnCreateTensorDescriptor(&convOutput_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(convOutput_descriptor,
                                        TensorFormat,
                                        DataType, 
                                        convOutDimBatchSize,
                                        convOutDimChannels,
                                        convOutDimHeight,
                                        convOutDimWidth));

  // -- Size references for next conv layer --- //

  

  // --- Determine the Convolution algorithm to be used in CNN layer ---//

  
  checkCUDNN(cudnnGetConvolutionForwardAlgorithm(CUDNN,
                                        input_descriptor,
                                        kernel_descriptor,
                                        convolution_descriptor,
                                        convOutput_descriptor,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        /*memoryLimitInBytes=*/0,
                                        &convolution_algorithm));

  
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

  // check if the user has asked to create a pooling layer for this conv layer
  if(POOL){

    if((outDims.Height - poolDims.Height)%2 != 0){
      poolDims.Height = (poolDims.Height == 2) ? 3 : 2;
    }

    if((outDims.Width - poolDims.Width)%2 != 0){
      poolDims.Width = (poolDims.Width == 2) ? 3 : 2;
    }

    
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

    }

    

  }

  void ConvLayers::buildConvLayer(){

  	// --- Set up the memory required for the convolution --- //
  
	  
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(CUDNN,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     convolution_descriptor,
                                                     convOutput_descriptor,
                                                     convolution_algorithm,
                                                     &workspaceBytes));

    // Initialize bias and kernel tensors here //

    // Bias
    cudaMallocManaged(&biasTensor, outDims.Channels * outDims.Batch * sizeof(float));
    cudaMemset(biasTensor,(float)BIAS_INIT_VAL,outDims.Channels * outDims.Batch * sizeof(float)); //initializing all the bias units to BIAS_INIT_VAL

    //Kernel
    //random_device rd{};
    //mt19937 gen{rd()};

    // for initialization of weights with gassian distribution with zero mean and variance as one
    //normal_distribution<> d{0,1}; 

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
      int poolSize =  outDims.Batch * outDims.Channels * outDims.Height * outDims.Width * sizeof(float);
      cudaMallocManaged(&poolTensor, poolSize); 
      cudaMemset(poolTensor, 0, poolSize);

    }

    /*
    cerr << "Workspace size: " << (workspaceBytes / 1048576.0) << "MB" << endl;
    */
}


void ConvLayers::fwdProp(){

  checkCUDNN(cudnnConvolutionForward(CUDNN,
                                      &alph,
                                      input_descriptor,
                                      inputTensor,
                                      kernel_descriptor,
                                      kernelTensor,
                                      convolution_descriptor,
                                      convolution_algorithm,
                                      d_workspace,
                                      workspaceBytes,
                                      &bet,
                                      convOutput_descriptor,
                                      conv_output));

  checkCUDNN(cudnnAddTensor(CUDNN, &alph, bias_descriptor,
                                  biasTensor ,&bet, convOutput_descriptor, conv_output));

  checkCUDNN(cudnnActivationForward(CUDNN,
                                      activation_descriptor,
                                      &alph,
                                      convOutput_descriptor,
                                      conv_output,
                                      &bet,
                                      convOutput_descriptor,
                                      conv_output));

  if(POOL){

    checkCUDNN(cudnnPoolingForward(CUDNN,
                                  pooling_descriptor,
                                  &alph,
                                  convOutput_descriptor,
                                  conv_output,
                                  &bet,
                                  poolTensor_descriptor,
                                  poolTensor));
  }


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

  public:
    cublasHandle_t CUBLAS;
    int inDims;
    float* inputTensor;
    float* outputTensor;
    int outDims;
    cudnnActivationMode_t ActivationMode;
    cudnnActivationDescriptor_t activationDesc;
    float *weights;
    float *bias;

    // Random nunmber generator for weights
    random_device rdd{};
    mt19937 generator{rdd()};
    normal_distribution<> dist{0,1};


    FCLayers( float* inputTensor_, int inDims_, int outDims_, cudnnActivationMode_t ActivationMode_, cublasHandle_t CUBLAS_){

      inputTensor = inputTensor_;
      inDims = inDims_;
      outDims = outDims_;
      CUBLAS = CUBLAS_;
      ActivationMode = ActivationMode_;

    }

    void getFCLayerSpecs();
    void buildFCLayer();

};


void FCLayers::getFCLayerSpecs(){

  checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
  checkCUDNN(cudnnSetActivationDescriptor(activationDesc,
                                        ActivationMode,
                                        CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/0));
}

void FCLayers::buildFCLayer(){

  // Initialization of weight matrix

  //Callibrator for weight initialization
  float callibrator = sqrt(2/inDims);


  float *hweights;
  hweights = (float*)malloc(sizeof(float)*inDims*outDims);
  for(int i = 0; i < outDims; i++){
    hweights[i*outDims + inDims] = dist(generator)*callibrator;
  }
 
  cudaMallocManaged(&weights, inDims*outDims*sizeof(float));
  cudaMemcpy(weights, hweights, inDims * outDims * sizeof(float),cudaMemcpyHostToDevice);
  
  // initialization of bias vector
  cudaMallocManaged(&bias,outDims*sizeof(float));
  cudaMemset(bias,BIAS_INIT_VAL,outDims*sizeof(float));

  cudaMallocManaged(&outputTensor,outDims*sizeof(float));
  cudaMemset(outputTensor,0,outDims*sizeof(float));
  
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
  cublasHandle_t cublas;
  cublasCreate(&cublas);


  convDim_t firstLayerInputDims; //dimensions of input to first layer
  //The input layer will be set here but will be given in each epoch. shift this for loop
  firstLayerInputDims.Height = image.rows;
  firstLayerInputDims.Width = image.cols;
  firstLayerInputDims.Channels = image.channels();
  firstLayerInputDims.Batch = BATCH_SIZE;
  
  float *input_layer1;
  cudaMallocManaged(&input_layer1,firstLayerInputDims.Height * firstLayerInputDims.Width * firstLayerInputDims.Channels * firstLayerInputDims.Batch * sizeof(float));
  cudaMemcpy(input_layer1,inputImage,sizeof(inputImage),cudaMemcpyHostToDevice);


  // start with the kernel specs
  kernelDim_t layerKernel1 = setKernelSpecs(3,5,5,1,1,1,1,1,1);

  ///////////////////////// set pooling specs like this, if there is a pooling layer after your conv layer/////////////////////////////////////////
  // poolDim_t poolDim1 = setPoolSpecs((bool)OVERLAP_POOLING); //setting a overlapping pool layer
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  /*
  This is the signature for conv layer without pooling layer after it.

    ConvLayers( int index, float* inT, convDim_t inDim, kernelDim_t kdims, int a, int b,
    cudnnTensorFormat_t t_format, cudnnDataType_t d_type, cudnnConvolutionMode_t c_mode, cudnnActivationMode_t ActMode,cudnnPoolingMode_t poolMode, cudnnHandle_t cud);

  This is the signature for conv layer with pooling layer after it

    ConvLayers( int index, float* inT, convDim_t inDim, kernelDim_t kdims, poolDim_t pdims, int a, int b,
    cudnnTensorFormat_t t_format, cudnnDataType_t d_type, cudnnConvolutionMode_t c_mode, cudnnActivationMode_t ActMode,cudnnPoolingMode_t poolMode, cudnnHandle_t cud); 
  */
  
  // Make the architecture

  // This is the convolutional layer constructor (without pooling layer)
  ConvLayers convlayer1(1, input_layer1, firstLayerInputDims, layerKernel1, alpha, beta, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, CUDNN_CROSS_CORRELATION,
                        CUDNN_ACTIVATION_RELU, cudnn);
  // with these functions, we allocate the space in GPU for layer operation. 
  convlayer1.getConvLayerSpecs();
  convlayer1.buildConvLayer(); // this is objectName.conv_output. If you have added a pool layer, use objectName.poolTensor as output of this layer for input of next layer 
                               // Here, memory is allocated and values is initialized to zero but no computation has been done yet.

  convlayer1.fwdProp();

  // At this point, we have defined the layer, but we havent implemented forward or backward prop. That will be done while we start training, while this is just defination
  // memory allocation for forward pass (backward pass to be implemented after ths)

  // create another layer, this time with pool, therefore the signature of conv constructor will be different from the previous layer (Declare poolDim_t if you want pooling layer)
  poolDim_t poolDim2 = setPoolSpecs((bool)OVERLAP_POOLING);

  kernelDim_t layerKernel2 = setKernelSpecs(3,6,6,1,1,1,1,1,1);
  ConvLayers convlayer2(2, convlayer1.conv_output, convlayer1.outDims, layerKernel2, poolDim2, alpha, beta, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, CUDNN_CROSS_CORRELATION,
                        CUDNN_ACTIVATION_RELU, CUDNN_POOLING_MAX, cudnn);
  convlayer2.getConvLayerSpecs();
  convlayer2.buildConvLayer();

  // build a fully connected layer
  int fclayer1_input_dims = convlayer2.outDims.Height * convlayer2.outDims.Width * convlayer2.outDims.Channels; 

  FCLayers fclayer1( convlayer2.poolTensor, fclayer1_input_dims, 100 , CUDNN_ACTIVATION_RELU, cublas);
  fclayer1.getFCLayerSpecs();
  fclayer1.buildFCLayer();

  int fclayer2_input_dims = fclayer1.outDims;

  FCLayers fclayer2(fclayer1.outputTensor, fclayer2_input_dims, 50, CUDNN_ACTIVATION_RELU, cublas); 
  fclayer2.getFCLayerSpecs();
  fclayer2.buildFCLayer();


  cudaDeviceSynchronize();

	return 0;
}