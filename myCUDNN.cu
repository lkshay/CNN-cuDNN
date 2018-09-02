#include <cudnn.h>
#include <cstdlib>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
using namespace std;
using namespace cv;
#define BATCH_SIZE 1

int gpu_id;
int device;


struct convDim_t{

  int Height;
  int Width;
  int Channels;
  int Batch;
};

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

struct convLayerSpec_t{

  cudnnTensorDescriptor_t input_desc;
  cudnnFilterDescriptor_t kernel_desc;
  cudnnConvolutionDescriptor_t convolution_desc;
  cudnnTensorDescriptor_t bias_desc;
  cudnnTensorDescriptor_t output_desc;
  cudnnConvolutionFwdAlgo_t convolution_algo;
  cudnnActivationDescriptor_t activation_desc;
  convDim_t outDim;

};

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


// --- A function to convert the image to a 3-D array to be passed into the input conv layer --- //

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
  convDim_t outDims;
  convDim_t inDims;
  kernelDim_t kernelDims;
  convLayerSpec_t layerSpecs;

  float* d_output{nullptr};

  

	ConvLayers( int index, float* inT, convDim_t inDim, kernelDim_t kdims, int a, int b,
		cudnnTensorFormat_t t_format, cudnnDataType_t d_type, cudnnConvolutionMode_t c_mode, cudnnActivationMode_t ActMode,cudnnHandle_t cud){

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

  void getConvLayerSpecs();

	float* buildLayer();
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

  int outDimHeight{0}, outDimWidth{0}, outDimChannels{0}, outDimBatchSize{0};

  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                 input_descriptor,
                                                 kernel_descriptor,
                                                 &outDimBatchSize,
                                                 &outDimChannels,
                                                 &outDimHeight,
                                                 &outDimWidth));


  cudnnTensorDescriptor_t bias_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&bias_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(bias_descriptor,
                                            TensorFormat,
                                            DataType,
                                            outDimBatchSize, outDimChannels,
                                            outDimHeight, outDimWidth));

  // ---Build the output Descriptor ---//

  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/TensorFormat,
                                      /*dataType=*/DataType,
                                      /*batch_size=*/outDimBatchSize,
                                      /*channels=*/outDimChannels,
                                      /*image_height=*/outDimHeight,
                                      /*image_width=*/outDimWidth));

  // -- Size references for next conv layer --- //

  outDims.Height = outDimHeight;
  outDims.Width = outDimWidth;
  outDims.Channels = outDimChannels;
  outDims.Batch = outDimBatchSize;

  // --- Determine the Convolution algorithm to be used in CNN layer ---//

  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  checkCUDNN(cudnnGetConvolutionForwardAlgorithm(CUDNN,
                                        input_descriptor,
                                        kernel_descriptor,
                                        convolution_descriptor,
                                        output_descriptor,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        /*memoryLimitInBytes=*/0,
                                        &convolution_algorithm));

  cudnnActivationDescriptor_t activation_descriptor;
  checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
  checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
                                        ActivationMode,
                                        CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/0));

  layerSpecs.input_desc = input_descriptor;
  layerSpecs.kernel_desc = kernel_descriptor;
  layerSpecs.convolution_desc = convolution_descriptor;
  layerSpecs.bias_desc = bias_descriptor;
  layerSpecs.output_desc = output_descriptor;
  layerSpecs.convolution_algo = convolution_algorithm;
  layerSpecs.activation_desc = activation_descriptor;
  layerSpecs.outDim = outDims;

  }

  float* ConvLayers::buildLayer(){

  	// --- Set up the memory required for the convolution --- //
  
	  size_t workspaceBytes{0};
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(CUDNN,
                                                     layerSpecs.input_desc,
                                                     layerSpecs.kernel_desc,
                                                     layerSpecs.convolution_desc,
                                                     layerSpecs.output_desc,
                                                     layerSpecs.convolution_algo,
                                                     &workspaceBytes));


    // Initialize bias and kernel tensors here //
    ////////////////////////////////
    /////////////////////////////////
    ////////////////////////////////
    ////////////////////////////
    /////////////////////////////////
    ////////////////////////////////
    //////////////////////////////
    /////////////////////////////
    ///////////////////////////
    /////////////////////////
    ///////////////////////
    ////////////////////////
    ////////////
    ////////////
    //
    ///////

    

    /*
    cerr << "Workspace size: " << (workspaceBytes / 1048576.0) << "MB"
            << endl;
    */

    // --- Allocate Memory in the GPU for layer operation --- //

    void* d_workspace{nullptr};
    cudaMallocManaged(&d_workspace, workspaceBytes);

    int out_bytes = layerSpecs.outDim.Batch * layerSpecs.outDim.Channels * layerSpecs.outDim.Height * layerSpecs.outDim.Width * sizeof(float);		// memory required for storing output of the layer
    
    cudaMallocManaged(&d_output, out_bytes);
    cudaMemset(d_output, 0, out_bytes);
  
    //////////////////////////////////////////////////////////////////////////

    //These 3 steps will be performed in each epoch: for forward prop

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
                                     output_descriptor,
                                     d_output));
	  
    checkCUDNN(cudnnActivationForward(CUDNN,
                                    activation_descriptor,
                                    &alph,
                                    output_descriptor,
                                    d_output,
                                    &bet,
                                    output_descriptor,
                                    d_output));

    checkCUDNN(cudnnAddTensor(CUDNN, &alph, bias_descriptor,
                                  biasTensor ,&bet, output_descriptor, d_output));

    cudnnDestroyActivationDescriptor(activation_descriptor);

    cudaDeviceSynchronize();
    //string str = "./generated_images/afterConvlayer" + to_string(layerIndex) + ".png";
    //save_image(str, d_output, outDimHeight, outDimWidth);
    
    //cudaFree(layerKernel);
    //cudaFree(d_output);
    //cudaFree(d_input);
    return d_output;

}


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
  cudnnCreate(&cudnn);

  convDim_t firstLayerInputDims; //dimensions of input to first layer

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

  kernelDim_t layerKernel1;

  layerKernel1.kernelSize = 3;
  layerKernel1.kernelHeight = 5;
  layerKernel1.kernelWidth = 5;
  layerKernel1.strideHeight = 1;
  layerKernel1.strideWidth = 1;
  layerKernel1.padHeight = 1;
  layerKernel1.padWidth = 1;
  layerKernel1.dilationHeight = 1;
  layerKernel1.dilationWidth = 1;

  

  // Now create the kernel tensor for layer 1





  /*
  This is the layout for convolutional layer constructor

  ConvLayers layer( int layerIndex,
  float *inputTensor;  
  float *kernelTensor,
  float *biasTensor,    
  int alph, int bet,
  cudnnHandle_t CUDNN,
  cudnnTensorFormat_t TensorFormat,
  cudnnDataType_t DataType,
  cudnnConvolutionMode_t ConvMode,
  cudnnActivationMode_t ActivationMode,
  convDim_t outDims,
  convDim_t inDims,
  kernelDim_t kernelDims);
  
  */




  cudaDeviceSynchronize();

	return 0;
}

