#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "squeezenet_params.h"
///#include "dog.h"
#include "hourglass.h"
//#include "tiger.h"
//#include "truck.h"


__global__ void maxpool(
    int input_size, int output_size,
    float *input_im, float *output_im){

  int channels = blockIdx.x * blockDim.x + threadIdx.x;

  input_im += channels*input_size*input_size;
  output_im += channels*output_size*output_size;

  for(int i=0; i<output_size; i++){

    for(int j=0; j<output_size; j++){

      float tmp = 0.0;

      for(int k =0; k<3; k++){

        for(int l = 0; l<3; l++){

          float value = input_im[(i * 2 + k) * input_size + j *2 +l];
          if(value > tmp)
            tmp = value;
        }
      }

      output_im[i*output_size + j] = tmp;

    }
  }
}

__global__ void conv3x3(
  int input_channels, int input_size,
  int pad, int stride, int start_channel,
  int output_size, float* input_im, float* filter_weight,
  float* filter_bias, float* output_im){

  int filter_index = blockIdx.x * blockDim.x + threadIdx.x;

  filter_weight += filter_index * input_channels * 9;
  float bias = filter_bias[filter_index];
  output_im += (start_channel + filter_index) * output_size * output_size;

  //loop over output feature map
  for(int i = 0; i < output_size; i++)
  {
    for(int j = 0; j < output_size; j++)
    {
      //compute one element in the output feature map
      float tmp = bias;

      //compute dot product of 2 input_channels x 3 x 3 matrix
      for(int k = 0; k < input_channels; k++)
      {
        for(int l = 0; l < 3; l++)
        {
          int h = i * stride + l - pad;
          for(int m = 0; m < 3; m++)
          {
            int w = j * stride + m - pad;
            if((h >= 0) && (h < input_size) && (w >= 0) && (w < input_size))
            {
              tmp += input_im[k * input_size * input_size + (i * stride + l - pad) * input_size + j * stride + m - pad] \
                               * filter_weight[9 * k + 3 * l + m];
            }
          }
        }
      }

      //add relu activation after conv
      output_im[i * output_size + j] = (tmp > 0.0) ? tmp : 0.0;
    }
  }
}


__global__ void conv1x1(
  int input_channels, int input_size,int threads,
  float* input_im, float* filter_weight,
  float* filter_bias, float* output_im){

  int filter_index = blockIdx.x * blockDim.x + threadIdx.x;
  if(filter_index<threads){
    filter_weight += filter_index * input_channels;

    float bias = filter_bias[filter_index];

    output_im += filter_index * input_size * input_size;//start_channel is for 1x1 feature map in fire layer

    //loop over output feature map
    //out
    for(int i = 0; i < input_size; i++)
    {
      for(int j = 0; j < input_size; j++)
      {
        float tmp = bias;
        for(int k = 0; k < input_channels; k++)
        {
          tmp += input_im[k * input_size * input_size + i * input_size + j] * filter_weight[k];
        }
        //add relu after conv
        output_im[i * input_size + j] = (tmp > 0.0) ? tmp : 0.0;
      }
    }
 }
}

//last layer use a 13 x 13 avgPool layer as classifier
//one class score per kernel
__global__ void avgpool(
  int threads,
  float* input_im,
  float* output_im)
{
  int class_index = blockIdx.x * blockDim.x + threadIdx.x;//get class score index
  if(class_index<threads){
    input_im += 169 * class_index;

    float tmp = 0.0f;

    for(int i = 0; i < 169; i++)
    {
      tmp += input_im[i];
    }

    output_im[class_index] = tmp / 169.0;
 }
}






void getLabel(unsigned int class_index);
void cleanup();

float * h_result_classifier = (float*)malloc((1000)*sizeof(float));
char class_label[201];

float * d_sample;
float * d_conv1_weight;
float * d_conv1_bias;
float * d_result_conv;
float * d_result_pool1;
float * d_result_block1_squeeze;
float * d_result_block1_expand;
float * d_result_pool2;
float * d_fire1_squeeze_weight;
float * d_fire1_squeeze_bias;
float * d_fire1_expand1x1_weight;
float * d_fire1_expand1x1_bias;
float * d_fire1_expand3x3_weight;
float * d_fire1_expand3x3_bias;

float * d_fire2_squeeze_weight;
float * d_fire2_squeeze_bias;
float * d_fire2_expand1x1_weight;
float * d_fire2_expand1x1_bias;
float * d_fire2_expand3x3_weight;
float * d_fire2_expand3x3_bias;

float * d_result_block2_squeeze;
float * d_result_block2_expand;
float * d_result_pool3;

float * d_fire3_squeeze_weight;
float * d_fire3_squeeze_bias;
float * d_fire3_expand1x1_weight;
float * d_fire3_expand1x1_bias;
float * d_fire3_expand3x3_weight;
float * d_fire3_expand3x3_bias;

float * d_fire4_squeeze_weight;
float * d_fire4_squeeze_bias;
float * d_fire4_expand1x1_weight;
float * d_fire4_expand1x1_bias;
float * d_fire4_expand3x3_weight;
float * d_fire4_expand3x3_bias;
float * d_result_block3_squeeze1;
float * d_result_block3_expand1;
float * d_result_block3_squeeze2;
float * d_result_block3_expand2;

float * d_fire5_squeeze_weight;
float * d_fire5_squeeze_bias;
float * d_fire5_expand1x1_weight;
float * d_fire5_expand1x1_bias;
float * d_fire5_expand3x3_weight;
float * d_fire5_expand3x3_bias;

float * d_fire6_squeeze_weight;
float * d_fire6_squeeze_bias;
float * d_fire6_expand1x1_weight;
float * d_fire6_expand1x1_bias;
float * d_fire6_expand3x3_weight;
float * d_fire6_expand3x3_bias;

float * d_fire7_squeeze_weight;
float * d_fire7_squeeze_bias;
float * d_fire7_expand1x1_weight;
float * d_fire7_expand1x1_bias;
float * d_fire7_expand3x3_weight;
float * d_fire7_expand3x3_bias;

float * d_fire8_squeeze_weight;
float * d_fire8_squeeze_bias;
float * d_fire8_expand1x1_weight;
float * d_fire8_expand1x1_bias;
float * d_fire8_expand3x3_weight;
float * d_fire8_expand3x3_bias;

float * d_classifier_conv_weight;
float * d_classifier_conv_bias;
float * d_result_classifier_conv;
float * d_result_classifier;

int main(){
  float time,total_time=0.0;
  cudaEvent_t start, stop;

  // conv1 and fire 1

  cudaMalloc(&d_sample,3*224*224*sizeof(float));
  cudaMalloc(&d_conv1_weight,sizeof(conv1_weight));
  cudaMalloc(&d_conv1_bias,sizeof(conv1_bias));
  cudaMalloc(&d_result_conv,sizeof(float) * (1 * 64 * 111 * 111));
  cudaMalloc(&d_result_pool1, sizeof(float) * (1 * 64 * 55 * 55));
  cudaMalloc(&d_result_block1_squeeze,sizeof(float) * (1 * 16* 55 * 55));
  cudaMalloc(&d_result_block1_expand,sizeof(float) * (1 * 128 * 55 * 55));
  cudaMalloc(&d_result_pool2,sizeof(float) * (1 * 128 * 27 * 27));
  cudaMalloc(&d_fire1_squeeze_weight,sizeof(fire1_squeeze_weight));
  //printf("%d\n",sizeof(fire1_squeeze_weight)/sizeof(float));
  //printf("%d\n",sizeof(fire1_squeeze_bias)/sizeof(float));
  cudaMalloc(&d_fire1_squeeze_bias,sizeof(fire1_squeeze_bias));
  cudaMalloc(&d_fire1_expand1x1_weight,sizeof(fire1_expand1x1_weight));
  //printf("fire1_expand1x1_weight:%d\n",sizeof(fire1_expand1x1_weight)/sizeof(float));
  cudaMalloc(&d_fire1_expand1x1_bias,sizeof(fire1_expand1x1_bias));
  //printf("fire1_expand1x1_bias:%d\n",sizeof(fire1_expand1x1_bias)/sizeof(float));
  cudaMalloc(&d_fire1_expand3x3_weight,sizeof(fire1_expand3x3_weight));
  //printf("fire1_expand3x3_weight:%d\n",sizeof(fire1_expand3x3_weight)/sizeof(float));
  cudaMalloc(&d_fire1_expand3x3_bias,sizeof(fire1_expand3x3_bias));
  //printf("fire1_expand3x3_bias:%d\n",sizeof(fire1_expand3x3_bias)/sizeof(float));

  //fire2


  cudaMalloc(&d_fire2_squeeze_weight,sizeof(fire2_squeeze_weight));
  //printf("fire2_squeeze_weight:%d\n",sizeof(fire2_squeeze_weight)/sizeof(float));
  cudaMalloc(&d_fire2_squeeze_bias,sizeof(fire2_squeeze_bias));
  //printf("fire2_squeeze_bias:%d\n",sizeof(fire2_squeeze_bias)/sizeof(float));
  cudaMalloc(&d_fire2_expand1x1_weight,sizeof(fire2_expand1x1_weight));
  //printf("fire2_expand1x1_weight:%d\n",sizeof(fire2_expand1x1_weight)/sizeof(float));
  cudaMalloc(&d_fire2_expand1x1_bias,sizeof(fire2_expand1x1_bias));
  //printf("fire2_expand1x1_bias:%d\n",sizeof(fire2_expand1x1_bias)/sizeof(float));
  cudaMalloc(&d_fire2_expand3x3_weight,sizeof(fire2_expand3x3_weight));
  //printf("fire2_expand3x3_weight:%d\n",sizeof(fire2_expand3x3_weight)/sizeof(float));
  cudaMalloc(&d_fire2_expand3x3_bias,sizeof(fire2_expand3x3_bias));
  //printf("fire2_expand3x3_bias:%d\n",sizeof(fire2_expand3x3_bias)/sizeof(float));


  //block2


  cudaMalloc(&d_result_block2_squeeze,sizeof(float) * (1 * 32 * 27 * 27));
  cudaMalloc(&d_result_block2_expand,sizeof(float) * (1 * 256 * 27 *27));
  cudaMalloc(&d_result_pool3,sizeof(float) * (1 * 256 * 13 * 13));


  //fire3


  cudaMalloc(&d_fire3_squeeze_weight,sizeof(fire3_squeeze_weight));
  //printf("fire3_squeeze_weight:%d\n",sizeof(fire3_squeeze_weight)/sizeof(float));
  cudaMalloc(&d_fire3_squeeze_bias,sizeof(fire3_squeeze_bias));
  //printf("fire3_squeeze_bias:%d\n",sizeof(fire3_squeeze_bias)/sizeof(float));
  cudaMalloc(&d_fire3_expand1x1_weight,sizeof(fire3_expand1x1_weight));
  //printf("fire3_expand1x1_weight:%d\n",sizeof(fire3_expand1x1_weight)/sizeof(float));
  cudaMalloc(&d_fire3_expand1x1_bias,sizeof(fire3_expand1x1_bias));
  //printf("fire3_expand1x1_bias:%d\n",sizeof(fire3_expand1x1_bias)/sizeof(float));
  cudaMalloc(&d_fire3_expand3x3_weight,sizeof(fire3_expand3x3_weight));
  //printf("fire3_expand3x3_weight:%d\n",sizeof(fire3_expand3x3_weight)/sizeof(float));
  cudaMalloc(&d_fire3_expand3x3_bias,sizeof(fire3_expand3x3_bias));
  //printf("fire3_expand3x3_bias:%d\n",sizeof(fire3_expand3x3_bias)/sizeof(float));

  //fire4

  cudaMalloc(&d_fire4_squeeze_weight,sizeof(fire4_squeeze_weight));
  //printf("fire4_squeeze_weight:%d\n",sizeof(fire4_squeeze_weight)/sizeof(float));
  cudaMalloc(&d_fire4_squeeze_bias,sizeof(fire4_squeeze_bias));
  //printf("fire4_squeeze_bias:%d\n",sizeof(fire4_squeeze_bias)/sizeof(float));
  cudaMalloc(&d_fire4_expand1x1_weight,sizeof(fire4_expand1x1_weight));
  //printf("fire4_expand1x1_weight:%d\n",sizeof(fire4_expand1x1_weight)/sizeof(float));
  cudaMalloc(&d_fire4_expand1x1_bias,sizeof(fire4_expand1x1_bias));
  //printf("fire4_expand1x1_bias:%d\n",sizeof(fire4_expand1x1_bias)/sizeof(float));
  cudaMalloc(&d_fire4_expand3x3_weight,sizeof(fire4_expand3x3_weight));
  //printf("fire4_expand3x3_weight:%d\n",sizeof(fire4_expand3x3_weight)/sizeof(float));
  cudaMalloc(&d_fire4_expand3x3_bias,sizeof(fire4_expand3x3_bias));
  //printf("fire4_expand3x3_bias:%d\n",sizeof(fire4_expand3x3_bias)/sizeof(float));
  cudaMalloc(&d_result_block3_squeeze1,sizeof(float) * (1 * 48 * 13 * 13));
  cudaMalloc(&d_result_block3_expand1,sizeof(float) * (1 * 384 * 13 * 13));
  cudaMalloc(&d_result_block3_squeeze2,sizeof(float) * (1 * 64 * 13 * 13));
  cudaMalloc(&d_result_block3_expand2,sizeof(float) * (1 * 512 * 13 * 13));

  //fire5


  cudaMalloc(&d_fire5_squeeze_weight,sizeof(fire5_squeeze_weight));
  //printf("fire5_squeeze_weight:%d\n",sizeof(fire5_squeeze_weight)/sizeof(float));
  cudaMalloc(&d_fire5_squeeze_bias,sizeof(fire5_squeeze_bias));
  //printf("fire5_squeeze_bias:%d\n",sizeof(fire5_squeeze_bias)/sizeof(float));
  cudaMalloc(&d_fire5_expand1x1_weight,sizeof(fire5_expand1x1_weight));
  //printf("fire5_expand1x1_weight:%d\n",sizeof(fire5_expand1x1_weight)/sizeof(float));
  cudaMalloc(&d_fire5_expand1x1_bias,sizeof(fire5_expand1x1_bias));
  //printf("fire5_expand1x1_bias:%d\n",sizeof(fire5_expand1x1_bias)/sizeof(float));
  cudaMalloc(&d_fire5_expand3x3_weight,sizeof(fire5_expand3x3_weight));
  //printf("fire5_expand3x3_weight:%d\n",sizeof(fire5_expand3x3_weight)/sizeof(float));
  cudaMalloc(&d_fire5_expand3x3_bias,sizeof(fire5_expand3x3_bias));
  //printf("fire5_expand3x3_bias:%d\n",sizeof(fire5_expand3x3_bias)/sizeof(float));

  //fire 6


  cudaMalloc(&d_fire6_squeeze_weight,sizeof(fire6_squeeze_weight));
  //printf("fire6_squeeze_weight:%d\n",sizeof(fire6_squeeze_weight)/sizeof(float));
  cudaMalloc(&d_fire6_squeeze_bias,sizeof(fire6_squeeze_bias));
  //printf("fire6_squeeze_bias:%d\n",sizeof(fire6_squeeze_bias)/sizeof(float));
  cudaMalloc(&d_fire6_expand1x1_weight,sizeof(fire6_expand1x1_weight));
  //printf("fire6_expand1x1_weight:%d\n",sizeof(fire6_expand1x1_weight)/sizeof(float));
  cudaMalloc(&d_fire6_expand1x1_bias,sizeof(fire6_expand1x1_bias));
  //printf("fire6_expand1x1_bias:%d\n",sizeof(fire6_expand1x1_bias)/sizeof(float));
  cudaMalloc(&d_fire6_expand3x3_weight,sizeof(fire6_expand3x3_weight));
  //printf("fire6_expand3x3_weight:%d\n",sizeof(fire6_expand3x3_weight)/sizeof(float));
  cudaMalloc(&d_fire6_expand3x3_bias,sizeof(fire6_expand3x3_bias));
  //printf("fire6_expand3x3_bias:%d\n",sizeof(fire6_expand3x3_bias)/sizeof(float));

  //fire 7


  cudaMalloc(&d_fire7_squeeze_weight,sizeof(fire7_squeeze_weight));
  //printf("fire7_squeeze_weight:%d\n",sizeof(fire7_squeeze_weight)/sizeof(float));
  cudaMalloc(&d_fire7_squeeze_bias,sizeof(fire7_squeeze_bias));
  //printf("fire7_squeeze_bias:%d\n",sizeof(fire7_squeeze_bias)/sizeof(float));
  cudaMalloc(&d_fire7_expand1x1_weight,sizeof(fire7_expand1x1_weight));
  //printf("fire7_expand1x1_weight:%d\n",sizeof(fire7_expand1x1_weight)/sizeof(float));
  cudaMalloc(&d_fire7_expand1x1_bias,sizeof(fire7_expand1x1_bias));
  //printf("fire7_expand1x1_bias:%d\n",sizeof(fire7_expand1x1_bias)/sizeof(float));
  cudaMalloc(&d_fire7_expand3x3_weight,sizeof(fire7_expand3x3_weight));
  //printf("fire7_expand3x3_weight:%d\n",sizeof(fire7_expand3x3_weight)/sizeof(float));
  cudaMalloc(&d_fire7_expand3x3_bias,sizeof(fire7_expand3x3_bias));
  //printf("fire7_expand3x3_bias:%d\n",sizeof(fire7_expand3x3_bias)/sizeof(float));

  //fire 8


  cudaMalloc(&d_fire8_squeeze_weight,sizeof(fire8_squeeze_weight));
  //printf("fire8_squeeze_weight:%d\n",sizeof(fire8_squeeze_weight)/sizeof(float));
  cudaMalloc(&d_fire8_squeeze_bias,sizeof(fire8_squeeze_bias));
  //printf("fire8_squeeze_bias:%d\n",sizeof(fire8_squeeze_bias)/sizeof(float));
  cudaMalloc(&d_fire8_expand1x1_weight,sizeof(fire8_expand1x1_weight));
  //printf("fire8_expand1x1_weight:%d\n",sizeof(fire8_expand1x1_weight)/sizeof(float));
  cudaMalloc(&d_fire8_expand1x1_bias,sizeof(fire8_expand1x1_bias));
  //printf("fire8_expand1x1_bias:%d\n",sizeof(fire8_expand1x1_bias)/sizeof(float));
  cudaMalloc(&d_fire8_expand3x3_weight,sizeof(fire8_expand3x3_weight));
  //printf("fire8_expand3x3_weight:%d\n",sizeof(fire8_expand3x3_weight)/sizeof(float));
  cudaMalloc(&d_fire8_expand3x3_bias,sizeof(fire8_expand3x3_bias));
  //printf("fire8_expand3x3_bias:%d\n",sizeof(fire8_expand3x3_bias)/sizeof(float));

  //classifier


  cudaMalloc(&d_classifier_conv_weight,sizeof(classifier_conv_weight));
  //printf("%d\n",sizeof(classifier_conv_weight)/sizeof(float));
  cudaMalloc(&d_classifier_conv_bias,sizeof(classifier_conv_bias));
  //printf("%d\n",sizeof(classifier_conv_bias)/sizeof(float));
  cudaMalloc(&d_result_classifier_conv,sizeof(float) * (1 * 1000 * 13 * 13));
  cudaMalloc(&d_result_classifier,sizeof(float) * 1000);

  printf("squeezenet starting\n");
  printf("conv1\n");
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord( start, 0 );
  cudaMemcpy(d_sample,sample,3*224*224*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv1_weight,conv1_weight,sizeof(conv1_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv1_bias,conv1_bias,sizeof(conv1_bias),cudaMemcpyHostToDevice);
  conv3x3<<<1,64>>>(3,224,0,2,0,111,d_sample,d_conv1_weight,d_conv1_bias,d_result_conv);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );


  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  maxpool<<<1,64>>>(111,55,d_result_conv,d_result_pool1);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire1_squeeze_weight,fire1_squeeze_weight,sizeof(fire1_squeeze_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire1_squeeze_bias,fire1_squeeze_bias,sizeof(fire1_squeeze_bias),cudaMemcpyHostToDevice);
  conv1x1<<<1,16>>>(64,55,16,d_result_pool1,d_fire1_squeeze_weight,d_fire1_squeeze_bias,d_result_block1_squeeze);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire1_expand1x1_weight,fire1_expand1x1_weight,sizeof(fire1_expand1x1_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire1_expand1x1_bias,fire1_expand1x1_bias,sizeof(fire1_expand1x1_bias),cudaMemcpyHostToDevice);
  conv1x1<<<1,64>>>(16,55,64,d_result_block1_squeeze,d_fire1_expand1x1_weight,d_fire1_expand1x1_bias,d_result_block1_expand);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );


  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire1_expand3x3_weight,fire1_expand3x3_weight,sizeof(fire1_expand3x3_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire1_expand3x3_bias,fire1_expand3x3_bias,sizeof(fire1_expand3x3_bias),cudaMemcpyHostToDevice);
  conv3x3<<<1,64>>>(16,55,1,1,64,55,d_result_block1_squeeze,d_fire1_expand3x3_weight,d_fire1_expand3x3_bias,d_result_block1_expand);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire2_squeeze_weight,fire2_squeeze_weight,sizeof(fire2_squeeze_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire2_squeeze_bias,fire2_squeeze_bias,sizeof(fire2_squeeze_bias),cudaMemcpyHostToDevice);
  conv1x1<<<1,16>>>(128,55,16,d_result_block1_expand,d_fire2_squeeze_weight,d_fire2_squeeze_bias,d_result_block1_squeeze);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );
  ///
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire2_expand1x1_weight,fire2_expand1x1_weight,sizeof(fire2_expand1x1_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire2_expand1x1_bias,fire2_expand1x1_bias,sizeof(fire2_expand1x1_bias),cudaMemcpyHostToDevice);
  conv1x1<<<1,64>>>(16,55,64,d_result_block1_squeeze,d_fire2_expand1x1_weight,d_fire2_expand1x1_bias,d_result_block1_expand);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire2_expand3x3_weight,fire2_expand3x3_weight,sizeof(fire2_expand3x3_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire2_expand3x3_bias,fire2_expand3x3_bias,sizeof(fire2_expand3x3_bias),cudaMemcpyHostToDevice);
  conv3x3<<<1,64>>>(16,55,1,1,64,55,d_result_block1_squeeze,d_fire2_expand3x3_weight,d_fire2_expand3x3_bias,d_result_block1_expand);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );


  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  maxpool<<<1,128>>>(55,27,d_result_block1_expand,d_result_pool2);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );



  //Block2
  //fire3
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire3_squeeze_weight,fire3_squeeze_weight,sizeof(fire3_squeeze_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire3_squeeze_bias,fire3_squeeze_bias,sizeof(fire3_squeeze_bias),cudaMemcpyHostToDevice);
  conv1x1<<<1,32>>>(128,27,32,d_result_pool2,d_fire3_squeeze_weight,d_fire3_squeeze_bias,d_result_block2_squeeze);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire3_expand1x1_weight,fire3_expand1x1_weight,sizeof(fire3_expand1x1_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire3_expand1x1_bias,fire3_expand1x1_bias,sizeof(fire3_expand1x1_bias),cudaMemcpyHostToDevice);
  conv1x1<<<1,128>>>(32,27,128,d_result_block2_squeeze,d_fire3_expand1x1_weight,d_fire3_expand1x1_bias,d_result_block2_expand);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire3_expand3x3_weight,fire3_expand3x3_weight,sizeof(fire3_expand3x3_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire3_expand3x3_bias,fire3_expand3x3_bias,sizeof(fire3_expand3x3_bias),cudaMemcpyHostToDevice);
  conv3x3<<<1,128>>>(32,27,1,1,128,27,d_result_block2_squeeze,d_fire3_expand3x3_weight,d_fire3_expand3x3_bias,d_result_block2_expand);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );




  //fire4
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire4_squeeze_weight,fire4_squeeze_weight,sizeof(fire4_squeeze_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire4_squeeze_bias,fire4_squeeze_bias,sizeof(fire4_squeeze_bias),cudaMemcpyHostToDevice);
  conv1x1<<<1,32>>>(256,27,32,d_result_block2_expand,d_fire4_squeeze_weight,d_fire4_squeeze_bias,d_result_block2_squeeze);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire4_expand1x1_weight,fire4_expand1x1_weight,sizeof(fire4_expand1x1_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire4_expand1x1_bias,fire4_expand1x1_bias,sizeof(fire4_expand1x1_bias),cudaMemcpyHostToDevice);
  conv1x1<<<1,128>>>(32,27,128,d_result_block2_squeeze,d_fire4_expand1x1_weight,d_fire4_expand1x1_bias,d_result_block2_expand);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );


  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire4_expand3x3_weight,fire4_expand3x3_weight,sizeof(fire4_expand3x3_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire4_expand3x3_bias,fire4_expand3x3_bias,sizeof(fire4_expand3x3_bias),cudaMemcpyHostToDevice);
  conv3x3<<<1,128>>>(32,27,1,1,128,27,d_result_block2_squeeze,d_fire4_expand3x3_weight,d_fire4_expand3x3_bias,d_result_block2_expand);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  maxpool<<<1,256>>>(27,13,d_result_block2_expand,d_result_pool3);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );



  //block3

  //fire5_squeeze
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire5_squeeze_weight,fire5_squeeze_weight,sizeof(fire5_squeeze_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire5_squeeze_bias,fire5_squeeze_bias,sizeof(fire5_squeeze_bias),cudaMemcpyHostToDevice);
  conv1x1<<<1,48>>>(256,13,48,d_result_pool3,d_fire5_squeeze_weight,d_fire5_squeeze_bias,d_result_block3_squeeze1);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire5_expand1x1_weight,fire5_expand1x1_weight,sizeof(fire5_expand1x1_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire5_expand1x1_bias,fire5_expand1x1_bias,sizeof(fire5_expand1x1_bias),cudaMemcpyHostToDevice);
  conv1x1<<<1,192>>>(48,13,192,d_result_block3_squeeze1,d_fire5_expand1x1_weight,d_fire5_expand1x1_bias,d_result_block3_expand1);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire5_expand3x3_weight,fire5_expand3x3_weight,sizeof(fire5_expand3x3_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire5_expand3x3_bias,fire5_expand3x3_bias,sizeof(fire5_expand3x3_bias),cudaMemcpyHostToDevice);
  conv3x3<<<1,192>>>(48,13,1,1,192,13,d_result_block3_squeeze1,d_fire5_expand3x3_weight,d_fire5_expand3x3_bias,d_result_block3_expand1);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );




  //fire6
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire6_squeeze_weight,fire6_squeeze_weight,sizeof(fire6_squeeze_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire6_squeeze_bias,fire6_squeeze_bias,sizeof(fire6_squeeze_bias),cudaMemcpyHostToDevice);
  conv1x1<<<1,48>>>(384,13,48,d_result_block3_expand1,d_fire6_squeeze_weight,d_fire6_squeeze_bias,d_result_block3_squeeze1);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire6_expand1x1_weight,fire6_expand1x1_weight,sizeof(fire6_expand1x1_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire6_expand1x1_bias,fire6_expand1x1_bias,sizeof(fire6_expand1x1_bias),cudaMemcpyHostToDevice);
  conv1x1<<<1,192>>>(48,13,192,d_result_block3_squeeze1,d_fire6_expand1x1_weight,d_fire6_expand1x1_bias,d_result_block3_expand1);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );


  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire6_expand3x3_weight,fire6_expand3x3_weight,sizeof(fire6_expand3x3_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire6_expand3x3_bias,fire6_expand3x3_bias,sizeof(fire6_expand3x3_bias),cudaMemcpyHostToDevice);
  conv3x3<<<1,192>>>(48,13,1,1,192,13,d_result_block3_squeeze1,d_fire6_expand3x3_weight,d_fire6_expand3x3_bias,d_result_block3_expand1);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );




  //fire7

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire7_squeeze_weight,fire7_squeeze_weight,sizeof(fire7_squeeze_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire7_squeeze_bias,fire7_squeeze_bias,sizeof(fire7_squeeze_bias),cudaMemcpyHostToDevice);
  conv1x1<<<1,64>>>(384,13,64,d_result_block3_expand1,d_fire7_squeeze_weight,d_fire7_squeeze_bias,d_result_block3_squeeze2);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );


  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire7_expand1x1_weight,fire7_expand1x1_weight,sizeof(fire7_expand1x1_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire7_expand1x1_bias,fire7_expand1x1_bias,sizeof(fire7_expand1x1_bias),cudaMemcpyHostToDevice);
  conv1x1<<<1,256>>>(64,13,256,d_result_block3_squeeze2,d_fire7_expand1x1_weight,d_fire7_expand1x1_bias,d_result_block3_expand2);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );


  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire7_expand3x3_weight,fire7_expand3x3_weight,sizeof(fire7_expand3x3_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire7_expand3x3_bias,fire7_expand3x3_bias,sizeof(fire7_expand3x3_bias),cudaMemcpyHostToDevice);
  conv3x3<<<1,256>>>(64,13,1,1,256,13,d_result_block3_squeeze2,d_fire7_expand3x3_weight,d_fire7_expand3x3_bias,d_result_block3_expand2);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );




  //fire8
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire8_squeeze_weight,fire8_squeeze_weight,sizeof(fire8_squeeze_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire8_squeeze_bias,fire8_squeeze_bias,sizeof(fire8_squeeze_bias),cudaMemcpyHostToDevice);
  conv1x1<<<1,64>>>(512,13,64,d_result_block3_expand2,d_fire8_squeeze_weight,d_fire8_squeeze_bias,d_result_block3_squeeze2);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire8_expand1x1_weight,fire8_expand1x1_weight,sizeof(fire8_expand1x1_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire8_expand1x1_bias,fire8_expand1x1_bias,sizeof(fire8_expand1x1_bias),cudaMemcpyHostToDevice);
  conv1x1<<<1,256>>>(64,13,256,d_result_block3_squeeze2,d_fire8_expand1x1_weight,d_fire8_expand1x1_bias,d_result_block3_expand2);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_fire8_expand3x3_weight,fire8_expand3x3_weight,sizeof(fire8_expand3x3_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_fire8_expand3x3_bias,fire8_expand3x3_bias,sizeof(fire8_expand3x3_bias),cudaMemcpyHostToDevice);
  conv3x3<<<1,256>>>(64,13,1,1,256,13,d_result_block3_squeeze2,d_fire8_expand3x3_weight,d_fire8_expand3x3_bias,d_result_block3_expand2);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  //Classifier
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
  cudaMemcpy(d_classifier_conv_weight,classifier_conv_weight,sizeof(classifier_conv_weight),cudaMemcpyHostToDevice);
  cudaMemcpy(d_classifier_conv_bias,classifier_conv_bias,sizeof(classifier_conv_bias),cudaMemcpyHostToDevice);
  conv1x1<<<4,256>>>(512,13,1000,d_result_block3_expand2,d_classifier_conv_weight,d_classifier_conv_bias,d_result_classifier_conv);
  float * result = (float*)malloc(sizeof(float) * (1 * 1000 * 13 * 13));
  cudaMemcpy(result,d_result_classifier_conv,sizeof(float) * (1 * 1000 * 13 * 13),cudaMemcpyDeviceToHost);
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  total_time += time;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );



    for(int i=0;i<1000;++i){
      float tmp =0;
      for(int j=0;j<169;++j){
        tmp = tmp + result[j];
      }
      h_result_classifier[i] = tmp/169;
      result = result + 169;
    }


  float tmp = 0.0f;
  unsigned int class_index = 0;
  for(int j = 0; j < 1000; j++)
  {
    if(h_result_classifier[j] > tmp)
    {
  	  tmp = h_result_classifier[j];
      class_index = j;
    }
  }

  getLabel(class_index);

  printf("\r\npredicted label: %s\r\n", class_label);

  cleanup();

  printf("Total Kernel execution time: %f\n",total_time);
  

  printf("done\n");

  return 0;
}

void getLabel(unsigned int class_index)
{
  int i;

  FILE *fp;

  fp = fopen("synset_words.txt", "r");
  for(i = 0; i < class_index + 1; i++)
  {
    fgets(class_label, sizeof(class_label), fp);
  }
  fclose(fp);
}

void cleanup(){
  cudaFree(d_sample);
  cudaFree(d_conv1_weight);
  cudaFree(d_conv1_bias);
  cudaFree(d_result_conv);
  cudaFree(d_result_pool1);
  cudaFree(d_result_block1_squeeze);
  cudaFree(d_result_block1_expand);
  cudaFree(d_fire1_squeeze_weight);
  cudaFree(d_fire1_squeeze_bias);
  cudaFree(d_fire1_expand1x1_weight);
  cudaFree(d_fire1_expand1x1_bias);
  cudaFree(d_fire1_expand3x3_weight);
  cudaFree(d_fire1_expand3x3_bias);

  cudaFree(d_fire2_squeeze_weight);
  cudaFree(d_fire2_squeeze_bias);
  cudaFree(d_fire2_expand1x1_weight);
  cudaFree(d_fire2_expand1x1_bias);
  cudaFree(d_fire2_expand3x3_weight);
  cudaFree(d_fire2_expand3x3_bias);

  cudaFree(d_result_block2_squeeze);
  cudaFree(d_result_block2_expand);
  cudaFree(d_result_pool2);

  cudaFree(d_fire3_squeeze_weight);
  cudaFree(d_fire3_squeeze_bias);
  cudaFree(d_fire3_expand1x1_weight);
  cudaFree(d_fire3_expand1x1_bias);
  cudaFree(d_fire3_expand3x3_weight);
  cudaFree(d_fire3_expand3x3_bias);

  cudaFree(d_fire4_squeeze_weight);
  cudaFree(d_fire4_squeeze_bias);
  cudaFree(d_fire4_expand1x1_weight);
  cudaFree(d_fire4_expand1x1_bias);
  cudaFree(d_fire4_expand3x3_weight);
  cudaFree(d_fire4_expand3x3_bias);

  cudaFree(d_result_pool3);
  cudaFree(d_result_block3_squeeze1);
  cudaFree(d_result_block3_expand1);
  cudaFree(d_result_block3_squeeze2);

  cudaFree(d_fire5_squeeze_weight);
  cudaFree(d_fire5_squeeze_bias);
  cudaFree(d_fire5_expand1x1_weight);
  cudaFree(d_fire5_expand1x1_bias);
  cudaFree(d_fire5_expand3x3_weight);
  cudaFree(d_fire5_expand3x3_bias);

  cudaFree(d_fire6_squeeze_weight);
  cudaFree(d_fire6_squeeze_bias);
  cudaFree(d_fire6_expand1x1_weight);
  cudaFree(d_fire6_expand1x1_bias);
  cudaFree(d_fire6_expand3x3_weight);
  cudaFree(d_fire6_expand3x3_bias);

  cudaFree(d_fire7_squeeze_weight);
  cudaFree(d_fire7_squeeze_bias);
  cudaFree(d_fire7_expand1x1_weight);
  cudaFree(d_fire7_expand1x1_bias);
  cudaFree(d_fire7_expand3x3_weight);
  cudaFree(d_fire7_expand3x3_bias);

  cudaFree(d_fire8_squeeze_weight);
  cudaFree(d_fire8_squeeze_bias);
  cudaFree(d_fire8_expand1x1_weight);
  cudaFree(d_fire8_expand1x1_bias);
  cudaFree(d_fire8_expand3x3_weight);
  cudaFree(d_fire8_expand3x3_bias);

  cudaFree(d_result_block3_expand2);
  cudaFree(d_result_classifier_conv);

  cudaFree(d_classifier_conv_weight);
  cudaFree(d_classifier_conv_bias);
  cudaFree(d_result_classifier);

  free(h_result_classifier);

}
