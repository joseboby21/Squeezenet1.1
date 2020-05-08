#include <stdio.h>
#include <stdlib.h>
#include <math.h>
# include <time.h>
#include "squeezenet_params.h"
//#include "hourglass.h"
//#include "elephant.h"
//#include "zebra.h"
//#include "dog.h"
//# include "tiger.h"
//#include "truck.h"
#include "lion.h"
clock_t T;
void getLabel(unsigned int);
char class_label[201];
void conv3x3(
  const int input_channels, const int input_size,
  const int pad, const int stride, const int start_channel,
  const int output_size,const int output_channels,const int input_im_size,
  const int filter_weight_size, const int filter_bias_size,const int output_im_size,
  const float* restrict input_im, const float* restrict filter_weight,
  const float* restrict filter_bias, float* restrict output_im){

    #pragma acc data copyin (input_im[0:input_im_size],filter_weight[0:filter_weight_size],filter_bias[0:filter_bias_size]) copyout(output_im[0:output_im_size])
    {

    #pragma acc parallel
    {
      #pragma acc loop gang
      for(int p=0;p<output_channels;++p){
        int filter_off = p * input_channels * 9;
        float bias = filter_bias[p];
        int offset;
        offset = (start_channel + p) * output_size * output_size;

        //loop over output feature map
        #pragma acc loop seq
        for(int i = 0; i < output_size; i++)
        {
          #pragma acc loop seq
          for(int j = 0; j < output_size; j++)
          {
            //compute one element in the output feature map
            float tmp = bias;

            //compute dot product of 2 input_channels x 3 x 3 matrix
            #pragma acc loop seq
            for(int k = 0; k < input_channels; k++)
            {
              #pragma acc loop seq
              for(int l = 0; l < 3; l++)
              {
                int h = i * stride + l - pad;
                #pragma acc loop seq
                for(int m = 0; m < 3; m++)
                {
                  int w = j * stride + m - pad;
                  if((h >= 0) && (h < input_size) && (w >= 0) && (w < input_size))
                  {
                    int fidx = filter_off + (9 * k + 3 * l + m);
                    tmp += input_im[k * input_size * input_size + (i * stride + l - pad) * input_size + j * stride + m - pad] \
                                     * filter_weight[fidx];
                  }
                }
              }
            }

            //add relu activation after conv
            int idx = offset + (i * output_size + j);
            output_im[idx] = (tmp > 0.0) ? tmp : 0.0;
          }
        }
      }
      //*time = T - clock();
      //*time = (((double)(*time))/CLOCKS_PER_SEC)*1000;
    }
    }
  }

  void conv1x1(
    const int input_channels, const int input_size,const int threads,
    const int output_channels, const int input_im_size,const int filter_weight_size,
    const int filter_bias_size, const int output_im_size,
    const float* restrict input_im, const float* restrict filter_weight,
    const float* restrict filter_bias, float* restrict output_im){
    #pragma acc data copyin(input_im[0:input_im_size],filter_weight[0:filter_weight_size],filter_bias[0:filter_bias_size]), copyout(output_im[0:output_im_size])
    {
      #pragma acc parallel
      {
        #pragma acc loop
        for(int c=0; c<output_channels;++c){
          //if(c<threads){
            int filter_off = c * input_channels;

            float bias = filter_bias[c];

            int output_off = c * input_size * input_size;//start_channel is for 1x1 feature map in fire layer

            //loop over output feature map
            //out
            #pragma acc loop seq
            for(int i = 0; i < input_size; i++)
            {
              #pragma acc loop seq
              for(int j = 0; j < input_size; j++)
              {
                float tmp = bias;
                #pragma acc loop seq
                for(int k = 0; k < input_channels; k++)
                {
                  int findx = filter_off + k;
                  tmp += input_im[k * input_size * input_size + i * input_size + j] * filter_weight[findx];
                }
                //add relu after conv
                int indx = output_off + (i * input_size + j);
                output_im[indx] = (tmp > 0.0) ? tmp : 0.0;
              }
            }
         //}
       }
     }
   }
  }

  void maxpool(
      const int input_size, const int output_size,
      const int output_channels, const int input_im_size, const int output_im_size,
      const float * restrict input_im, float * restrict output_im){
    #pragma acc data copyin(input_im[0:input_im_size]) copyout(output_im[0:output_im_size])
    {
      #pragma acc parallel
      {
        #pragma acc loop gang
        for(int c =0;c<output_channels;++c){
          int in_offset = c*input_size*input_size;
          int op_offset = c*output_size*output_size;


          #pragma acc loop seq
          for(int i=0; i<output_size; i++){
            #pragma acc loop seq
            for(int j=0; j<output_size; j++){

              float tmp = 0.0;
              #pragma acc loop seq
              for(int k =0; k<3; k++){
                #pragma acc loop seq
                for(int l = 0; l<3; l++){
                  int indx = in_offset + (i * 2 + k) * input_size + (j *2 +l);
                  float value = input_im[indx];
                  if(value > tmp)
                    tmp = value;
                }
              }
              int oindx = op_offset + i*output_size + j;
              output_im[oindx] = tmp;

            }
          }
        }
      }
    }
  }


      void main(){
        float timeT =0.0;
        clock_t t;
        t = clock();
        float * time = (float*)malloc(sizeof(float));
        float * conv_result = (float*)malloc(sizeof(float) * (1 * 64 * 111 * 111));
        conv3x3(3,224,0,2,0,111,64,150528,1728,64,788544,sample,conv1_weight,conv1_bias,conv_result);

        float * maxpool_result1 = (float*)malloc(1 * 64 * 55 * 55*sizeof(float));
        maxpool(111,55,64,788544,193600,conv_result,maxpool_result1);
        //Block1
        // fire1
        float * result_block1_squeeze = (float*)malloc(1 * 16* 55 * 55*sizeof(float));
        conv1x1(64,55,16,16,193600,1024,16,48400,maxpool_result1,fire1_squeeze_weight,fire1_squeeze_bias,result_block1_squeeze);
        float * result_block1_expand = (float*)malloc(387200*sizeof(float));
        conv1x1(16,55,64,64,48400,1024,64,387200,result_block1_squeeze,fire1_expand1x1_weight,fire1_expand1x1_bias,result_block1_expand);
        conv3x3(16,55,1,1,64,55,64,48400,9216,64,387200,result_block1_squeeze,fire1_expand3x3_weight,fire1_expand3x3_bias,result_block1_expand);



        //fire2

        conv1x1(128,55,16,16,387200,2048,16,48400,result_block1_expand,fire2_squeeze_weight,fire2_squeeze_bias,result_block1_squeeze);
        conv1x1(16,55,64,64,48400,1024,64,387200,result_block1_squeeze,fire2_expand1x1_weight,fire2_expand1x1_bias,result_block1_expand);
        conv3x3(16,55,1,1,64,55,64,48400,9216,64,387200,result_block1_squeeze,fire2_expand3x3_weight,fire2_expand3x3_bias,result_block1_expand);


        float * result_pool2 = (float*)malloc(93312*sizeof(float));
        maxpool(55,27,128,387200,93312,result_block1_expand,result_pool2);
        //Block2
        //fire3

        float * result_block2_squeeze = (float*)malloc(23328*sizeof(float));
        conv1x1(128,27,32,32,93312,4096,32,23328,result_pool2,fire3_squeeze_weight,fire3_squeeze_bias,result_block2_squeeze);
        float * result_block2_expand = (float*)malloc(186624*sizeof(float));
        conv1x1(32,27,128,128,23328,4096,128,186624,result_block2_squeeze,fire3_expand1x1_weight,fire3_expand1x1_bias,result_block2_expand);
        conv3x3(32,27,1,1,128,27,128,23328,36864,128,186624,result_block2_squeeze,fire3_expand3x3_weight,fire3_expand3x3_bias,result_block2_expand);



        //fire4
        conv1x1(256,27,32,32,186624,8192,32,23328,result_block2_expand,fire4_squeeze_weight,fire4_squeeze_bias,result_block2_squeeze);
        conv1x1(32,27,128,128,23328,4096,128,186624,result_block2_squeeze,fire4_expand1x1_weight,fire4_expand1x1_bias,result_block2_expand);
        conv3x3(32,27,1,1,128,27,128,23328,36864,128,186624,result_block2_squeeze,fire4_expand3x3_weight,fire4_expand3x3_bias,result_block2_expand);




        float * result_pool3 = (float*)malloc(43264*sizeof(float));
        maxpool(27,13,256,186624,43264,result_block2_expand,result_pool3);

        //Block3
        //fire5
        float * result_block3_squeeze1 = (float*)malloc(8112*sizeof(float));
        conv1x1(256,13,48,48,43264,12288,48,8112,result_pool3,fire5_squeeze_weight,fire5_squeeze_bias,result_block3_squeeze1);
        float * result_block3_expand1 = (float*)malloc(64896*sizeof(float));
        conv1x1(48,13,192,192,8112,9216,192,64896,result_block3_squeeze1,fire5_expand1x1_weight,fire5_expand1x1_bias,result_block3_expand1);
        conv3x3(48,13,1,1,192,13,192,8112,82944,192,64896,result_block3_squeeze1,fire5_expand3x3_weight,fire5_expand3x3_bias,result_block3_expand1);


        //fire6
        conv1x1(384,13,48,48,64896,18432,48,8112,result_block3_expand1,fire6_squeeze_weight,fire6_squeeze_bias,result_block3_squeeze1);
        conv1x1(48,13,192,192,8112,9216,192,64896,result_block3_squeeze1,fire6_expand1x1_weight,fire6_expand1x1_bias,result_block3_expand1);
        conv3x3(48,13,1,1,192,13,192,8112,82944,192,64896,result_block3_squeeze1,fire6_expand3x3_weight,fire6_expand3x3_bias,result_block3_expand1);



        //fire7
        float * result_block3_squeeze2 = (float*)malloc(10816*sizeof(float));
        conv1x1(384,13,64,64,64896,24576,64,10816,result_block3_expand1,fire7_squeeze_weight,fire7_squeeze_bias,result_block3_squeeze2);
        float * result_block3_expand2 = (float*)malloc(86528*sizeof(float));
        conv1x1(64,13,256,256,10816,16384,256,86528,result_block3_squeeze2,fire7_expand1x1_weight,fire7_expand1x1_bias,result_block3_expand2);
        conv3x3(64,13,1,1,256,13,256,10816,147456,256,86528,result_block3_squeeze2,fire7_expand3x3_weight,fire7_expand3x3_bias,result_block3_expand2);



        conv1x1(512,13,64,64,86528,32768,64,10816,result_block3_expand2,fire8_squeeze_weight,fire8_squeeze_bias,result_block3_squeeze2);
        conv1x1(64,13,256,256,10816,16384,256,86528,result_block3_squeeze2,fire8_expand1x1_weight,fire8_expand1x1_bias,result_block3_expand2);
        conv3x3(64,13,1,1,256,13,256,10816,147456,256,86528,result_block3_squeeze2,fire8_expand3x3_weight,fire8_expand3x3_bias,result_block3_expand2);

        float * result_classifier_conv = (float*)malloc(169000*sizeof(float));
        conv1x1(512,13,1000,1000,86528,512000,1000,169000,result_block3_expand2,classifier_conv_weight,classifier_conv_bias,result_classifier_conv);
        float * result_classifier = (float*)malloc(1000*sizeof(float));
        for(int i=0;i<1000;++i){
          float tmp =0;
          for(int j=0;j<169;++j){
            tmp = tmp + result_classifier_conv[j];
          }
          result_classifier[i] = tmp/169;
          result_classifier_conv = result_classifier_conv + 169;
        }

        float tmp = 0.0f;
        unsigned int class_index = 0;
        for(int j = 0; j < 1000; j++)
        {
          if(result_classifier[j] > tmp)
          {
        	  tmp = result_classifier[j];
            class_index = j;
          }
        }
        timeT = clock() - t;
        timeT = (((double)timeT)/CLOCKS_PER_SEC)*1000;

        getLabel(class_index);
        printf("\r\npredicted label: %s\r\n", class_label);
        printf("Execution Time: %f\n",timeT);
        printf("%f\n",*time);
        printf("done\n");

        free(conv_result);
        free(maxpool_result1);
        free(result_block1_squeeze);
        free(result_block1_expand);
        free(result_pool2);
        free(result_block2_squeeze);
        free(result_block2_expand);
        free(result_pool3);
        free(result_block3_squeeze1);
        free(result_block3_expand1);
        free(result_block3_squeeze2);
        free(result_block3_expand2);


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
