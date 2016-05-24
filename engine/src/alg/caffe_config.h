/*
 * caffe_config.h
 *
 *  Created on: 08/04/2016
 *      Author: chenzhen
 */

#ifndef CAFFE_CONFIG_H_
#define CAFFE_CONFIG_H_

#include <string>

using namespace std;

typedef struct {

    bool is_model_encrypt = false;
    int batch_size = 1;
    int class_num = 0;
    int target_min_size = 400;
    int target_max_size = 1000;
    int rescale = 1;
    int gpu_id = 0;
    bool use_gpu = true;
    float means[3] = {0, 0, 0};
    float threshold = 0.0;
    string deploy_file;
    string model_file;

} CaffeConfig;

#endif /* CAFFE_CONFIG_H_ */
