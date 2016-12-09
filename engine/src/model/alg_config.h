#ifndef ALG_CONFIG_H_
#define ALG_CONFIG_H_

#include <string>

using namespace std;

namespace dg {

typedef struct {
    bool is_model_encrypt = true;
    int batch_size = 1;
    int target_min_size = 400;
    int target_max_size = 1000;
    int gpu_id = 0;
    bool use_gpu = true;
    bool is_driver = true;
    float threshold = 0.0;
    string deploy_file;
    string model_file;
} VehicleBeltConfig;

typedef struct {
    bool car_only = false;
    bool is_model_encrypt = true;
    int batch_size = 1;
    float target_min_size=0.001;
    float target_max_size=0.001;
    int gpu_id = 0;
    bool use_gpu = true;
    string deploy_file;
    string model_file;
    string confirm_deploy_file;
    string confirm_model_file;
    float threshold = 0.5;
} VehicleCaffeDetectorConfig;

}


#endif