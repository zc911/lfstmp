/*
 * detector.h
 *
 *  Created on: Aug 12, 2015
 *      Author: chenzhen
 */

#ifndef DETECTOR_H_
#define DETECTOR_H_

#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>

#include "model/basic.h"
#include "model/model.h"

using namespace std;

namespace dg {

class Detector {
public:
    Detector() {
    }
    virtual ~Detector() {
    }
};

class VehicleDetector {
public:
    typedef struct {
        bool car_only = false;
        bool is_model_encrypt = false;
        int batch_size = 1;
        int target_min_size = 600;
        int target_max_size = 1000;
        int gpu_id = 0;
        bool use_gpu = true;
        string deploy_file;
        string model_file;
        float threshold = 0.5;
    } VehicleCaffeDetectorConfig;

    virtual int DetectBatch(vector<cv::Mat> &img,
                            vector<vector<Detection> > &detect_results) = 0;
};
}
#endif /* DETECTOR_H_ */
