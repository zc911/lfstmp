/*
 * caffe_detector.h

 *
 *  Created on: Apr 18, 2016
 *      Author: jiajaichen
 */

#ifndef ENGINE_SRC_ALG_CAR_ONLY_DETECTOR_H_
#define ENGINE_SRC_ALG_CAR_ONLY_DETECTOR_H_

#include <cassert>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <caffe/caffe.hpp>
#include "model/basic.h"
#include "detector.h"
#include "caffe_config.h"

using namespace std;
using namespace caffe;
using namespace cv;

namespace dg {

class CarOnlyCaffeDetector: public VehicleDetector {

public:

    CarOnlyCaffeDetector(const VehicleCaffeDetectorConfig &config);

    virtual ~CarOnlyCaffeDetector();

    virtual int DetectBatch(const vector<cv::Mat> &img,
                            vector<vector<Detection> > &detect_results);
protected:
    vector<Blob<float> *> PredictBatch(vector<Mat> imgs);
    void WrapBatchInputLayer(vector<vector<Mat>> *input_batch);

    void PreprocessBatch(const vector<Mat> imgs,
                         vector<vector<Mat>> *input_batch);
private:
    boost::shared_ptr<caffe::Net<float>> net_;
    int num_channels_;
    cv::Size input_geometry_;
    bool device_setted_;
    VehicleCaffeDetectorConfig caffe_config_;
    int scale_num_;
    Mat means_;
    int rescale_;

};

}
#endif /* ENGINE_SRC_ALG_CAR_ONLY_DETECTOR_H_ */
