//
// Created by jiajaichen on 16-6-20.
//

#ifndef PROJECT_ACCELERATE_H
#define PROJECT_ACCELERATE_H


#include <cassert>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "dgcaffe/caffe/caffe.hpp"
#include "model/basic.h"
#include "detector.h"
#include "caffe_config.h"

using namespace std;
using namespace cv;
using namespace caffe;

namespace dg {

class Accelerate{

public:


    Accelerate ();
    virtual ~Accelerate();

protected:
    boost::shared_ptr<caffe::Net<float>> net_;
    int num_channels_;
    cv::Size input_geometry_;
    bool device_setted_;
#ifdef SHOW_VIS
    vector<Scalar> color_;
    vector<string> tags_;
#endif

};
}
#endif //PROJECT_ACCELERATE_H
