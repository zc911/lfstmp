/*============================================================================
 * File Name   : pedestrian_classifier.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年6月30日 上午10:08:13
 * Description : 
 * ==========================================================================*/
#ifndef PEDESTRIAN_CLASSIFIER_H_
#define PEDESTRIAN_CLASSIFIER_H_

#include <algorithm>
#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "dgcaffe/caffe/caffe.hpp"

using namespace std;
using namespace cv;
using namespace caffe;

namespace dg {

class PedestrianClassifier {
public:
    typedef struct {
        bool is_model_encrypt = true;
        int gpu_id = 0;
        bool use_gpu = true;
        string tag_name_path;
        string deploy_file;
        string model_file;
        string layer_name;
    } PedestrianConfig;

    typedef struct {
        int index;
        string tagname;
        float threshold_lower;
        float threshold_upper;
        int categoryId;
        int mappingId;
    } Tag;

    typedef struct {
        int index;
        string tagname;
        float confidence;
        float threshold_lower;
        float threshold_upper;
        int categoryId;
        int mappingId;
    } PedestrianAttribute;
    PedestrianClassifier(PedestrianConfig &pconf);
    virtual ~PedestrianClassifier();
    std::vector<vector<PedestrianAttribute>> BatchClassify(
        const vector<cv::Mat> &imgs);

public:
    vector<Tag> tagtable_;
    int batch_size_;

private:
    void LoadTagnames(const string &name_list);
    void AttributePredict(const vector<Mat> &imgs,
                          vector<vector<float> > &results);

private:
    caffe::shared_ptr<Net<float> > net_;
    Rect crop_rect_;
    bool use_gpu_;
    string layer_name_;
    int num_channels_;
    int height_;
    int width_;
    int crop_height_;
    int crop_width_;
    int pixel_scale_;
    float pixel_means_[3];
};

} /* namespace dg */

#endif /* PEDESTRIAN_CLASSIFIER_H_ */
