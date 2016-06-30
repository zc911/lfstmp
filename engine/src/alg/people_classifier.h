/*============================================================================
 * File Name   : people_classifier.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年6月30日 上午10:08:13
 * Description : 
 * ==========================================================================*/
#ifndef PEOPLE_CLASSIFIER_H_
#define PEOPLE_CLASSIFIER_H_

#include <algorithm>
#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/caffe.hpp>

using namespace std;
using namespace cv;
using namespace caffe;

namespace dg
{

struct PeopleAttr {
    string tagname;
    float confidence;
};

class PeopleClassifier
{
public:
	typedef struct {
	        bool is_model_encrypt = false;
	        int gpu_id = 0;
	        bool use_gpu = true;
	        string tag_name_path;
	        string deploy_file;
	        string model_file;
	        string layer_name;
	    } PeopleConfig;
	PeopleClassifier(PeopleConfig &peopleconfig);
	virtual ~PeopleClassifier();
	std::vector<PeopleAttr> BatchClassify(const vector<string> &img_filenames);

public:
    vector<string> tagnames_;
    int batch_size_;

private:
    void LoadTagnames(const string &name_list);
	void AttributePredict(const vector<Mat> &imgs, vector<vector<float> > &results);

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

#endif /* PEOPLE_CLASSIFIER_H_ */
