/*============================================================================
 * File Name   : face_feature_extractor.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年4月21日 下午1:31:27
 * Description : 
 * ==========================================================================*/
#ifndef FACE_FEATURE_EXTRACTOR_H_
#define FACE_FEATURE_EXTRACTOR_H_

#include <string>

#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>

#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/render_face_detections.h>

#include "model/basic.h"
#include "model/rank_feature.h"

using namespace std;
using namespace cv;
using namespace caffe;

namespace dg
{

struct InnFaceFeature {
	float data[256];
};

class FaceFeatureExtractor
{
public:
	FaceFeatureExtractor(const string& model_file, const string& trained_file,
			const bool use_gpu, const int batch_size, const string &align_model,
			const string &avg_face);

	virtual ~FaceFeatureExtractor();
	std::vector<FaceRankFeature> Extract(const std::vector<Mat> &imgs);
	std::vector<Mat> Align(std::vector<Mat> imgs);

private:
	void Detection2Points(const dlib::full_object_detection &detection,
			std::vector<dlib::point> &points);

private:
	std::shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	int batch_size_;
	string layer_name_;
	bool useGPU_;

	dlib::shape_predictor sp_;
	std::vector<dlib::point> avg_face_points_;
	dlib::frontal_face_detector detector_;
};

} /* namespace dg */

#endif /* FACE_FEATURE_EXTRACTOR_H_ */
