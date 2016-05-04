/*============================================================================
 * File Name   : face_feature_extractor.cpp
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年4月21日 下午1:31:28
 * Description : 
 * ==========================================================================*/
#include "alg/face_feature_extractor.h"

namespace dg
{

FaceFeatureExtractor::FaceFeatureExtractor(const string& model_file,
		const string& trained_file, const bool use_gpu, const int batch_size,
		const string &align_model, const string &avg_face) :
		batch_size_(batch_size), detector_(dlib::get_frontal_face_detector())
{
	if (use_gpu)
	{
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(0);
		useGPU_ = true;
	}
	else
	{
		Caffe::set_mode(Caffe::CPU);
		useGPU_ = false;
	}

	layer_name_ = "eltwise6";

	LOG(INFO)<< "loading model file: " << model_file;
	net_.reset(new Net<float>(model_file, TEST));
	LOG(INFO)<< "loading trained file : " << trained_file;
	net_->CopyTrainedLayersFrom(trained_file);

	Blob<float>* input_layer = net_->input_blobs()[0];
	do
	{
		std::vector<int> shape = input_layer->shape();
		shape[0] = batch_size_;
		input_layer->Reshape(shape);
		net_->Reshape();
	} while (0);
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 1) << "Input layer should be gray scale.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	dlib::deserialize(align_model) >> sp_;
	cv::Mat avg_face_img = cv::imread(avg_face);
	dlib::cv_image<dlib::bgr_pixel> avg_face_image(avg_face_img);

	std::vector<dlib::rectangle> avg_face_bbox = detector_(avg_face_image);
	assert(avg_face_bbox.size() == 1);
	dlib::full_object_detection shape = sp_(avg_face_image, avg_face_bbox[0]);
	Detection2Points(shape, avg_face_points_);

}

void FaceFeatureExtractor::Detection2Points(
		const dlib::full_object_detection &detection,
		std::vector<dlib::point> &points)
{
	points.resize(0);
	for (unsigned long i = 0; i < detection.num_parts(); i++)
	{
		points.push_back(detection.part(i));
	}
}

std::vector<Mat> FaceFeatureExtractor::Align(std::vector<Mat> imgs)
{
	std::vector<Mat> result;
	for (int i = 0; i < imgs.size(); i++)
	{
		std::vector<dlib::point> points;
		dlib::full_object_detection shape;
		dlib::point_transform_affine trans;
		dlib::cv_image<dlib::bgr_pixel> image(imgs[i]);

		dlib::rectangle bbox(0, 0, imgs[i].cols, imgs[i].rows);
		shape = sp_(image, bbox);
		if (shape.num_parts() != avg_face_points_.size())
		{
			cv::Mat face = Mat::zeros(128, 128, CV_8UC3);
			continue;

		}
		Detection2Points(shape, points);
		trans = find_affine_transform(avg_face_points_, points);

		dlib::array2d<dlib::bgr_pixel> out(128, 128);
		dlib::transform_image(image, out, dlib::interpolate_bilinear(), trans);
		cv::Mat face = toMat(out).clone();
		result.push_back(face);

	}

	return result;
}

std::vector<FaceRankFeature> FaceFeatureExtractor::Extract(
		const std::vector<Mat> &imgs)
{
	std::vector<Mat> align_imgs = Align(imgs);
	std::vector<FaceRankFeature> features;
	Blob<float>* input_blob = net_->input_blobs()[0];
	assert(align_imgs.size() <= batch_size_);
	features.resize(align_imgs.size());
	float* input_data = input_blob->mutable_cpu_data();
	int cnt = 0;
	for (size_t i = 0; i < align_imgs.size(); i++)
	{
		Mat sample;
		Mat img = align_imgs[i];

		if (img.channels() == 3 && num_channels_ == 1)
			cvtColor(img, sample, CV_BGR2GRAY);
		else if (img.channels() == 4 && num_channels_ == 1)
			cvtColor(img, sample, CV_BGRA2GRAY);
		else
			sample = img;

		assert(sample.channels() == 1);
		assert(
				(sample.rows == input_geometry_.height)
						&& (sample.cols == input_geometry_.width));
		for (int i = 0; i < sample.rows; i++)
		{
			for (int j = 0; j < sample.cols; j++)
			{
				input_data[cnt] = sample.at<uchar>(i, j) / 255.0f;
				cnt += 1;
			}
		}
	}

	net_->ForwardPrefilled();
	if (useGPU_)
	{
		cudaDeviceSynchronize();
	}

	auto output_blob = net_->blob_by_name(layer_name_);
	const float *output_data = output_blob->cpu_data();
	for (size_t i = 0; i < align_imgs.size(); i++)
	{
		InnFaceFeature feature;
		const float *data = output_data
				+ i * sizeof(InnFaceFeature) / sizeof(float);
		memcpy(&feature, data, sizeof(InnFaceFeature));

		FaceRankFeature face_feature;
		for (int j = 0; j < 256; ++j)
		{
			face_feature.descriptor_.push_back(feature.data[j]);
		}
		features[i] = face_feature;
	}
	return features;
}

FaceFeatureExtractor::~FaceFeatureExtractor()
{
}

} /* namespace dg */
