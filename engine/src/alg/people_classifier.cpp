/*============================================================================
 * File Name   : people_classifier.cpp
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年6月30日 上午10:08:13
 * Description : 
 * ==========================================================================*/
#include "people_classifier.h"

namespace dg
{

PeopleClassifier::PeopleClassifier(PeopleConfig &peopleconfig) :
		height_(360), width_(205), crop_height_(350), crop_width_(180), pixel_scale_(
				256), pixel_means_
		{ 104, 117, 123 }
{
	/* Load the network. */
	net_.reset(new Net<float>(peopleconfig.deploy_file, TEST, peopleconfig.is_model_encrypt));
	net_->CopyTrainedLayersFrom(peopleconfig.model_file);
	layer_name_ = peopleconfig.layer_name;

	Blob<float>* input_blob = net_->input_blobs()[0];
	batch_size_ = input_blob->num();
	num_channels_ = input_blob->channels();
	CHECK(num_channels_ == 3) << "Input layer should have 3 channels.";
	input_blob->Reshape(batch_size_, num_channels_, crop_height_, crop_width_);

	if (peopleconfig.use_gpu)
	{
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(peopleconfig.gpu_id);
		use_gpu_ = true;
	}
	else
	{
		Caffe::set_mode(Caffe::CPU);
		use_gpu_ = false;
	}

	// load attrib names
	LoadTagnames(peopleconfig.tag_name_path);

	//calculate crop rectangle coordinates
	int offset_h = height_ - crop_height_;
	int offset_w = width_ - crop_width_;
	offset_h = offset_h / 2;
	offset_w = offset_w / 2;
	crop_rect_ = Rect(offset_w, offset_h, crop_width_, crop_height_);
}

PeopleClassifier::~PeopleClassifier()
{
}

void PeopleClassifier::LoadTagnames(const string &name_list)
{
	ifstream fp(name_list);
	tagnames_.resize(0);
	while (!fp.eof())
	{
		string tagname = "";
		fp >> tagname;
		if (tagname == "")
			continue;
		tagnames_.push_back(tagname);
	}
}

void PeopleClassifier::AttributePredict(const vector<Mat> &imgs,
		vector<vector<float> > &results)
{
	Blob<float>* input_blob = net_->input_blobs()[0];
	int num_imgs = static_cast<int>(imgs.size());
	assert(num_imgs <= batch_size_);
	vector<int> shape =
	{ num_imgs, 3, crop_height_, crop_width_ };
	input_blob->Reshape(shape);
	net_->Reshape();
	float* input_data = input_blob->mutable_cpu_data();
	int cnt = 0;

	for (size_t i = 0; i < imgs.size(); i++)
	{
		Mat sample, img = imgs[i];
		if (img.channels() == 4 && num_channels_ == 3)
			cvtColor(img, sample, CV_BGRA2BGR);
		else if (img.channels() == 1 && num_channels_ == 3)
			cvtColor(img, sample, CV_GRAY2BGR);
		else
			sample = img;

		if ((sample.rows != height_) || (sample.cols != width_))
		{
			resize(sample, sample, Size(width_, height_));
			sample(crop_rect_).copyTo(sample);
		}

		for (int k = 0; k < sample.channels(); k++)
		{
			for (int row = 0; row < sample.rows; row++)
			{
				for (int col = 0; col < sample.cols; col++)
				{
					input_data[cnt] = (float(sample.at<uchar>(row, col * 3 + k))
							- pixel_means_[k]) / pixel_scale_;
					cnt++;
				}
			}
		}
	}

	net_->ForwardPrefilled();
	if (use_gpu_)
	{
		cudaDeviceSynchronize();
	}

	auto output_blob = net_->blob_by_name(layer_name_);
	const float *output_data = output_blob->cpu_data();
	const int feature_len = output_blob->channels();
	assert(feature_len == static_cast<int>(tagnames_.size()));

	results.resize(imgs.size());
	for (size_t i = 0; i < imgs.size(); i++)
	{
		const float *data = output_data + i * feature_len;
		vector<float> &feature = results[i];
		feature.resize(feature_len);
		for (int idx = 0; idx < feature_len; ++idx)
		{
			feature[idx] = data[idx];
		}
	}
}

std::vector<PeopleAttr> PeopleClassifier::BatchClassify(
		const vector<string> &img_filenames)
{
	vector<Mat> images;
	std::vector<PeopleAttr> attrs;
	vector<vector<float> > results;
	for (size_t i = 0; i < img_filenames.size(); i++)
	{
		Mat image;
		image = imread(img_filenames[i], -1);
		if (image.empty())
		{
			cout << "Wrong Image: " << img_filenames[i] << endl;
			continue;
		}
		images.push_back(image);
	}

	AttributePredict(images, results);
	for (size_t idx = 0; idx < results.size(); idx++)
	{
		PeopleAttr attr;
		for (size_t a_idx = 0; a_idx < 46; a_idx++)
		{
			attr.tagname = tagnames_[a_idx];
			attr.confidence = results[idx][a_idx];
			attrs.push_back(attr);
		}
	}
	return attrs;
}

} /* namespace dg */
