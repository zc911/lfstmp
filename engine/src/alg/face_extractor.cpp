/*============================================================================
 * File Name   : face_extractor.cpp
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/19/2016
 * Description : 
 * ==========================================================================*/

#include "face_extractor.h"

using namespace dg;

float get_cos_similarity(const vector<float> & A, const vector<float> & B) {
    float dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (unsigned int i = 0; i<A.size(); ++i) {
        dot += A[i] * B[i];
        denom_a += A[i] * A[i];
        denom_b += B[i] * B[i];
    }
    return abs(dot) / (sqrt(denom_a) * sqrt(denom_b));
}

FaceExtractor::FaceExtractor(string deploy, string model, int batchSize, int gpuId)
{
#if USE_CUDA
        Caffe::SetDevice(gpuId);
        Caffe::set_mode(Caffe::GPU);
        LOG(INFO)<< "Use device " << gpuId << endl;
#else
        Caffe::set_mode(Caffe::CPU);
        LOG(WARNING) << "Use CPU only" << endl;
#endif
    
    batch_size_ = batchSize;

    /* Load the network. */
    net_.reset(new Net<float>(deploy, TEST));
    net_->CopyTrainedLayersFrom(model);
    CHECK_EQ(net_->num_inputs(), 1)<< "Network should have exactly one input.";
    Blob<float>* input_layer = net_->input_blobs()[0];

    vector<int> shape = input_layer->shape();
    shape[0] = batch_size_;
    input_layer->Reshape(shape);
    net_->Reshape();

    num_channels_ = input_layer->channels();
    cout<<num_channels_<<endl;
    CHECK(num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

FaceExtractor::~FaceExtractor()
{

}

void FaceExtractor::Classify(const vector<Mat>& images, const vector<FaceFeature>& features, vector<vector<Score>>& predicts)
{
    Blob<float>* input_blob = net_->input_blobs()[0];
    float* input_data = input_blob->mutable_cpu_data();

    int index = 0;
    for(const Mat& image : images)
    {
        Mat sample(image);
        if(num_channels_ == 1)
        {
            switch(image.channels())
            {
                case 3:
                    cvtColor(image, sample, CV_BGR2GRAY);
                    break;
                case 4:
                    cvtColor(image, sample, CV_BGRA2GRAY);
                    break;
            }
        }

        assert(sample.channels() == 1);
        assert(sample.rows == input_geometry_.height);
        assert(sample.cols == input_geometry_.width);
        for (int i = 0; i < sample.rows; i++) {
            for (int j = 0; j < sample.cols; j++) {
                input_data[index++] = sample.at<uchar>(i, j) / 255.0f;
            }
        }
    }

    net_->ForwardPrefilled();
#if USE_CUDA
    cudaDeviceSynchronize();
#endif

    auto output_blob = net_->blob_by_name( "eltwise6");
    const float *begin = output_blob->cpu_data();

    vector<float> output_batch = std::vector<float>(begin, begin + 256 * images.size());
    for (size_t i = 0; i < images.size(); i++) {
        std::vector<float> output(output_batch.begin() + i * 256,
                                  output_batch.begin() + i * 256 + 256);

        vector<Score> pred;
        for (int i = 0; i < features.size(); i ++) {
            Score p(i, get_cos_similarity(output, features[i].descriptor_));
            pred.push_back(p);
        }
        predicts.push_back(pred);

    }
}