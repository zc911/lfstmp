/*============================================================================
 * File Name   : face_feature_extractor.cpp
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年4月21日 下午1:31:28
 * Description : 
 * ==========================================================================*/
#include "face_feature_extractor.h"
#include "../caffe_helper.h"
#include "debug_util.h"
#include "log/log_val.h"

namespace dg {

FaceFeatureExtractor::FaceFeatureExtractor(
    const FaceFeatureExtractorConfig &config)
    : device_setted_(false),
      batch_size_(config.batch_size) {

    use_gpu_ = config.use_gpu;
    gpu_id_ = config.gpu_id;
    pixel_scale_ = 256;
    pixel_means_ = vector<float> {128, 128, 128};
    layer_name_ = "eltwise6";


    if (use_gpu_) {
        Caffe::SetDevice(config.gpu_id);
        Caffe::set_mode(Caffe::GPU);
        use_gpu_ = true;
    } else {
        Caffe::set_mode(Caffe::CPU);
        use_gpu_ = false;
    }

    ModelsMap *modelsMap = ModelsMap::GetInstance();

    string deploy_content;
    modelsMap->getModelContent(config.deploy_file, deploy_content);
    net_.reset(
        new Net<float>(config.deploy_file, deploy_content, TEST));
    string model_content;
    modelsMap->getModelContent(config.model_file, model_content);
    net_->CopyTrainedLayersFrom(config.model_file, model_content);


    Blob<float> *input_layer = net_->input_blobs()[0];
    ReshapeNetBatchSize(net_, batch_size_);

    num_channels_ = input_layer->channels();

    CHECK(num_channels_ == 1) << "Input layer should be gray scale.";

}


void FaceFeatureExtractor::miniBatchExtractor(const vector<Mat> &faces, vector<FaceRankFeature> &miniBatchResults) {

    Blob<float> *input_blob = net_->input_blobs()[0];

    float *input_data = input_blob->mutable_cpu_data();
    for (size_t i = 0; i < faces.size(); i++) {
        Mat sample;
        Mat face = faces[i];

        if (face.cols == 0 || face.rows == 0) {
            face = cv::Mat::zeros(1, 1, CV_8UC3);
        }

        if (face.cols != input_blob->width() || face.rows != input_blob->height())
            resize(face, face, Size(input_blob->width(), input_blob->height()));


        GenerateSample(num_channels_, face, sample);
        size_t image_off = i * sample.channels() * sample.rows * sample.cols;

        for (int k = 0; k < sample.channels(); k++) {
            size_t channel_off = k * sample.rows * sample.cols;
            for (int row = 0; row < sample.rows; row++) {
                size_t row_off = row * sample.cols;
                for (int col = 0; col < sample.cols; col++) {
                    input_data[image_off + channel_off + row_off + col] =
                        (float(sample.at<uchar>(row, col * sample.channels() + k)) - pixel_means_[k])
                            / pixel_scale_;
                }
            }
        }
    }

    net_->ForwardPrefilled();
    if (use_gpu_) {
        cudaDeviceSynchronize();
    }

    auto output_blob = net_->blob_by_name(layer_name_);
    const float *output_data = output_blob->cpu_data();
    const int feature_len = output_blob->channels();

    if (feature_len <= 0) {
        LOG(ERROR) << "Face feature len invalid: " << feature_len << endl;
        return;
    }

    miniBatchResults.clear();
    miniBatchResults.resize(faces.size());

    for (size_t i = 0; i < faces.size(); i++) {
        const float *data = output_data + i * feature_len;
        FaceRankFeature face_feature;
//        cout << "feature float: " << endl;
        for (int idx = 0; idx < feature_len; ++idx) {
            face_feature.descriptor_.push_back(data[idx]);
//            cout << data[idx] << " ";
        }

        miniBatchResults[i] = face_feature;

    }
}

std::vector<FaceRankFeature> FaceFeatureExtractor::Extract(
    const std::vector<Mat> &faces) {

    vector<FaceRankFeature> results;
    if (faces.size() == 0) {
        LOG(ERROR) << "Faces is empty" << endl;
        return results;
    }


    vector<FaceRankFeature> miniBatchResults;


    struct timeval start, finish;

    gettimeofday(&start, NULL);
    if (faces.size() <= batch_size_) {
        // BUG here when reshap the net
//        ReshapeNetBatchSize(net_, faces.size());
        miniBatchExtractor(faces, miniBatchResults);
        results.insert(results.end(), miniBatchResults.begin(), miniBatchResults.end());

    } else {
        vector<Mat> miniBatch;

        for (int i = 0; i < faces.size(); ++i) {
            miniBatch.push_back(faces[i]);
            if (miniBatch.size() == batch_size_) {

                // BUG here when reshap the net
//                ReshapeNetBatchSize(net_, miniBatch.size());
                miniBatchExtractor(miniBatch, miniBatchResults);
                results.insert(results.end(), miniBatchResults.begin(), miniBatchResults.end());
                miniBatch.clear();
                miniBatchResults.clear();
            }
        }
        if (miniBatch.size() > 0) {
            // BUG here when reshap the net
//            ReshapeNetBatchSize(net_, miniBatch.size());
            miniBatchExtractor(miniBatch, miniBatchResults);
            results.insert(results.end(), miniBatchResults.begin(), miniBatchResults.end());
        }
    }
    gettimeofday(&finish, NULL);
    VLOG(VLOG_PROCESS_COST) << "Faces feature extract costs: " << TimeCostInMs(start, finish) << endl;

    return results;

}

FaceFeatureExtractor::~FaceFeatureExtractor() {
}

} /* namespace dg */
