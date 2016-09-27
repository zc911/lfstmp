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
//    do {
//        std::vector<int> shape = input_layer->shape();
//        shape[0] = batch_size_;
//        input_layer->Reshape(shape);
//        net_->Reshape();
//    } while (0);

    num_channels_ = input_layer->channels();

    CHECK(num_channels_ == 1) << "Input layer should be gray scale.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

}


void FaceFeatureExtractor::miniBatchExtractor(vector<Mat> &alignImgs, vector<FaceRankFeature> &miniBatchResults) {

    Blob<float> *input_blob = net_->input_blobs()[0];

    float *input_data = input_blob->mutable_cpu_data();
    for (size_t i = 0; i < alignImgs.size(); i++) {
        Mat sample;
        Mat face = alignImgs[i];

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
    miniBatchResults.resize(alignImgs.size());

    for (size_t i = 0; i < alignImgs.size(); i++) {
        const float *data = output_data + i * feature_len;
        FaceRankFeature face_feature;
        for (int idx = 0; idx < feature_len; ++idx) {
            face_feature.descriptor_.push_back(data[idx]);
        }

        miniBatchResults[i] = face_feature;

    }
}

std::vector<FaceRankFeature> FaceFeatureExtractor::Extract(
    std::vector<Mat> &align_imgs) {

    vector<FaceRankFeature> results;
    if (align_imgs.size() == 0) {
        LOG(ERROR) << "Faces is empty" << endl;
        return results;
    }
    vector<FaceRankFeature> miniBatchResults;
    if (align_imgs.size() <= batch_size_) {

        ReshapeNetBatchSize(net_, align_imgs.size());
        miniBatchExtractor(align_imgs, miniBatchResults);
        results.insert(results.end(), miniBatchResults.begin(), miniBatchResults.end());
    } else {
        vector<Mat> miniBatch;
        for (int i = 0; i < align_imgs.size(); ++i) {
            miniBatch.push_back(align_imgs[i]);
            if (miniBatch.size() == batch_size_) {
                ReshapeNetBatchSize(net_, miniBatch.size());
                miniBatchExtractor(miniBatch, miniBatchResults);
                results.insert(results.end(), miniBatchResults.begin(), miniBatchResults.end());
                miniBatch.clear();
                miniBatchResults.clear();
            }
        }
        if (miniBatch.size() > 0) {
            ReshapeNetBatchSize(net_, miniBatch.size());
            miniBatchExtractor(miniBatch, miniBatchResults);
            results.insert(results.end(), miniBatchResults.begin(), miniBatchResults.end());
        }
    }

    return results;



//    for (auto miniBatch : PrepareBatch(faces, batch_size_)) {
//        cout << "mini batch size: " << miniBatch.size() << endl;
//        std::vector<FaceRankFeature> miniBatchResults;



//    }

    return results;

//    if (!device_setted_) {
//        Caffe::SetDevice(gpu_id_);
//        Caffe::set_mode(Caffe::GPU);
//        device_setted_ = true;
//    }
//    std::vector<Mat> align_imgs = Align(imgs);
//    std::vector<FaceRankFeature> features;
//    Blob<float> *input_blob = net_->input_blobs()[0];
//    assert(align_imgs.size() <= batch_size_);
//    features.resize(align_imgs.size());
//    float *input_data = input_blob->mutable_cpu_data();
//    int cnt = 0;
//    for (size_t i = 0; i < align_imgs.size(); i++) {
//        Mat sample;
//        Mat img = align_imgs[i];
//
//        if (img.channels() == 3 && num_channels_ == 1)
//            cvtColor(img, sample, CV_BGR2GRAY);
//        else if (img.channels() == 4 && num_channels_ == 1)
//            cvtColor(img, sample, CV_BGRA2GRAY);
//        else
//            sample = img;
//
//        assert(sample.channels() == 1);
//        assert((sample.rows == input_geometry_.height)
//                   && (sample.cols == input_geometry_.width));
//        for (int i = 0; i < sample.rows; i++) {
//            for (int j = 0; j < sample.cols; j++) {
//                input_data[cnt] = sample.at<uchar>(i, j) / 255.0f;
//                cnt += 1;
//            }
//        }
//    }
//
//    net_->ForwardPrefilled();
//    if (use_gpu_) {
//        cudaDeviceSynchronize();
//    }
//
//    auto output_blob = net_->blob_by_name(layer_name_);
//    const float *output_data = output_blob->cpu_data();
//    for (size_t i = 0; i < align_imgs.size(); i++) {
//        InnFaceFeature feature;
//        const float *data = output_data
//            + i * sizeof(InnFaceFeature) / sizeof(float);
//        memcpy(&feature, data, sizeof(InnFaceFeature));
//
//        FaceRankFeature face_feature;
//        for (int j = 0; j < 256; ++j) {
//            face_feature.descriptor_.push_back(feature.data[j]);
//        }
//        features[i] = face_feature;
//    }
//    return features;
}

FaceFeatureExtractor::~FaceFeatureExtractor() {
}

} /* namespace dg */
