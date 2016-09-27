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
      batch_size_(config.batch_size),
      detector_(dlib::get_frontal_face_detector()) {

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
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    dlib::deserialize(config.align_model) >> sp_;
    cv::Mat avg_face_img = cv::imread(config.align_deploy);
    dlib::cv_image<dlib::bgr_pixel> avg_face_image(avg_face_img);

    std::vector<dlib::rectangle> avg_face_bbox = detector_(avg_face_image);
    assert(avg_face_bbox.size() == 1);
    dlib::full_object_detection shape = sp_(avg_face_image, avg_face_bbox[0]);
    Detection2Points(shape, avg_face_points_);

}

void FaceFeatureExtractor::Detection2Points(
    const dlib::full_object_detection &detection,
    std::vector<dlib::point> &points) {
    points.resize(0);
    for (unsigned long i = 0; i < detection.num_parts(); i++) {
        points.push_back(detection.part(i));
    }
}

std::vector<Mat> FaceFeatureExtractor::Align(std::vector<Mat> imgs) {
    std::vector<Mat> result;
    for (int i = 0; i < imgs.size(); i++) {
        std::vector<dlib::point> points;
        dlib::full_object_detection shape;
        dlib::point_transform_affine trans;
        dlib::cv_image<dlib::bgr_pixel> image(imgs[i]);

        dlib::rectangle bbox(0, 0, imgs[i].cols, imgs[i].rows);
        shape = sp_(image, bbox);
        if (shape.num_parts() != avg_face_points_.size()) {
            cv::Mat face = cv::Mat::zeros(128, 128, CV_8UC3);
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
    const std::vector<Mat> &faces) {

    vector<FaceRankFeature> results;
    if (faces.size() == 0) {
        LOG(ERROR) << "Faces is empty" << endl;
        return results;
    }


    vector<FaceRankFeature> miniBatchResults;

    std::vector<Mat> align_imgs;

    struct timeval start, finish;
    gettimeofday(&start, NULL);

    align_imgs = Align(faces);
    gettimeofday(&finish, NULL);

    VLOG(VLOG_PROCESS_COST) << "Faces align cost: " << TimeCostInMs(start, finish) << endl;

    gettimeofday(&start, NULL);
    if (faces.size() <= batch_size_) {
        // BUG
        ReshapeNetBatchSize(net_, align_imgs.size());
        miniBatchExtractor(align_imgs, miniBatchResults);
        results.insert(results.end(), miniBatchResults.begin(), miniBatchResults.end());

    } else {
        vector<Mat> miniBatch;

        for (int i = 0; i < align_imgs.size(); ++i) {
            miniBatch.push_back(align_imgs[i]);
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
