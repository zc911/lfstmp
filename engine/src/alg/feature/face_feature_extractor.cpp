/*============================================================================
 * File Name   : face_feature_extractor.cpp
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年4月21日 下午1:31:28
 * Description : 
 * ==========================================================================*/
#include "face_feature_extractor.h"

namespace dg {

FaceFeatureExtractor::FaceFeatureExtractor(
    const FaceFeatureExtractorConfig &config)
    : device_setted_(false),
      batch_size_(config.batch_size),
      detector_(dlib::get_frontal_face_detector()) {
    use_gpu_ = config.use_gpu;
    gpu_id_ = config.gpu_id;
    if (use_gpu_) {
        Caffe::SetDevice(config.gpu_id);
        Caffe::set_mode(Caffe::GPU);
        use_gpu_ = true;
    } else {
        Caffe::set_mode(Caffe::CPU);
        use_gpu_ = false;
    }

    layer_name_ = "eltwise6";


        ModelsMap *modelsMap = ModelsMap::GetInstance();

    string deploy_content;
    modelsMap->getModelContent(config.deploy_file,deploy_content);
    net_.reset(
        new Net<float>(config.deploy_file,deploy_content,TEST));
    string model_content;
    modelsMap->getModelContent(config.model_file,model_content);
        net_->CopyTrainedLayersFrom(config.model_file,model_content);


    Blob<float> *input_layer = net_->input_blobs()[0];
    do {
        std::vector<int> shape = input_layer->shape();
        shape[0] = batch_size_;
        input_layer->Reshape(shape);
        net_->Reshape();
    } while (0);
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
    const std::vector<Mat> &imgs) {
    if (!device_setted_) {
        Caffe::SetDevice(gpu_id_);
        Caffe::set_mode(Caffe::GPU);
        device_setted_ = true;
    }
    std::vector<Mat> align_imgs = Align(imgs);
    std::vector<FaceRankFeature> features;
    Blob<float> *input_blob = net_->input_blobs()[0];
    assert(align_imgs.size() <= batch_size_);
    features.resize(align_imgs.size());
    float *input_data = input_blob->mutable_cpu_data();
    int cnt = 0;
    for (size_t i = 0; i < align_imgs.size(); i++) {
        Mat sample;
        Mat img = align_imgs[i];

        if (img.channels() == 3 && num_channels_ == 1)
            cvtColor(img, sample, CV_BGR2GRAY);
        else if (img.channels() == 4 && num_channels_ == 1)
            cvtColor(img, sample, CV_BGRA2GRAY);
        else
            sample = img;

        assert(sample.channels() == 1);
        assert((sample.rows == input_geometry_.height)
                   && (sample.cols == input_geometry_.width));
        for (int i = 0; i < sample.rows; i++) {
            for (int j = 0; j < sample.cols; j++) {
                input_data[cnt] = sample.at<uchar>(i, j) / 255.0f;
                cnt += 1;
            }
        }
    }

    net_->ForwardPrefilled();
    if (use_gpu_) {
        cudaDeviceSynchronize();
    }

    auto output_blob = net_->blob_by_name(layer_name_);
    const float *output_data = output_blob->cpu_data();
    for (size_t i = 0; i < align_imgs.size(); i++) {
        InnFaceFeature feature;
        const float *data = output_data
            + i * sizeof(InnFaceFeature) / sizeof(float);
        memcpy(&feature, data, sizeof(InnFaceFeature));

        FaceRankFeature face_feature;
        for (int j = 0; j < 256; ++j) {
            face_feature.descriptor_.push_back(feature.data[j]);
        }
        features[i] = face_feature;
    }
    return features;
}

FaceFeatureExtractor::~FaceFeatureExtractor() {
}

} /* namespace dg */
