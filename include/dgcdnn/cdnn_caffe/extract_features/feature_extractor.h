/*
 * Author : Wengrenliang 
 * email : wengrenliang@baidu.com 
 * The implementation for features extraction based on caffe
 */
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h> 
#include "caffe_interface.h"
#include "boost/algorithm/string.hpp"
#include "FaUtil.h"
 
/*
 * FeatureExtractor is preferred when using GPU to extract features;
 * It uses single thread to extract features
 */
class FeatureExtractor {
  public:
    enum face_patch {main_face = 0, left_eye, right_eye, nose, mouth, middle_eye, 
        left_mouth, right_mouth, mini_face, left_brow, right_brow, middle_brow};

    FeatureExtractor();
    FeatureExtractor(const std::string& model_def, const std::string& weight_file, 
        const std::string & layer_name, const std::vector<int> & patch_ids, const bool & is_color);
    ~FeatureExtractor();

    bool setImg(float* pdata, const cv::Mat& img, const std::vector<float> lmks);
    virtual int extract_features(const cv::Mat& img, const std::vector<float>& lmks, 
        std::vector<float>& features);
    bool ReadFacePatchImageToData(const cv::Mat& img, const std::vector <float> & total_landmarks,
        float* transformed_data);
  protected:
    void init(const std::string& model_def, const std::string& weight_file, 
        const std::string & layer_name, const std::vector<int> & patch_ids);
    void * net_;
    bool is_color_;
    std::string layername_;
    std::vector<int> patch_ids_;
};

/*
 * MultiThreadFeatureExtractor is preferred when using CPU to extract features;
 */
class MultiThreadFeatureExtractor: public FeatureExtractor {
  public:
    MultiThreadFeatureExtractor();
    MultiThreadFeatureExtractor(
        const std::vector<std::string> & model_defs, 
        const std::vector<std::string> & weight_files, 
        const std::vector<std::string> & layer_names, 
        const std::vector<int> & patch_ids,
        const bool & is_color);
    ~MultiThreadFeatureExtractor();

    void init(const std::vector<std::string> & model_defs, 
        const std::vector<std::string> & weight_files, 
        const std::vector<std::string> & layer_names, 
        const std::vector<int> & patch_ids);

    virtual int extract_features(const cv::Mat& img, const std::vector<float>& lmks, 
        std::vector<float>& features);
  private:
    std::vector<void *> nets_;
    std::vector<std::string> layernames_;
    int thread_num_;
    std::vector<std::vector<boost::shared_ptr<vis::ICaffeBlob> > > outputblobs_;
    float* blobData_;
};

/*
 * Don't use MultiProcFeatureExtractor, its efficiency is the worst among the three;
 */
class MultiProcFeatureExtractor: public FeatureExtractor {
  public:
    MultiProcFeatureExtractor();
    MultiProcFeatureExtractor(
        const std::vector<std::string> & model_defs, 
        const std::vector<std::string> & weight_files,
        const std::vector<std::string> & layer_names, 
        const std::vector<int> & patch_ids, 
        const std::vector<int> & patch_dims,
        const bool & is_color);
    ~MultiProcFeatureExtractor();
    void init(
        const std::vector<std::string> & model_defs, 
        const std::vector<std::string> & weight_files, 
        const std::vector<std::string> & layer_names, 
        const std::vector<int> & patch_ids, 
        const std::vector<int> & patch_dims);

    virtual int extract_features(const cv::Mat& img, const std::vector<float>& lmks, 
        std::vector<float>& features);
  private:
    std::vector<void *> nets_;
    std::vector<std::string> layernames_;
    std::vector<int> patch_dims_;
    int thread_num_;
    std::vector<std::vector<boost::shared_ptr<vis::ICaffeBlob> > > outputblobs_;
    float* blobData_;
};



