#ifndef  __PYRAMID_DENSEBOX_H_
#define  __PYRAMID_DENSEBOX_H_
#include <string>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>

// #include "fcn_detector.h"
// #include "caffe_interface.h"
#include "caffe/caffe.hpp"
#include "detector/pryd_util.hpp"

namespace db {
// using namespace std;
// using namespace cv;

#define FACE_TEMPLATE_SIZE 32
#define MIN_SCALE_FACE_TO_IMAGE 0.01f
#define MIN_IMAGE_SIZE 20
#define MAX_IMAGE_SIZE 5000 
#define NMS_THRESH     0.1

// using vis::ICaffeBlob;
// using vis::ICaffePredict;
// using vis::ICaffePreProcess;
// using vis::createcaffepreprocess;
// using vis::createcaffeblob;
// using vis::CaffePreProcessParam;
// using vis::CaffeParam;

class PyramidDenseBox{
    public:
        PyramidDenseBox()
        {
            templateSize_      = FACE_TEMPLATE_SIZE;
            minDetFaceSize_    = 24;
            maxDetFaceSize_    = -1 ;
            minImgSize_        = MIN_IMAGE_SIZE;
            maxImgSize_        = MAX_IMAGE_SIZE;

            minScaleFaceToImg_ = MIN_SCALE_FACE_TO_IMAGE;
            maxScaleFaceToImg_ = 1.0f;
            stepScale_         = 2.0f;

            heat_map_a_        = 4.0f;
            heat_map_b_        = 2.0f;

            mean_r_            = 94.7109f;
            mean_g_            = 99.1183f;
            mean_b_            = 95.7652f;

            mvnPower_          = 1.0f;
            mvnScale_          = 0.01f;
            mvnShift_          = 0.0f;

            pad_w_             = 24;
            pad_h_             = 24;
            max_stride_        = 8;

            class_num_         = 1;
            channel_per_scale_ = 5;

            nms_threshold_     = 0.7f;
            nms_overlap_ratio_ = 0.5f;
            nms_top_n_         = 100;
        }

        PyramidDenseBox(float minDetFaceSize, float maxDetFaceSize)
        {
            templateSize_      = FACE_TEMPLATE_SIZE;
            minDetFaceSize_    = minDetFaceSize;
            maxDetFaceSize_    = maxDetFaceSize;
            minImgSize_        = MIN_IMAGE_SIZE;
            maxImgSize_        = MAX_IMAGE_SIZE;

            minScaleFaceToImg_ = MIN_SCALE_FACE_TO_IMAGE;
            maxScaleFaceToImg_ = 1.0f;
            stepScale_         = 2.0f;

            heat_map_a_        = 4.0f;
            heat_map_b_        = 2.0f;

            mean_r_            = 94.7109f;
            mean_g_            = 99.1183f;
            mean_b_            = 95.7652f;

            mvnPower_          = 1.0f;
            mvnScale_          = 0.01f;
            mvnShift_          = 0.0f;

            pad_w_             = 24;
            pad_h_             = 24;
            max_stride_        = 8;

            class_num_         = 1;
            channel_per_scale_ = 5;

            nms_threshold_     = 0.7f;
            nms_overlap_ratio_ = 0.5f;
            nms_top_n_         = 100;

        }

        PyramidDenseBox(float minDetFaceSize, float maxDetFaceSize, float minScaleFaceToImg)
        {
	    templateSize_      = FACE_TEMPLATE_SIZE;
            minDetFaceSize_    = minDetFaceSize;
            maxDetFaceSize_    = maxDetFaceSize;
            minImgSize_        = MIN_IMAGE_SIZE;
            maxImgSize_        = MAX_IMAGE_SIZE;

            minScaleFaceToImg_ = minScaleFaceToImg;
            maxScaleFaceToImg_ = 1.0f;
            stepScale_         = 2.0f;

            heat_map_a_        = 4.0f;
            heat_map_b_        = 2.0f;

            mean_r_            = 94.7109f;
            mean_g_            = 99.1183f;
            mean_b_            = 95.7652f;

            mvnPower_          = 1.0f;
            mvnScale_          = 0.01f;
            mvnShift_          = 0.0f;

            pad_w_             = 24;
            pad_h_             = 24;
            max_stride_        = 8;

            class_num_         = 1;
            channel_per_scale_ = 5;

            nms_threshold_     = 0.7f;
            nms_overlap_ratio_ = 0.5f;
            nms_top_n_         = 100;
	}
        ~PyramidDenseBox(){};

    private:
        int templateSize_;
        int minDetFaceSize_;
        int maxDetFaceSize_;
        int minImgSize_;
        int maxImgSize_;

        float minScaleFaceToImg_;
        float maxScaleFaceToImg_;
        float stepScale_;

        float heat_map_a_;
        float heat_map_b_;

        float mean_r_;
        float mean_g_;
        float mean_b_;

        float mvnPower_;
        float mvnScale_;
        float mvnShift_;

        int pad_w_;
        int pad_h_;
        int max_stride_;

        int class_num_;
        int channel_per_scale_;

        float nms_threshold_;
        float nms_overlap_ratio_;
        float nms_top_n_;

    public:
        //detection rbox
        bool predictPyramidDenseBox( caffe::shared_ptr<caffe::Net<float> > caffe_net, cv::Mat &img, std::vector< RotateBBox<float> >& rotatedFaces);
        bool predictPyramidDenseBox( caffe::shared_ptr<caffe::Net<float> > caffe_net, const std::vector<cv::Mat> &img, std::vector<std::vector< RotateBBox<float> > >& rotatedFaces);
    private:
        bool setDetSize(const int imgWidth, const int imgHeight, const int template_size, 
                const int minDetFaceSize, const int maxDetFaceSize, 
                const int minImgSize, const int maxImgSize, 
                const float minScaleFaceToImg, const float maxScaleFaceToImg, 
                float& scale_start, float& scale_end);
        bool constructPyramidImgs(const cv::Mat & img, std::vector<cv::Mat>& pyramidImgs); 
        template <typename Dtype> 
        void setImgDenseBox(Dtype* pblob, cv::Mat& img, cv::Scalar img_mean, float mvnPower, float mvnScale, float mvnShift);
        void predictDenseBox(caffe::shared_ptr<caffe::Net<float> > caffe_net, cv::Mat& img, std::vector< RotateBBox<float> >& faces);
        void predictDenseBox(caffe::shared_ptr<caffe::Net<float> > caffe_net, const std::vector<cv::Mat>& imgs, std::vector<std::vector<RotateBBox<float> > >& faces);
        void predictDenseBox(caffe::shared_ptr<caffe::Net<float> > caffe_net, const cv::Mat& org_img, const std::vector<cv::Mat>& pyramid_imgs, std::vector< RotateBBox<float> >& faces);
	void dataToBBox(const float* output_data, const std::vector<int>& data_shape, std::vector<RotateBBox<float> >& faces);
	void DBoxMatToBlob(const cv::Mat& org_img, cv::Size extend_size, float* blob_data);

};
}
#endif
