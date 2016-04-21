/*
 * window_detector.h
 *
 *  Created on: Apr 19, 2016
 *      Author: jiajaichen
 */

#ifndef SRC_ALG_WINDOW_CAFFE_DETECTOR_H_
#define SRC_ALG_WINDOW_CAFFE_DETECTOR_H_
#include <cassert>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>s

#include <caffe/caffe.hpp>
#include "model/basic.h"
#include "detector.h"
#include "caffe_config.h"
using namespace std;
using namespace caffe;
using namespace cv;
namespace dg {
class WindowCaffeDetector {
 public:
     WindowCaffeDetector(CaffeConfig &config);
     virtual ~WindowCaffeDetector();


     virtual int DetectBatch(const vector<cv::Mat> &img,const vector<cv::Mat> &resized_img,vector<vector<Detection> > &detect_results);
 protected:
     vector<Rect> Detect(vector<Mat> resized_imgs,
                         vector<Mat> imgs);
     std::vector<Blob<float>*> PredictBatch(vector<Mat> imgs);
     void WrapBatchInputLayer(vector<vector<Mat> > *input_batch);

     void PreprocessBatch(const vector<Mat> imgs,
                          vector<vector<Mat> >* input_batch);
 private:
     boost::shared_ptr<caffe::Net<float> > net_;
     int num_channels_;
     cv::Size input_geometry_;
     bool device_setted_;
     CaffeConfig caffe_config_;
     Mat means_;
     int rescale_;

};
}
#endif /* SRC_ALG_WINDOW_CAFFE_DETECTOR_H_ */
