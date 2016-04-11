/*
 * caffe_detector.h
 *
 *  Created on: Sep 28, 2015
 *      Author: irene
 */

#ifndef SRC_ALG_CAFFE_DETECTOR_H_
#define SRC_ALG_CAFFE_DETECTOR_H_

#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ctime>
#include <cassert>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/caffe.hpp"
#include "model/basic.h"
#include "detector.h"
#include "caffe_config.h"

using namespace std;

namespace dg {

class CaffeDetector : public Detector {
 public:
    CaffeDetector(const string& model_file, const string& trained_file,
                  const bool use_GPU, const int batch_size,
                  const int gpuId = 0);
    virtual ~CaffeDetector();
    virtual vector<BoundingBox> Detect(const string image_filename,
                                       const int target_image_size);
    virtual vector<BoundingBox> Detect(const Mat & img,
                                       const int target_image_size);
    void ChangeMean(float a, float b, float c);  //change mean before read imgs
    void ChangeTargetSize(float target_min_size, float target_max_size);
 protected:
    void SetMean();

    vector<Blob<float>*> PredictBatch(const vector<Mat>& imgs);

    void WrapBatchInputLayer(vector<vector<Mat> > *input_batch);

    void PreprocessBatch(const vector<Mat> imgs,
                         vector<vector<Mat> >* input_batch);
    float target_min_size_;
    float target_max_size_;
    boost::shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    int batch_size_;
    int scale_num;

    float means_[3];
    cv::Mat mean_;
    bool rescale_;
 private:
    int gpu_id_;
    bool device_setted_;
};
class CaffeDetectorAdvance : public CaffeDetector {
 public:
    CaffeDetectorAdvance(const string& model_file, const string& trained_file,
                         const bool use_GPU, const int batch_size,
                         const int gpuId = 0);
    virtual ~CaffeDetectorAdvance();
    virtual vector<BoundingBox> Detect(const string image_filename,
                                       const int target_image_size);
    virtual vector<BoundingBox> Detect(const Mat & img,
                                       const int target_image_size);
    void setPrimaryResult(vector<BoundingBox>&);
 private:

    vector<BoundingBox> vbbox_;
};
}
#endif /* SRC_ALG_CAFFE_DETECTOR_H_ */
