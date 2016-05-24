/*
 * caffe_helper.h
 *
 *  Created on: Apr 21, 2016
 *      Author: jiajaichen
 */

#ifndef SRC_ALG_CAFFE_HELPER_H_
#define SRC_ALG_CAFFE_HELPER_H_

#include <utility>
#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include "model/basic.h"
#include "model/model.h"

using namespace cv;
using namespace std;

namespace dg {

struct Bbox {
    float confidence;
    Rect rect;
    bool deleted;
    int cls_id;
};

static vector<vector<Mat> > PrepareBatch(const vector<Mat> &image,
                                         int batch_size) {
    vector<vector<Mat> > vimg;
    vector<Mat> img = image;
    if (img.size() == 0)
        return vimg;

    int padding_size = (batch_size - img.size() % batch_size) % batch_size;
    for (int i = 0; i < padding_size; i++) {
        Mat zero = Mat::zeros(img[0].rows, img[0].cols, CV_8UC3);
        img.push_back(zero);
    }
    int idx = 0;
    while (idx < img.size() / batch_size) {
        auto tmp = img.begin() + idx * batch_size;
        vimg.push_back(vector<Mat>(tmp, tmp + batch_size));
        idx++;
    }

    return vimg;
}

static bool PredictionMoreCmp(const Prediction &b1, const Prediction &b2) {
    return b1.second > b2.second;
}
static bool PredictionLessCmp(const Prediction &b1, const Prediction &b2) {
    return b1.second < b2.second;
}
static void SortPrediction(vector<vector<Prediction> > &dstPreds) {
    for (int i = 0; i < dstPreds.size(); i++) {

        vector<Prediction> dstPred = dstPreds.at(i);
        sort(dstPred.begin(), dstPred.end(), PredictionMoreCmp);
        dstPreds[i] = dstPred;

    }
}

static Prediction MaxPrediction(vector<Prediction> &pre) {
    vector<Prediction>::iterator max = max_element(pre.begin(), pre.end(),
                                                   PredictionLessCmp);
    return *max;
}

static bool PairCompare(const std::pair<float, int> &lhs,
                        const std::pair<float, int> &rhs) {
    return lhs.first > rhs.first;
}

static std::vector<int> Argmax(const std::vector<float> &v, int N) {
    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(),
                      PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

static bool detectionCmp(Detection b1, Detection b2) {
    return b1.confidence > b2.confidence;
}

static void detectionNMS(vector<Detection> &p, float threshold) {
    sort(p.begin(), p.end(), detectionCmp);
    int cnt = 0;
    for (int i = 0; i < p.size(); i++) {
        if (p[i].deleted)
            continue;
        cnt += 1;
        for (int j = i + 1; j < p.size(); j++) {
            if (!p[j].deleted) {
                cv::Rect intersect = p[i].box & p[j].box;
                float iou =
                    intersect.area() * 1.0
                        / (p[i].box.area() + p[j].box.area()
                            - intersect.area());
                if (iou > threshold) {
                    p[j].deleted = true;
                }
                if (intersect.x >= p[i].box.x - 0.2
                    && intersect.y >= p[i].box.y - 0.2
                    && (intersect.x + intersect.width)
                        <= (p[i].box.x + p[i].box.width + 0.2)
                    && (intersect.y + intersect.height)
                        <= (p[i].box.y + p[i].box.height + 0.2)) {
                    p[j].deleted = true;

                }
            }
        }
    }
}

static bool BboxCmp(struct Bbox b1, struct Bbox b2) {
    return b1.confidence > b2.confidence;
}

static void NMS(vector<struct Bbox> &p, float threshold) {
    sort(p.begin(), p.end(), BboxCmp);
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i].deleted)
            continue;
        for (size_t j = i + 1; j < p.size(); ++j) {

            if (!p[j].deleted) {
                cv::Rect intersect = p[i].rect & p[j].rect;
                float iou = intersect.area() * 1.0f / p[j].rect.area();
                if (iou > threshold) {
                    p[j].deleted = true;
                }
            }
        }
    }
}

static float ReScaleImage(Mat &img, unsigned int scale) {

    Size resize_r_c;
    float resize_ratio = 1;

    if (img.rows > scale && img.cols > scale) {
        if (img.rows < img.cols) {
            resize_ratio = float(scale) / img.rows;
            resize_r_c = Size(img.cols * resize_ratio, scale);
            resize(img, img, resize_r_c);
        } else {
            resize_ratio = float(scale) / img.cols;
            resize_r_c = Size(scale, img.rows * resize_ratio);
            resize(img, img, resize_r_c);
        }
    }

    return resize_ratio;
}

static void CheckChannel(Mat &img, unsigned char tarChannel, Mat &newImage) {

    if (img.channels() == 3 && tarChannel == 1)
        cvtColor(img, newImage, CV_BGR2GRAY);
    else if (img.channels() == 4 && tarChannel == 1)
        cvtColor(img, newImage, CV_BGRA2GRAY);
    else if (img.channels() == 4 && tarChannel == 3)
        cvtColor(img, newImage, CV_RGBA2BGR);
    else if (img.channels() == 1 && tarChannel == 3)
        cvtColor(img, newImage, CV_GRAY2BGR);
    else
        newImage = img;
}

static void GenerateSample(int num_channels_, cv::Mat &img, cv::Mat &sample) {
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, CV_GRAY2BGR);
    else
        sample = img;
}

}

#endif /* SRC_ALG_CAFFE_HELPER_H_ */
