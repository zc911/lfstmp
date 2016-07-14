/*
 * classifier.h
 *
 *  Created on: Aug 13, 2015
 *      Author: chenzhen
 */

#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#include <utility>
#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;
namespace dg {

/**
 * The basic classifier interface which defines the functions
 * a real classifier must implement.
 */
class Classifier {
public:
    Classifier() {
    }
    virtual ~Classifier() {
    }
    virtual int ClassifyBatch(
        const vector<cv::Mat> &imgs,
        vector<vector<Prediction> > &predict_results) = 0;
};
}
#endif /* CLASSIFIER_H_ */
