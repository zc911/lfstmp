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
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

namespace deepglint {

typedef pair<int, float> Prediction;

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

    virtual vector<vector<Prediction> > Classify(const vector<Mat> &imgs) = 0;
    virtual vector<vector<Prediction> > ClassifyBatch(
            const vector<Mat> &imgs) = 0;
};
}
#endif /* CLASSIFIER_H_ */
