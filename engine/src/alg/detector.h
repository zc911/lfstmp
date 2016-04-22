/*
 * detector.h
 *
 *  Created on: Aug 12, 2015
 *      Author: chenzhen
 */

#ifndef DETECTOR_H_
#define DETECTOR_H_

#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>

#include "model/basic.h"
#include "model/model.h"

using namespace std;

namespace dg {

class Detector {
 public:
    Detector() {
    }
    virtual ~Detector() {
    }

    virtual vector<Detection> Detect(const cv::Mat &img) = 0;
    virtual vector<vector<Detection>> DetectBatch(
            const vector<cv::Mat> &img) = 0;
};
}
#endif /* DETECTOR_H_ */
