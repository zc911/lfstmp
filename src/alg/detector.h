/*
 * detector.h
 *
 *  Created on: Aug 12, 2015
 *      Author: chenzhen
 */

#ifndef DETECTOR_H_
#define DETECTOR_H_

#include <opencv2/core/core.hpp>
#include <algorithm>

#include "model/basis.h"

using namespace cv;

namespace deepglint {

class Detector {
 public:
    Detector(const string model) {
    }
    virtual ~Detector() {
    }

    virtual vector<BoundingBox> Detect(const Mat &img,
                                       const int target_image_size) = 0;

};
}
#endif /* DETECTOR_H_ */
