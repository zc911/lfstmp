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
};
}
#endif /* DETECTOR_H_ */
