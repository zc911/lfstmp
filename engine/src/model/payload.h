/*
 * payload.h
 *
 *  Created on: 01/04/2016
 *      Author: chenzhen
 */

#ifndef PAYLOAD_H_
#define PAYLOAD_H_

#include <opencv2/core/core.hpp>

#include "basic.h"
#include "model.h"

namespace deepglint {

class Payload {
 public:
    Payload(Identification id, cv::Mat data)
            : id_(id),
              data_(data) {

    }
    ~Payload() {

    }
 private:
    Identification id_;
    cv::Mat data_;
};

}

#endif /* PAYLOAD_H_ */
