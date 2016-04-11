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

namespace dg {

class Payload {
 public:

    // TODO
    Payload(Identification id, unsigned int width, unsigned int height,
            unsigned char *data)
            : id_(id) {
        cv::Mat tmp = cv::Mat(height, width, CV_8UC3, data);
        tmp.copyTo(data_);

    }
    ~Payload() {
        data_.release();
        rgb_.release();
    }
 private:
    Identification id_;
    cv::Mat data_;
    cv::Mat rgb_;
};

}

#endif /* PAYLOAD_H_ */
