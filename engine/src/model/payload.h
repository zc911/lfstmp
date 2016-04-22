/*
 * payload.h
 *
 *  Created on: 01/04/2016
 *      Author: chenzhen
 */

#ifndef PAYLOAD_H_
#define PAYLOAD_H_

#include <iostream>
#include <opencv2/core/core.hpp>

#include "basic.h"
#include "model.h"

using namespace std;
namespace dg {

class Payload {
 public:

    // TODO init data_ as YUV and rbg_ as BGR format
    Payload(Identification id, unsigned int width, unsigned int height,
            unsigned char *data)
            : id_(id) {
        cv::Mat tmp = cv::Mat(height, width, CV_8UC4, data);
        tmp.copyTo(data_);

    }

    Payload(Identification id, cv::Mat data)
            : id_(id),
              data_(data) {
    }

    ~Payload() {
        data_.release();
        rgb_.release();
    }

    cv::Mat data() {
        return data_;
    }
 private:
    Identification id_;
    cv::Mat data_;
    cv::Mat rgb_;
};

}

#endif /* PAYLOAD_H_ */
