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
        if (data != NULL) {
            cv::Mat tmp = cv::Mat(height, width, CV_8UC4, data);
            tmp.copyTo(data_);
        }

    }

    Payload(Identification id, cv::Mat data)
        : id_(id),
          data_(data) {
    }

    ~Payload() {
        data_.release();
        rgb_.release();
    }

    void Update(unsigned int width, unsigned int height, unsigned char *data) {

        if (data == NULL) {
            LOG(ERROR) << "Data is null, update payload failed" << endl;
            return;
        }
        //data_.release();
        if (data_.cols & data_.rows != 0) {
            if (width != data_.cols || height != data_.rows) {
                LOG(ERROR) << "Input data invalid resolution: " << width << "*" << height << endl;
                return;
            }
        }

        cv::Mat tmp = cv::Mat(height, width, CV_8UC4, data);
        tmp.copyTo(data_);
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
