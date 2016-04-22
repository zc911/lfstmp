/*
 * basic.h
 *
 *  Created on: 29/03/2016
 *      Author: chenzhen
 */

#ifndef BASIC_H_
#define BASIC_H_

#include <stdint.h>
#include <vector>
#include <utility>
#include <opencv2/core/core.hpp>

using namespace std;

namespace dg {

typedef int64_t Identification;
typedef float Confidence;
typedef int64_t Timestamp;
typedef pair<int, float> Prediction;
typedef vector<uchar> Feature;
typedef cv::Rect Box;
typedef uint64_t Operation;

enum ContentType {
    IMAGE_JPEG = 1,
    FILE_MP4 = 2,
    FILE_AVI = FILE_MP4,
    STREAM_RTSP = 4
};

enum MessageStatus {
    MESSAGE_STATUS_INIT = 1,
    MESSAGE_STATUS_SENT = 2,
};

enum Operations {
    OPERATION_DETECT = 1,
    OPERATION_TRACK = 2,
    OPERATION_VEHICLE_STYLE = 4,
    OPERATION_VEHICLE_COLOR = 8,
    OPERATION_VEHICLE_PLATE = 16,
};

typedef struct {
    int id;
    string uri;
    ContentType type;
    unsigned int width;
    unsigned int height;
    unsigned int chanel;
} MetaData;

typedef struct VideoMetaData : public MetaData {
    unsigned int fps;
} VideoMetaData;

}

#endif /* BASIC_H_ */
