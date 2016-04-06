/*
 * basic.h
 *
 *  Created on: 29/03/2016
 *      Author: chenzhen
 */

#ifndef BASIC_H_
#define BASIC_H_

#include <vector>
#include <utility>
#include <opencv2/core/core.hpp>

using namespace std;

namespace deepglint {

typedef long long int Identification;
typedef float Confidence;
typedef long long int Timestamp;
typedef pair<int, float> Prediction;
typedef vector<uchar> Feature;

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

typedef cv::Rect Box;

// TODO
//typedef struct {
//    int id;
//    float confidence;
//    Box rect;
//    Box gt;
//    bool deleted;
//} BoundingBox;

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

typedef struct {
    Identification id;
    Timestamp timestamp;
    MessageStatus status;
    MetaData *video_meta_data;
    Object *object;
} Message;

}

#endif /* BASIC_H_ */
