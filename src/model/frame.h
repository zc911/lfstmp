/*
 * Frame.h
 *
 *  Created on: 01/04/2016
 *      Author: chenzhen
 */

#ifndef FRAME_H_
#define FRAME_H_

#include <vector>
#include <pthread.h>

#include "payload.h"

namespace deepglint {

using namespace std;

typedef enum {

} FrameType;

typedef enum {

} FrameStatus;

class Frame {
 public:
    Frame();
    ~Frame();
 private:
    volatile Identification id_;
    volatile Timestamp timestamp_;
    volatile FrameType type_;
    volatile FrameStatus status_;
    pthread_mutex_t status_lock_;
    pthread_mutex_t type_lock_;
    Payload payload_;
    vector<Object *> objects_;
    cv::Mat render_data_;
};

class FrameBatch {
 public:
    FrameBatch();
    ~FrameBatch();
 private:
    Identification id_;
    unsigned int batch_size_;
    vector<Frame> frames_;
};

}

#endif /* FRAME_H_ */
