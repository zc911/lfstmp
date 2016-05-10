/*
 * ringbuffer.h
 *
 *  Created on: 05/04/2016
 *      Author: chenzhen
 */

#ifndef RINGBUFFER_H_
#define RINGBUFFER_H_
#include "model/frame.h"
#include <boost/circular_buffer.hpp>

namespace dg {

class RingBuffer {
 public:
    RingBuffer(unsigned int capacity);
    ~RingBuffer();

    int TryPut(Frame *frame);
    int TryPut(unsigned int width, unsigned int height, unsigned char *data);
    void Put(Frame *frame);
    Frame* Get(unsigned int index);
    Frame* Front();
    Frame* Back();
    bool IsEmpty();
    unsigned int Size();

 private:
    boost::circular_buffer<Frame*> buffer_;
    unsigned int capacity_;
    Identification last_id_;
};

}

#endif /* RINGBUFFER_H_ */
