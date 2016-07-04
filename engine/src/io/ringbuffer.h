/*
 * ringbuffer.h
 *
 *  Created on: Dec 23, 2015
 *      Author: chenzhen
 */

#ifndef RINGBUFFER_H_
#define RINGBUFFER_H_

#include "model/frame.h"
#include <vector>

using namespace std;

namespace dg {

const unsigned int RINGBUFFER_DEFAULT_SIZE = 120;
const int RAW_FRAME_SLEEP_INTERVAL = 20 * 1000;
const unsigned long long RAW_FRAME_SLEEP_TIMEOUT = 60000 * 1000;

/**
 * This class defines a ring buffer
 */
class RingBuffer {
public:
    RingBuffer(unsigned int buffSize, unsigned int frameWidth,
               unsigned int frameHeight);

//    static RingBuffer &Instance(unsigned int frameWidth,
//                                unsigned int frameHeight, int size =
//    RINGBUFFER_DEFAULT_SIZE) {
//        if (RingBuffer::instance_ == NULL) {
//            RingBuffer::instance_ = new RingBuffer(size, frameWidth,
//                                                   frameHeight);
//        }
//        return *RingBuffer::instance_;
//    }
    ~RingBuffer();

    inline unsigned int BufferSize() {
        return buffer_size_;
    }

    /**
     * Get frame by abstract index. 0 <= index < buffer_size
     */
    inline Frame *GetFrame(const unsigned int index) {
        int i = index;
        if (i < 0) {
            i = 0;
        }
        if (i >= buffer_size_) {
            i = i % buffer_size_;
        }
        return content_[i];
    }
    /**
     * Get frame by the offset of index.
     * If offset > 0, get the newer frame.
     * If offset < 0, get the older frame.
     */
    Frame *GetFrameOffset(const unsigned int index, const int offset);

    /**
     * Get the next tracking frame index.
     * See also LatestDetectedFrame()
     */
    inline unsigned int NextTrackingFrameIndex() {
        return cur_tracked_pos_;
    }
    inline void SetNextTrackingFrameIndex(unsigned int index) {
        if (index >= buffer_size_) {
            index = index % buffer_size_;
        }
        cur_tracked_pos_ = index;
    }

    /**
     * Get the next tracking frame
     * See also LatestDetectedFrameIndex()
     */
    inline Frame *NextTrackingFrame(int &index) {
        index = NextTrackingFrameIndex();
        return GetFrame(index);
    }

    inline Frame *LatestFrame() {
        return GetFrame(cur_write_pos_);
    }

    inline int CurrentIndex() {
        return cur_read_pos_;
    }

    void SetFrame(Frame *f);
    void SetFrame(long long frameId, unsigned int width, unsigned int height,
                  unsigned char *data);

    Frame *TryNextFrame(int &index);
    Frame *NextFrame(int &index);

private:
//    static RingBuffer *instance_;


    unsigned int buffer_size_;
    unsigned int max_frame_width_;
    unsigned int max_frame_height_;
    vector<Frame *> content_;
    volatile unsigned int cur_write_pos_;
    volatile unsigned int cur_read_pos_;
    volatile unsigned int cur_tracked_pos_;

};
}

#endif /* RINGBUFFER_H_ */
