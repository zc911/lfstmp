#include "ringbuffer.h"

namespace dg {

RingBuffer::RingBuffer(unsigned int capacity)
        : capacity_(capacity),
          last_id_(0) {
    buffer_ = boost::circular_buffer<Frame*>(capacity);
}
RingBuffer::~RingBuffer() {
    for (int i = 0; i < buffer_.size(); ++i) {
        Frame *f = buffer_[i];
        if (f)
            delete f;
    }
}

int RingBuffer::TryPut(Frame *frame) {
    buffer_.push_back(frame);
    return 0;
}
int RingBuffer::TryPut(unsigned int width, unsigned int height,
                       unsigned char *data) {
    Frame *f = Front();
    if (f != NULL) {
        if (f->status() == FRAME_STATUS_FINISHED) {
            delete f;
            f = NULL;
        } else {
            cout << "skip put frame" << endl;
            return -1;
        }
    }

    f = new Frame(last_id_++, width, height, data);
    TryPut(f);
    return 0;
}
Frame* RingBuffer::Get(unsigned int index) {
    int i = index;
    if (IsEmpty()) {
        return NULL;
    }
    if (index < 0 || index >= buffer_.size()) {
        index = 0;
    }
    return buffer_[index];
}
Frame* RingBuffer::Front() {
    if (IsEmpty()) {
        return NULL;
    }
    return buffer_.front();
}

Frame* RingBuffer::Back() {
    return buffer_.back();
}

bool RingBuffer::IsEmpty() {
    return buffer_.empty();
}

unsigned int RingBuffer::Size() {
    return buffer_.size();
}

}
