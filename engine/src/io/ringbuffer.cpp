#include "ringbuffer.h"

namespace dg {

RingBuffer::RingBuffer(unsigned int buffSize, unsigned int frameWidth,
                       unsigned int frameHeight)
    : buffer_size_(buffSize),
      frame_width_(frameWidth),
      frame_height_(frameHeight),
      cur_write_pos_(-1),
      cur_read_pos_(-1),
      cur_tracked_pos_(0) {

    content_ = vector<Frame *>(buffer_size_);
    // pre-init the content in the buffer
    for (int i = 0; i < buffer_size_; ++i) {
        content_[i] = new Frame(-1, frameWidth, frameHeight, NULL);
    }

}

RingBuffer::~RingBuffer() {
    for (int i = 0; i < content_.size(); ++i) {
        Frame *f = content_[i];
        if (f != NULL) {
            delete f;
        }
    }
}

void RingBuffer::SetFrame(long long frameId, unsigned int dataWidth,
                          unsigned int dataHeight, unsigned char *data) {
    int pos = cur_write_pos_ + 1;
    if (pos >= buffer_size_) {
        pos = 0;
    }

    Frame *oldFrame = content_[pos];
    if (oldFrame == NULL) {
        LOG(ERROR) << "Ringbuffer not initailized" << endl;
        return;
    }

//    unsigned long long frameSleep = 0;


    while (!oldFrame->CheckStatus(FRAME_STATUS_INIT)
        && !oldFrame->CheckStatus(FRAME_STATUS_FINISHED)) {

        LOG(WARNING) << "Skip frame because the old frame not ready: " << oldFrame->status() << endl;
        return;
    }

    oldFrame->Reset();
    oldFrame->set_id(frameId);
    oldFrame->payload()->Update(dataWidth, dataHeight, data);
    oldFrame->set_status(FRAME_STATUS_NEW);

    cur_write_pos_ = pos;

}

Frame *RingBuffer::TryNextFrame(int &index) {
    int pos = cur_read_pos_ + 1;
    if (pos >= buffer_size_) {
        pos = 0;
    }
    index = pos;
    return content_[pos];
}

Frame *RingBuffer::NextFrame(int &index) {
    cur_read_pos_++;
    if (cur_read_pos_ >= buffer_size_) {
        cur_read_pos_ = 0;
    }
    index = cur_read_pos_;
    return content_[cur_read_pos_];
}

Frame *RingBuffer::GetFrameOffset(const unsigned int index, const int offset) {
    int pos = index + offset;
    if (pos >= ((int) buffer_size_)) {
        pos = pos % ((int) buffer_size_);
    } else if (pos < 0 && abs(pos) < ((int) buffer_size_)) {
        pos = ((int) buffer_size_) + pos;
    } else if (pos < 0 && abs(pos) >= ((int) buffer_size_)) {
        pos = ((int) buffer_size_) + (pos % ((int) buffer_size_));
    }
    return GetFrame(pos);
}
}