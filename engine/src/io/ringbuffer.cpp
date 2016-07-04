#include "ringbuffer.h"

namespace dg {

RingBuffer::RingBuffer(unsigned int buffSize, unsigned int frameWidth,
                       unsigned int frameHeight)
    : buffer_size_(buffSize),
      max_frame_width_(frameWidth),
      max_frame_height_(frameHeight),
      cur_write_pos_(-1),
      cur_read_pos_(-1),
      cur_tracked_pos_(0) {
    content_ = vector<Frame *>(buffer_size_);
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

//void RingBuffer::SetFrame(Frame *f) {
//    if (f == NULL) {
//        DLOG(INFO) << "Set NULL frame" << endl;
//        return;
//    }
//    int pos = cur_write_pos_ + 1;
//    if (pos >= buffer_size_) {
//        pos = 0;
//    }
//    Frame *oldFrame = content_[pos];
//    unsigned long long frameSleep = 0;
//
//    while (oldFrame != NULL
//        && (oldFrame->GetStatus() & FRAME_STATUS_FINISHED) == 0) {
//        DLOG(INFO) << "Can not Write frame " << dec << f->FrameId() << ". The old frame status: " << dec
//            << oldFrame->GetStatus() << " id: " << dec << oldFrame->FrameId() << "address: " << dec << oldFrame << endl;
//        return;
//        usleep(RAW_FRAME_SLEEP_INTERVAL);
//        frameSleep += RAW_FRAME_SLEEP_INTERVAL;
//        if (frameSleep >= RAW_FRAME_SLEEP_TIMEOUT) {
//            LOG(WARNING) << "Frame raw sleep timeout" << endl;
//            delete f;
//            return;
//        }
//
//        oldFrame = content_[pos];
//    }
//    if (oldFrame != NULL) {
//        delete oldFrame;
//    }
//    cur_write_pos_ = pos;
//    content_[cur_write_pos_] = f;
//    DLOG(INFO) << "Write frame: " << content_[cur_write_pos_]->FrameId() << endl;
//
//}

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
    oldFrame->set_status(FRAME_STATUS_ABLE_TO_DISPLAY);
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