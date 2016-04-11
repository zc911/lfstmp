/*
 * simple_engine.h
 *
 *  Created on: 11/04/2016
 *      Author: chenzhen
 */

#ifndef SIMPLE_ENGINE_H_
#define SIMPLE_ENGINE_H_
#include "engine.h"
#include "model/model.h"
#include "model/frame.h"
#include "io/stream_tube.h"
#include "processor/simple_processor.h"
#include "vis/display.h"

namespace dg {

class SimpleEngine : public Engine {
 public:

    SimpleEngine(RingBuffer *buffer, Displayer *displayer)
            : Engine(),
              buffer_(buffer),
              cur_frame_(0) {
        displayer_ = displayer;
        processor_ = new SimpleProcessor();
        tube_ = new StreamTube(buffer, "/home/chenzhen/video/road1.mp4", 25,
                               640, 480, true);

    }

    virtual ~SimpleEngine() {

    }

    virtual void Process() {
        tube_->StartAsyn();
        cout << "start process" << endl;
        while (1) {
            if (buffer_->IsEmpty()) {
                continue;
            }
            if (cur_frame_ >= buffer_->Size()) {
                continue;
            }
            Frame *f = buffer_->Get(cur_frame_);
            if (f == NULL) {
                continue;
            }
            cur_frame_++;
            processor_->Update(f);
            usleep(40 * 1000);

        }
    }
    virtual int Stop() {
        return 0;
    }
    virtual int Release() {
        return 0;
    }

 private:
    StreamTube *tube_;
    RingBuffer *buffer_;
    Displayer *displayer_;
    Processor *processor_;
    unsigned int cur_frame_;
}
;

}

#endif /* SIMPLE_ENGINE_H_ */
