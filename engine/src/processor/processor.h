/*
 * processor.h
 *
 *  Created on: Jan 4, 2016
 *      Author: chenzhen
 */

#ifndef PROCESSOR_H_
#define PROCESSOR_H_

#include "model/basic.h"
#include "model/model.h"

namespace dg {
/**
 * the basic processor interface.
 */
class Processor {
 public:
    Processor()
            : next_(0) {

    }
    virtual ~Processor() {

    }

    Processor* SetNextProcessor(Processor *proc) {
        Processor *old = next_;
        next_ = proc;
        return old;
    }

    Processor* GetNextProcessor() {
        return next_;
    }

    virtual bool Proceed(Frame *frame) {
        if (next_ != NULL) {
            next_->Update(frame);
            return true;
        }
        return false;
    }

    virtual void Update(Frame *frame) = 0;
    virtual void Update(FrameBatch *frameBatch) = 0;

    virtual bool checkOperation(Frame *frame) = 0;
    virtual bool checkStatus(Frame *frame) = 0;

 protected:
    Processor *next_;

};
}
#endif /* PROCESSOR_H_ */
