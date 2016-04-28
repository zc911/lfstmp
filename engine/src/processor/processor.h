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
#include "model/frame.h"

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
        DLOG(INFO)<<"set next processor"<<endl;
        Processor *old = next_;
        next_ = proc;
        return old;
    }

    Processor* GetNextProcessor() {
        return next_;
    }

    virtual void Proceed(FrameBatch *frameBatch) {
        if (next_ != NULL) {
            next_->Update(frameBatch);
        }
    }

    virtual void Update(FrameBatch *frameBatch) = 0;

    virtual void beforeUpdate(FrameBatch *frameBatch) = 0;
    virtual bool checkStatus(Frame *frame) = 0;

protected:
    Processor *next_;

};}
#endif /* PROCESSOR_H_ */
