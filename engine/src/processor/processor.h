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
/// The basic processor interface. It defines the
/// interfaces each derived processor must to implement.
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

    /// Update the input Frame.
    virtual void Update(Frame *frame) = 0;

    /// Update the input FrameBatch.
    /// A FrameBatch is a package of one or more Frame.
    virtual void Update(FrameBatch *frameBatch) = 0;

    virtual void beforeUpdate(FrameBatch *frameBatch) {};
    virtual bool checkStatus(Frame *frame) = 0;

    /// This method will invoke the next processor chained to the
    /// current processor.
    /// Each processor must invoke Proceed to drive the engine running.
    virtual void Proceed(Frame *frame) {
        if (next_ != NULL) {
            next_->Update(frame);
        }
    }
    /// This method will invoke the next processor chained to the
    /// current processor.
    /// Each processor must invoke Proceed to drive the engine running.
    virtual void Proceed(FrameBatch *frameBatch) {
        if (next_ != NULL) {
            next_->Update(frameBatch);
        }
    }

protected:
    Processor *next_;

};}
#endif /* PROCESSOR_H_ */
