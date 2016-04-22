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

    virtual void Update(Frame *frame) = 0;
    virtual void Update(FrameBatch *frameBatch) = 0;
//    virtual Frame* operator()(Frame* frame);
//    virtual FrameBatch* operator()(FrameBatch* frameBatch);

    virtual bool checkOperation(Frame *frame) = 0;
    virtual bool checkStatus(Frame *frame) = 0;

 protected:
    Processor *next_;

};
}
#endif /* PROCESSOR_H_ */
