/*
 * processor.h
 *
 *  Created on: Jan 4, 2016
 *      Author: chenzhen
 */

#ifndef PROCESSOR_H_
#define PROCESSOR_H_

#include "model/model.h"
namespace deepglint {
/**
 * the basic processor interface.
 */
class Processor {
 public:
    Processor()
            : next_(NULL) {

    }
    virtual ~Processor() {

    }
    virtual void Update(Frame *frame) = 0;
 protected:
    Processor *next_;
};
}
#endif /* PROCESSOR_H_ */
