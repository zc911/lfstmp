/*
 * processor_tbbref.h
 *
 *  Created on: 01/04/2016
 *      Author: chenzhen
 */

#ifndef PROCESSOR_TBB_REF_H_
#define PROCESSOR_TBB_REF_H_
namespace deepglint {

class ProcessorRef {
 public:
    ProcessorRef()
            : proc_(NULL),
              next_(NULL) {

    }
    virtual ~ProcessorRef() {

    }
    virtual Frame* operator()(Frame* f) = 0;
    void SetNext(ProcessorRef *next) {
        if (next_ != NULL) {
            delete next_;
            next_ = NULL;
        }
        next_ = next;
    }
 protected:
    Processor *proc_;
    ProcessorRef *next_;
};

}
#endif /* PROCESSOR_TBB_REF_H_ */
