/*
 * simple_processor.h
 *
 *  Created on: 11/04/2016
 *      Author: chenzhen
 */

#ifndef SIMPLE_PROCESSOR_H_
#define SIMPLE_PROCESSOR_H_

#include "processor.h"
#include <iostream>

using namespace std;

namespace dg {

class SimpleProcessor : public Processor {
 public:
    SimpleProcessor()
            : Processor() {

    }
    virtual ~SimpleProcessor() {

    }
    virtual void Update(Frame *frame) {
        if (!checkOperation(frame)) {
            cout << "operation no allowed" << endl;
            return;
        }
        if (!checkStatus(frame)) {
            cout << "check status failed " << endl;
            return;
        }
        cout << "process frame: " << frame->id() << endl;
        usleep(100 * 1000);
        frame->set_status(FRAME_STATUS_FINISHED);
    }
    virtual void Update(FrameBatch *frameBatch) {

    }
//    virtual Frame* operator()(Frame* frame) {
//        return NULL;
//    }
//    virtual FrameBatch* operator()(FrameBatch* frameBatch) {
//        return NULL;
//    }
    virtual bool checkOperation(Frame *frame) {
        return true;
    }
    virtual bool checkStatus(Frame *frame) {
        return frame->status() == FRAME_STATUS_FINISHED ? false : true;
    }

};

}

#endif /* SIMPLE_PROCESSOR_H_ */
