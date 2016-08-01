//
// Created by chenzhen on 7/7/16.
//

#ifndef PROJECT_SIMPLE_PARALLEL_PROCESSOR_H_H
#define PROJECT_SIMPLE_PARALLEL_PROCESSOR_H_H
#include "model/basic.h"
#include "processor/processor.h"
#include "processor/parallel/parallel_node.h"

namespace dg {
class BasicParallelProcessor: public ParallelProcessorNode {
public:
    BasicParallelProcessor(Processor *proc) : processor_(proc) {

    }
    Frame *operator()(Frame *f) {
//        usleep(30 * 1000);
        FrameBatch fb(f->id());
        fb.AddFrame(f, false);
        Operation op;
        op.Set(OPERATION_VEHICLE_DETECT);
        f->set_operation(op);
        cout << "QUEUE load:" << QueueLoad() << endl;
        if(f->id() % 2 == 0){
            processor_->Update(&fb);
        }

        f->set_status(FRAME_STATUS_DETECTED, false);
        f->set_status(FRAME_STATUS_ABLE_TO_DISPLAY, false);
        cout << "Detect frame finished: " << f->id() << endl;

    }
protected:
    Processor *processor_;
};
}


#endif //PROJECT_SIMPLE_PARALLEL_PROCESSOR_H_H
