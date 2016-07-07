//
// Created by chenzhen on 7/7/16.
//

#ifndef PROJECT_SIMPLE_PARALLEL_PROCESSOR_H_H
#define PROJECT_SIMPLE_PARALLEL_PROCESSOR_H_H
#include "processor/processor.h"
#include "processor/parallel/parallel_node.h"

namespace dg {
class BasicParallelProcessor: public ParallelProcessorNode {
public:
    BasicParallelProcessor(Processor *proc) : processor_(proc) {

    }
    Frame *operator()(Frame *f) {
        usleep(30 * 1000);
        f->set_status(FRAME_STATUS_DETECTED);
        f->set_status(FRAME_STATUS_ABLE_TO_DISPLAY);

    }
protected:
    Processor *processor_;
};
}


#endif //PROJECT_SIMPLE_PARALLEL_PROCESSOR_H_H
