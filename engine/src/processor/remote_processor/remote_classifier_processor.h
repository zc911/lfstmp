//
// Created by chenzhen on 7/27/16.
//

#ifndef PROJECT_REMOTE_CLASSIFIER_PROCESSOR_H_H
#define PROJECT_REMOTE_CLASSIFIER_PROCESSOR_H_H

#include "remote_basic_processor.h"

namespace dg {
/// This class invoke witness service to classify objects

// TODO Implements
class RemoteClassifierProcessor: public RemoteBasicProcessor {
public:
    bool process(Frame *frame) override;
    bool process(FrameBatch *frame) override;
};
}

#endif //PROJECT_REMOTE_CLASSIFIER_PROCESSOR_H_H
