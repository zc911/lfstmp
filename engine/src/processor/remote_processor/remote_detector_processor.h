//
// Created by chenzhen on 7/27/16.
//

#ifndef PROJECT_REMOTE_DETECTOR_PROCESSOR_H_H
#define PROJECT_REMOTE_DETECTOR_PROCESSOR_H_H
#include "remote_basic_processor.h"
namespace dg {

/// This class invoke witness service to detect objects
/// from a frame image

// TODO Implements
class RemoteDetectorProcessor: public RemoteBasicProcessor {
public:
    bool process(Frame *frame) override;
    bool process(FrameBatch *frame) override;
};

}

#endif //PROJECT_REMOTE_DETECTOR_PROCESSOR_H_H
