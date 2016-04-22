/*
 * witness_engine.h
 *
 *  Created on: Apr 22, 2016
 *      Author: chenzhen
 */

#ifndef WITNESS_ENGINE_H_
#define WITNESS_ENGINE_H_

#include "simple_engine.h"
#include "processor/processor.h"
#include "processor/vehicle_multi_type_detector_processor.h"
#include "processor/vehicle_classifier_processor.h"

namespace dg {

class WitnessEngine : SimpleEngine {
 public:
    WitnessEngine() {
        processor_ = NULL;
        is_init_ = false;
    }
    ~WitnessEngine() {

    }

    virtual void Process(Frame *frame) {
        if (!is_init_) {
            init();
        }
        processor_->Update(frame);
    }

 private:
    void init() {
        processor_ = new VehicleMultiTypeDetectorProcessor();
        processor_->SetNextProcessor(new VehicleClassifierProcessor());
        is_init_ = true;
    }

    Processor *processor_;
    bool is_init_;
};

}

#endif /* WITNESS_ENGINE_H_ */
