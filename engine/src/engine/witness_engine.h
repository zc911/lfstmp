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
#include "processor/vehicle_color_processor.h"
#include "processor/vehicle_marker_classifier_processor.h"
namespace dg {

class WitnessEngine : SimpleEngine {
 public:
    WitnessEngine() {
        processor_ = NULL;
        is_init_ = false;
        init();
    }
    ~WitnessEngine() {

    }

    virtual void Process(Frame *frame) {
        if (!is_init_) {
            init();
        }
        processor_->Update(frame);
    }
    virtual void Process(FrameBatch *framebatch) {
        if (!is_init_) {
            init();
        }
        processor_->Update(framebatch);
    }
 private:
    void init() {
        processor_ = new VehicleMultiTypeDetectorProcessor();
    //    processor_->SetNextProcessor(new VehicleClassifierProcessor());
        processor_->SetNextProcessor(new VehicleMarkerClassifierProcessor());

        is_init_ = true;
    }

    Processor *processor_;
    bool is_init_;
};

}

#endif /* WITNESS_ENGINE_H_ */
