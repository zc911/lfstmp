/*
 * witness_engine.h
 *
 *  Created on: Apr 22, 2016
 *      Author: chenzhen
 */

#ifndef WITNESS_ENGINE_H_
#define WITNESS_ENGINE_H_

#include "config.h"
#include "simple_engine.h"
#include "processor/processor.h"
#include "processor/vehicle_multi_type_detector_processor.h"
#include "processor/vehicle_classifier_processor.h"

namespace dg {

class WitnessEngine : SimpleEngine {

 public:
    WitnessEngine(const Config &config);
    ~WitnessEngine();
    virtual void Process(Frame *frame);

 private:
    void init(const Config &config);
    void initFeatureOptions(const Config &config);

    Processor *vehicle_processor_;
    Processor *face_processor_;

    bool is_init_;

    bool enable_vehicle_;
    bool enable_vehicle_type_;
    bool enable_vehicle_color_;
    bool enable_vehicle_plate_;
    bool enable_vehicle_plate_enhance_;
    bool enable_vehicle_marker_;
    bool enable_vehicle_feature_vector_;

    bool enable_face_;

};

}

#endif /* WITNESS_ENGINE_H_ */
