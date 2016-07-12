/*
 * witness_engine.h
 *
 *  Created on: Apr 22, 2016
 *      Author: chenzhen
 */

#ifndef WITNESS_ENGINE_H_
#define WITNESS_ENGINE_H_

#include "config.h"
#include "fs_util.h"
#include "simple_engine.h"
#include "processor/processor.h"
#include "engine_config_value.h"

namespace dg {

class WitnessEngine: public SimpleEngine {

public:
    WitnessEngine(const Config &config);
    ~WitnessEngine();
    virtual void Process(FrameBatch *frame);

private:
    void init(const Config &config);
    void initFeatureOptions(const Config &config);

    void recordPerformance();
    Processor *vehicle_processor_;
    Processor *face_processor_;

    bool is_init_;

    bool enable_vehicle_;
    bool enable_vehicle_detect_;

    bool enable_vehicle_type_;
    bool enable_vehicle_color_;
    bool enable_vehicle_plate_;
    bool enable_vehicle_plate_gpu_;
    bool enable_vehicle_marker_;
    bool enable_vehicle_feature_vector_;
    bool enable_vehicle_pedestrian_attr_;

    bool enable_face_;
    bool enable_face_detect_;

    bool enable_face_feature_vector_;
    bool isWarmuped_ = true;

};

}

#endif /* WITNESS_ENGINE_H_ */
