/*
 * vehicle_detector_processor.h
 *
 *  Created on: 13/04/2016
 *      Author: chenzhen
 */

#ifndef VEHICLE_DETECTOR_PROCESSOR_H_
#define VEHICLE_DETECTOR_PROCESSOR_H_

#include <vector>
#include <glog/logging.h>
#include "util/debug_util.h"
#include "processor.h"
#include "algorithm_factory.h"

using namespace std;
namespace dg {

class VehicleMultiTypeDetectorProcessor: public Processor {
public:

    VehicleMultiTypeDetectorProcessor(bool car_only, bool accelate, bool enable_demo);

    ~VehicleMultiTypeDetectorProcessor();

protected:

    virtual bool process(Frame *frame) {
        return false;
    }

    virtual bool process(FrameBatch *frameBatch);


    bool beforeUpdate(FrameBatch *frameBatch);
    virtual bool RecordFeaturePerformance();
    virtual string processorName() {
        return "VehicleMultiTypeDetectorProcessor";
    }

private:

    dgvehicle::AlgorithmProcessor *vehicle_detector_ = NULL;
    dgvehicle::AlgorithmProcessor *car_only_detector_ = NULL;
    dgvehicle::AlgorithmProcessor *car_only_confirm_ = NULL;
    int base_id_;
    bool car_only_;
    float threshold_;

};

}
#endif /* VEHICLE_DETECTOR_PROCESSOR_H_ */
