/*
 * vehicle_detector_processor.h
 *
 *  Created on: 13/04/2016
 *      Author: chenzhen
 */

#ifndef VEHICLE_DETECTOR_PROCESSOR_H_
#define VEHICLE_DETECTOR_PROCESSOR_H_

#include <glog/logging.h>
#include "processor.h"
#include "alg/vehicle_multi_type_detector.h"
#include "util/debug_util.h"

namespace dg {

class VehicleMultiTypeDetectorProcessor : public Processor {
 public:

    VehicleMultiTypeDetectorProcessor(
            const VehicleMultiTypeDetector::VehicleMultiTypeConfig &config);

    ~VehicleMultiTypeDetectorProcessor();

    virtual void Update(Frame *frame) {

    }

    void Update(FrameBatch *frameBatch);

    void beforeUpdate(FrameBatch *frameBatch);
    bool checkStatus(Frame *frame);

 private:
    VehicleMultiTypeDetector *detector_;
    int base_id_;

};

}
#endif /* VEHICLE_DETECTOR_PROCESSOR_H_ */
