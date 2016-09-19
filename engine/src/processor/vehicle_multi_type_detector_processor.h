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
#include "alg/detector/vehicle_caffe_detector.h"
#include "alg/detector/car_only_confirm_caffe_detector.h"
#include "alg/detector/car_only_caffe_detector.h"

using namespace std;
namespace dg {

class VehicleMultiTypeDetectorProcessor: public Processor {
public:

    VehicleMultiTypeDetectorProcessor(
        const VehicleCaffeDetectorConfig &config);

    ~VehicleMultiTypeDetectorProcessor();

protected:

    virtual bool process(Frame *frame) {
        return false;
    }

    virtual bool process(FrameBatch *frameBatch);


    bool beforeUpdate(FrameBatch *frameBatch);
    virtual bool RecordFeaturePerformance();

private:

    VehicleCaffeDetector *vehicle_detector_ = NULL;
    CarOnlyCaffeDetector *car_only_detector_ = NULL;
    CarOnlyConfirmCaffeDetector *car_only_confirm_ = NULL;
    VehicleCaffeDetectorConfig config_;
    int base_id_;

};

}
#endif /* VEHICLE_DETECTOR_PROCESSOR_H_ */
