/*
 * vehicle_belt_classifier_processor.h
 *
 *  Created on: Sep 6, 2016
 *      Author: jiajaichen
 */

#ifndef SRC_PROCESSOR_VEHICLE_PHONE_DETECTOR_PROCESSOR_H_
#define SRC_PROCESSOR_VEHICLE_PHONE_DETECTOR_PROCESSOR_H_
#include "processor/processor.h"
//#include "alg/detector/phone_caffe_ssd_detector.h"
#include "algorithm_factory.h"
namespace dg {

class VehiclePhoneClassifierProcessor: public Processor {
public:

    VehiclePhoneClassifierProcessor(float threshold);
    ~VehiclePhoneClassifierProcessor();
protected:
    virtual bool process(Frame *frame) {
        return false;
    }

    virtual bool process(FrameBatch *frameBatch);

    virtual bool beforeUpdate(FrameBatch *frameBatch);
    virtual bool RecordFeaturePerformance();


private:
    dgvehicle::AlgorithmProcessor *detector_ = NULL;
    float threshold_;
    vector<Object *> objs_;
    vector<Mat> images_;
    vector<Detection> detections_;
//    int marker_target_min_;
//    int marker_target_max_;
};

}

#endif /* SRC_PROCESSOR_VEHICLE_MARKER_CLASSIFIER_PROCESSOR_H_ */
