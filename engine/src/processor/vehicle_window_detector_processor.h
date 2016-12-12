/*
 * vehicle_window_detector_processor.h
 *
 *  Created on: Sep 5, 2016
 *      Author: jiajaichen
 */

#ifndef SRC_PROCESSOR_VEHICLE_WINDOW_CLASSIFIER_PROCESSOR_H_
#define SRC_PROCESSOR_VEHICLE_WINDOW_CLASSIFIER_PROCESSOR_H_
#include "processor/processor.h"
#include "algorithm_factory.h"
namespace dg {

class VehicleWindowDetectorProcessor: public Processor {
public:
    VehicleWindowDetectorProcessor();
    ~VehicleWindowDetectorProcessor();
protected:
    virtual bool process(Frame *frame) {
        return false;
    }
    virtual bool process(FrameBatch *frameBatch);
    virtual bool beforeUpdate(FrameBatch *frameBatch);
    virtual bool RecordFeaturePerformance();

private:
    dgvehicle::AlgorithmProcessor *ssd_window_detector_ = NULL;
    vector<Object *> objs_;
    vector<Mat> images_;
    vector<Mat> resized_images_;
 //   int window_target_min_;
 //   int window_target_max_;
};

}

#endif /* SRC_PROCESSOR_VEHICLE_MARKER_CLASSIFIER_PROCESSOR_H_ */
