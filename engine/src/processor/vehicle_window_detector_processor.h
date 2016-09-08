/*
 * vehicle_window_detector_processor.h
 *
 *  Created on: Sep 5, 2016
 *      Author: jiajaichen
 */

#ifndef SRC_PROCESSOR_VEHICLE_WINDOW_CLASSIFIER_PROCESSOR_H_
#define SRC_PROCESSOR_VEHICLE_WINDOW_CLASSIFIER_PROCESSOR_H_
#include "processor/processor.h"
#include "alg/detector/window_caffe_ssd_detector.h"
namespace dg {

class VehicleWindowDetectorProcessor: public Processor {
public:
    VehicleWindowDetectorProcessor(const VehicleCaffeDetectorConfig &wConfig);
    ~VehicleWindowDetectorProcessor();
protected:
    virtual bool process(Frame *frame) {
        return false;
    }
    virtual bool process(FrameBatch *frameBatch);
    virtual bool beforeUpdate(FrameBatch *frameBatch);
    virtual bool RecordFeaturePerformance();

private:
    WindowCaffeSsdDetector *ssd_window_detector_=NULL;
    vector<Object *> objs_;
    vector<Mat> images_;
    vector<Mat> resized_images_;
    int window_target_min_;
    int window_target_max_;
};

}

#endif /* SRC_PROCESSOR_VEHICLE_MARKER_CLASSIFIER_PROCESSOR_H_ */
