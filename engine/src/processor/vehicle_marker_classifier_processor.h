/*
 * vehicle_marker_classifier_processor.h
 *
 *  Created on: Apr 26, 2016
 *      Author: jiajaichen
 */

#ifndef SRC_PROCESSOR_VEHICLE_MARKER_CLASSIFIER_PROCESSOR_H_
#define SRC_PROCESSOR_VEHICLE_MARKER_CLASSIFIER_PROCESSOR_H_

#include "processor/processor.h"
#include "alg/marker_caffe_classifier.h"
#include "alg/window_caffe_detector.h"

namespace dg {

class VehicleMarkerClassifierProcessor : public Processor {
 public:
    VehicleMarkerClassifierProcessor(
            WindowCaffeDetector::WindowCaffeConfig & wConfig,
            MarkerCaffeClassifier::MarkerConfig &mConfig);

    ~VehicleMarkerClassifierProcessor();

 protected:
    virtual bool process(Frame *frame) {
        return false;
    }

    virtual bool process(FrameBatch *frameBatch);

    virtual bool beforeUpdate(FrameBatch *frameBatch);
    virtual bool RecordFeaturePerformance();



 private:
    MarkerCaffeClassifier *classifier_;
    WindowCaffeDetector *detector_;
    vector<Object *> objs_;
    vector<Mat> images_;
    vector<Mat> resized_images_;
    int window_target_min_;
    int window_target_max_;
    int marker_target_min_;
    int marker_target_max_;
}
;

}

#endif /* SRC_PROCESSOR_VEHICLE_MARKER_CLASSIFIER_PROCESSOR_H_ */
