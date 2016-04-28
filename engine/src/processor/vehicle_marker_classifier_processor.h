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
    VehicleMarkerClassifierProcessor();

    ~VehicleMarkerClassifierProcessor();

    virtual void Update(FrameBatch *frameBatch);

    virtual void beforeUpdate(FrameBatch *frameBatch);
    virtual bool checkStatus(Frame *frame);
 protected:

 private:
    MarkerCaffeClassifier *classifier_;
    WindowCaffeDetector *detector_;
    vector<Object *> objs_;
    vector<Mat> images_;
    vector<Mat> resized_images_;

};

}

#endif /* SRC_PROCESSOR_VEHICLE_MARKER_CLASSIFIER_PROCESSOR_H_ */
