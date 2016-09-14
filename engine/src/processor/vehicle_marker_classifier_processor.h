/*
 * vehicle_marker_classifier_processor.h
 *
 *  Created on: Apr 26, 2016
 *      Author: jiajaichen
 */

#ifndef SRC_PROCESSOR_VEHICLE_MARKER_CLASSIFIER_PROCESSOR_H_
#define SRC_PROCESSOR_VEHICLE_MARKER_CLASSIFIER_PROCESSOR_H_
#include "processor/processor.h"
#include "alg/classification/marker_caffe_classifier.h"
#include "alg/detector/window_caffe_detector.h"
#include "alg/detector/window_caffe_ssd_detector.h"
#include "alg/detector/marker_caffe_ssd_detector.h"
namespace dg {

class VehicleMarkerClassifierProcessor: public Processor {
public:
    VehicleMarkerClassifierProcessor(
        WindowCaffeDetector::WindowCaffeConfig &wConfig,
        MarkerCaffeClassifier::MarkerConfig &mConfig);
    VehicleMarkerClassifierProcessor(VehicleCaffeDetectorConfig &wConfig,
                                     VehicleCaffeDetectorConfig &mConfig,bool isVisualization);
    ~VehicleMarkerClassifierProcessor();
protected:
    virtual bool process(Frame *frame) {
        return false;
    }

    virtual bool process(FrameBatch *frameBatch);

    virtual bool beforeUpdate(FrameBatch *frameBatch);
    virtual bool RecordFeaturePerformance();


private:
    MarkerCaffeClassifier *classifier_=NULL;
    WindowCaffeDetector *detector_=NULL;
    WindowCaffeSsdDetector *ssd_window_detector_=NULL;
    MarkerCaffeSsdDetector *ssd_marker_detector_=NULL;
    vector<Object *> objs_;
    vector<Mat> images_;
    vector<Mat> resized_images_;
    bool isSsd=true;
    int window_target_min_;
    int window_target_max_;
    int marker_target_min_;
    int marker_target_max_;
    bool isVisualization_=false;
    vector<Scalar> color_;
};

}

#endif /* SRC_PROCESSOR_VEHICLE_MARKER_CLASSIFIER_PROCESSOR_H_ */
