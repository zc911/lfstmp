/*
 * vehicle_belt_classifier_processor.h
 *
 *  Created on: Sep 6, 2016
 *      Author: jiajaichen
 */

#ifndef SRC_PROCESSOR_VEHICLE_BELT_CLASSIFIER_PROCESSOR_H_
#define SRC_PROCESSOR_VEHICLE_BELT_CLASSIFIER_PROCESSOR_H_
#include "processor/processor.h"
//#include "alg/detector/window_caffe_ssd_detector.h"
//#include "alg/classification/belt_classifier.h"
#include "algorithm_factory.h"
#include "model/alg_config.h"

namespace dg {

class VehicleBeltClassifierProcessor: public Processor {
public:

    VehicleBeltClassifierProcessor(VehicleBeltConfig &bConfig, bool drive);
    ~VehicleBeltClassifierProcessor();
protected:
    virtual bool process(Frame *frame) {
        return false;
    }

    virtual bool process(FrameBatch *frameBatch);

    virtual bool beforeUpdate(FrameBatch *frameBatch);
    virtual bool RecordFeaturePerformance();


private:
    dgvehicle::AlgorithmProcessor *belt_classifier_ = NULL;

    vector<Object *> objs_;
    vector<vector<Rect> > fobs_;
    vector<vector<float> >params_;
    vector<Mat> images_;
    bool is_driver = true;
    float threshold_;
    int marker_target_min_;
    int marker_target_max_;
};

}

#endif /* SRC_PROCESSOR_VEHICLE_MARKER_CLASSIFIER_PROCESSOR_H_ */
