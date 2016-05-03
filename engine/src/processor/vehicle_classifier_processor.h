/*
 * vehicle_classifier_processor.h
 *
 *  Created on: Apr 22, 2016
 *      Author: chenzhen
 */

#ifndef VEHICLE_CLASSIFIER_PROCESSOR_H_
#define VEHICLE_CLASSIFIER_PROCESSOR_H_

#include "processor/processor.h"
#include "alg/vehicle_caffe_classifier.h"
#include "processor_helper.h"
namespace dg {

class VehicleClassifierProcessor : public Processor {
 public:

    VehicleClassifierProcessor(const vector< VehicleCaffeClassifier::VehicleCaffeConfig> &configs);

    ~VehicleClassifierProcessor();

    virtual void Update(Frame *frame) {

    }

    virtual void Update(FrameBatch *frameBatch);

    virtual void beforeUpdate(FrameBatch *frameBatch);
    virtual bool checkStatus(Frame *frame);
 protected:
    vector<Mat> vehicles_resized_mat(FrameBatch *frameBatch);
 private:
    vector<VehicleCaffeClassifier*> classifiers_;
    vector<Object *> objs_;
    vector<Mat> images_;

};

}

#endif /* VEHICLE_CLASSIFIER_PROCESSOR_H_ */
