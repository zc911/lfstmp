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
namespace dg {

class VehicleClassifierProcessor : public Processor {
 public:

    VehicleClassifierProcessor();

    ~VehicleClassifierProcessor();

    virtual void Update(FrameBatch *frameBatch);

    virtual void beforeUpdate(FrameBatch *frameBatch);
    virtual bool checkStatus(Frame *frame);
 protected:
    vector<Mat> vehicles_resized_mat(FrameBatch *frameBatch);
 private:
    int classifiers_size_;
    vector<VehicleCaffeClassifier*> classifiers_;
    vector<Object *> objs_;
    vector<Mat> images_;

};

}

#endif /* VEHICLE_CLASSIFIER_PROCESSOR_H_ */
