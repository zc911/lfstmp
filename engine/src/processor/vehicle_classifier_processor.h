/*
 * vehicle_classifier_processor.h
 *
 *  Created on: Apr 22, 2016
 *      Author: chenzhen
 */

#ifndef VEHICLE_CLASSIFIER_PROCESSOR_H_
#define VEHICLE_CLASSIFIER_PROCESSOR_H_

#include "processor/processor.h"
//#include "alg/classification/vehicle_caffe_classifier.h"
#include "processor_helper.h"
#include "algorithm_factory.h"

namespace dg {

class VehicleClassifierProcessor: public Processor {
 public:

    VehicleClassifierProcessor(string &mappingFilePath, bool encrypt = false);

    ~VehicleClassifierProcessor();

 protected:
    virtual bool process(Frame *frame) {
        return false;
    }
    virtual bool process(FrameBatch *frameBatch);
    virtual bool beforeUpdate(FrameBatch *frameBatch);
    virtual bool RecordFeaturePerformance();

 private:
    void vehiclesResizedMat(FrameBatch *frameBatch);
    void initIsHeadRepo(string &mapFilePath);

    vector<dgvehicle::AlgorithmProcessor *> classifiers_;
    vector<Object *> objs_;
    vector<Mat> images_;
    vector<unsigned short> vehicle_is_head_repo_;

};

}

#endif /* VEHICLE_CLASSIFIER_PROCESSOR_H_ */
