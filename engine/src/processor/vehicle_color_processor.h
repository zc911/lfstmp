/*
 * vehicle_color_processor.h
 *
 *  Created on: Apr 26, 2016
 *      Author: jiajiachen
 */

#ifndef SRC_PROCESSOR_VEHICLE_COLOR_PROCESSOR_H_
#define SRC_PROCESSOR_VEHICLE_COLOR_PROCESSOR_H_

#include "processor/processor.h"
#include "alg/vehicle_caffe_classifier.h"
#include "processor_helper.h"

namespace dg {

class VehicleColorProcessor : public Processor {
 public:
    VehicleColorProcessor(const vector< VehicleCaffeClassifier::VehicleCaffeConfig> &configs);
    ~VehicleColorProcessor();

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

#endif /* SRC_PROCESSOR_VEHICLE_COLOR_PROCESSOR_H_ */
