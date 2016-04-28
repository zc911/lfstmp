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

namespace dg {

class VehicleColorProcessor : public Processor {
 public:
    VehicleColorProcessor();
    ~VehicleColorProcessor();


    virtual void Update(FrameBatch *frameBatch);

    virtual bool checkOperation(Frame *frame);

    virtual bool checkStatus(Frame *frame) ;

protected:
    vector<Mat > vehicles_resized_mat(FrameBatch *frameBatch);
private:
    VehicleCaffeClassifier *classifier_;
    vector<Object *> objs_;
};

}

#endif /* SRC_PROCESSOR_VEHICLE_COLOR_PROCESSOR_H_ */
