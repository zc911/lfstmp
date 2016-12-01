/*
 * vehicle_classifier_processor.h
 *
 *  Created on: Apr 22, 2016
 *      Author: chenzhen
 */

#ifndef NON_MOTOR_VEHICLE_CLASSIFIER_PROCESSOR_H_
#define NON_MOTOR_VEHICLE_CLASSIFIER_PROCESSOR_H_

#include "processor/processor.h"
#include "algorithm_factory.h"
#include "processor_helper.h"
namespace dg {

class NonMotorVehicleClassifierProcessor: public Processor {
public:

    NonMotorVehicleClassifierProcessor();

    ~NonMotorVehicleClassifierProcessor();

protected:
    virtual bool process(Frame *frame) {
        return false;
    }
    virtual bool process(FrameBatch *frameBatch);
    virtual bool beforeUpdate(FrameBatch *frameBatch);
    virtual bool RecordFeaturePerformance();
private:
    void vehiclesResizedMat(FrameBatch *frameBatch);
    dgvehicle::INonMotorVehicleClassifier *nonMotorVehicleClassifier = nullptr;
    vector<Object *> objs_;
    vector<Mat> images_;
};

}

#endif /* VEHICLE_CLASSIFIER_PROCESSOR_H_ */
