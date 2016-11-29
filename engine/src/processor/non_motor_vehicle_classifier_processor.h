/*
 * vehicle_classifier_processor.h
 *
 *  Created on: Apr 22, 2016
 *      Author: chenzhen
 */

#ifndef NON_MOTOR_VEHICLE_CLASSIFIER_PROCESSOR_H_
#define NON_MOTOR_VEHICLE_CLASSIFIER_PROCESSOR_H_

#include "processor/processor.h"
#include "alg/non_motor_vehicle_classifier.h"
#include "processor_helper.h"
namespace dg {

class NonMotorVehicleClassifierProcessor: public Processor {
public:

    NonMotorVehicleClassifierProcessor(
        NonMotorVehicleClassifier::NonMotorVehicleConfig &nmConfig);

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
    NonMotorVehicleClassifier *nonMotorVehicleClassifier = NULL;
    vector<Object *> objs_;
    vector<Mat> images_;
};

}

#endif /* VEHICLE_CLASSIFIER_PROCESSOR_H_ */
