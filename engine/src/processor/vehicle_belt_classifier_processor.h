/*
 * vehicle_belt_classifier_processor.h
 *
 *  Created on: Sep 6, 2016
 *      Author: jiajaichen
 */

#ifndef SRC_PROCESSOR_VEHICLE_BELT_CLASSIFIER_PROCESSOR_H_
#define SRC_PROCESSOR_VEHICLE_BELT_CLASSIFIER_PROCESSOR_H_
#include "processor/processor.h"
#include "algorithm_factory.h"

namespace dg {

class VehicleBeltClassifierProcessor: public Processor {
public:
    enum BeltLable{
        BELT_LABLE_NO = 0,
        BELT_NOT_SURE = 1,
        BELT_YES = 2
    };

    VehicleBeltClassifierProcessor(float threshold, bool drive);
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
    vector<vector<float> >params_;
    vector<Mat> images_;
    bool is_driver_;
    float threshold_;
};

}

#endif /* SRC_PROCESSOR_VEHICLE_MARKER_CLASSIFIER_PROCESSOR_H_ */
