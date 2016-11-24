/*============================================================================
 * File Name   : pedestrian_classifier_processor.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : Jul 1, 2016 8:42:35 AM
 * Description :
 * ==========================================================================*/
#ifndef SRC_PROCESSOR_PEDESTRIAN_CLASSIFIER_PROCESSOR_H_
#define SRC_PROCESSOR_PEDESTRIAN_CLASSIFIER_PROCESSOR_H_

//#include "alg/classification/pedestrian_classifier.h"
#include "processor/processor.h"
#include "algorithm_factory.h"

namespace dg {

class PedestrianClassifierProcessor: public Processor {

public:
    PedestrianClassifierProcessor();
    virtual ~PedestrianClassifierProcessor();

protected:
    virtual bool process(Frame *frame) {
        return false;
    }

    virtual bool process(FrameBatch *frameBatch);

    virtual bool beforeUpdate(FrameBatch *frameBatch);
    virtual bool RecordFeaturePerformance();

private:
    dgvehicle::IPedestrianClassifier *classifier_ = NULL;
    vector<Object *> objs_;
    vector<Mat> images_;
};
}

#endif /* SRC_PROCESSOR_PEDESTRIAN_CLASSIFIER_PROCESSOR_H_ */
