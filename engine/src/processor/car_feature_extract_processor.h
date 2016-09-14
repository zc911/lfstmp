/*
 * car_feature_extract_processor.h
 *
 *  Created on: Apr 27, 2016
 *      Author: chenzhen
 */

#ifndef CAR_FEATURE_EXTRACT_PROCESSOR_H_
#define CAR_FEATURE_EXTRACT_PROCESSOR_H_

#include "alg/feature/car_feature_extractor.h"
#include "model/frame.h"
#include "model/model.h"
#include "processor.h"

namespace dg {

class CarFeatureExtractProcessor: public Processor {
public:
    CarFeatureExtractProcessor();
    virtual ~CarFeatureExtractProcessor();

protected:
    virtual bool process(Frame *frame) {
        return false;
    }
    virtual bool process(FrameBatch *frameBatch);
    virtual bool beforeUpdate(FrameBatch *frameBatch);
    virtual bool RecordFeaturePerformance();


private:
    void extract(vector<Object *> &objs);

    CarFeatureExtractor *extractor_=NULL;
    vector<Object *> vehicle_to_processed_;
};

} /* namespace dg */

#endif /* CAR_FEATURE_EXTRACT_PROCESSOR_H_ */
